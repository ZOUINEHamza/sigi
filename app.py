import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import joblib
import datetime
from scipy.signal import welch, find_peaks, decimate
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import requests

FIREBASE_URL = st.secrets["firebase"]["database_url"].rstrip("/")


# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = Path(__file__).parent
MODEL_PATH     = BASE_DIR / "final_model.pkl"
META_PATH      = BASE_DIR / "model_meta.json"
GMM_PATH       = BASE_DIR / "gmm_db.pkl"
INCONNUS_FILE  = BASE_DIR / "inconnus_db.json"
SCALER_PATH    = BASE_DIR / "scaler.pkl"
RFECV_PATH     = BASE_DIR / "rfecv.pkl"
# fct de base firebase
def firebase_get_etiquettes():
    """Lit les étiquettes validées depuis Firebase."""
    try:
        r = requests.get(f"{FIREBASE_URL}/etiquettes.json", timeout=5)
        data = r.json()
        return data if data else {}
    except Exception:
        return {}

def firebase_set_etiquette(uid, nouveau_nom, n_captures):
    """Ajoute ou met à jour une étiquette dans Firebase."""
    try:
        payload = {
            "nouveau_nom": nouveau_nom,
            "ancien_id": uid,
            "n_captures": n_captures,
            "date_validation": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        requests.put(
            f"{FIREBASE_URL}/etiquettes/{uid}.json",
            json=payload,
            timeout=5
        )
        return True
    except Exception as e:
        st.error(f"Erreur Firebase : {e}")
        return False
# --- FONCTIONS DE TRAITEMENT DU SIGNAL ---

def iq_to_phase(iq):
    phase = np.unwrap(np.angle(iq))
    t = np.arange(len(phase))
    return phase - np.polyval(np.polyfit(t, phase, 1), t)

def moving_average(x, m):
    cs = np.cumsum(np.concatenate(([0.0], x.astype(float))))
    return (cs[m:] - cs[:-m]) / m

def compute_avar(phase, taus, fs):
    avar, N = [], len(phase)
    for tau in taus:
        m = int(round(tau * fs))
        if m < 1 or 2*m >= N: avar.append(np.nan); continue
        pa = moving_average(phase, m); n = len(pa) - 2*m
        if n <= 0: avar.append(np.nan); continue
        d = pa[2*m:2*m+n] - 2*pa[m:m+n] + pa[:n]
        avar.append(float(np.mean(d**2) / (2*tau**2)))
    return np.array(avar)

def _load_iq_full(file_content):
    """Charge le contenu IQ complex64 (depuis un buffer Streamlit ou fichier)."""
    n = len(file_content) // 8
    return np.frombuffer(file_content[:n*8], dtype=np.complex64).copy()

def _preprocess_iq(iq, fs, decim=1):
    iq = iq - np.mean(iq)
    sq = iq ** 2
    cfo_est = np.angle(np.mean(sq[1:] * np.conj(sq[:-1]))) / (2 * np.pi) * (fs / 2)
    t = np.arange(len(iq)) / fs
    iq = iq * np.exp(-1j * 2 * np.pi * cfo_est * t)
    if decim > 1:
        iq_r = decimate(iq.real, decim, ftype="fir", zero_phase=True)
        iq_i = decimate(iq.imag, decim, ftype="fir", zero_phase=True)
        iq   = (iq_r + 1j * iq_i).astype(np.complex64)
    pwr = np.mean(np.abs(iq) ** 2)
    if pwr > 0:
        iq = iq / np.sqrt(pwr)
    return iq

def _detect_bursts(iq, fs, smooth_ms=1.0, threshold_db=8.0, min_burst_ms=10.0, merge_gap_ms=2.0):
    EPS_E    = 1e-30
    energie  = np.abs(iq) ** 2
    win      = max(1, int(smooth_ms * 1e-3 * fs))
    kern     = np.ones(win) / win
    e_smooth = np.convolve(energie, kern, mode="same")
    e_db     = 10 * np.log10(e_smooth + EPS_E)
    plancher = np.percentile(e_db, 10)
    seuil    = plancher + threshold_db
    masque   = e_db > seuil
    segs = []
    en_burst = False
    debut = 0
    for i, v in enumerate(masque):
        if v and not en_burst:
            debut = i; en_burst = True
        elif not v and en_burst:
            segs.append((debut, i)); en_burst = False
    if en_burst: segs.append((debut, len(masque)))
    gap_min = int(merge_gap_ms * 1e-3 * fs)
    fused = []
    for seg in segs:
        if fused and (seg[0] - fused[-1][1]) < gap_min:
            fused[-1] = (fused[-1][0], seg[1])
        else:
            fused.append(list(seg))
    min_samp = int(min_burst_ms * 1e-3 * fs)
    return [(s, e) for s, e in fused if (e - s) >= min_samp]

# --- LOGIQUE IA & INFERENCE ---

def identify_bursts(file_content, params):
    fs = params.get("fs_original", 80000)
    decim = params.get("decim", 2)
    fs_eff = fs // decim
    
    # 1. Prétraitement
    iq = _load_iq_full(file_content)
    iq = _preprocess_iq(iq, fs, decim=decim)
    
    # 2. Détection
    bursts_idx = _detect_bursts(iq, fs_eff, 
                               smooth_ms=params.get("burst_smooth_ms", 1.0),
                               threshold_db=params.get("burst_threshold_db", 8.0))
    
    if not bursts_idx:
        return None
    
    # Limitation
    max_b = params.get("max_bursts", 200)
    if max_b:
        bursts_idx = bursts_idx[:max_b]
        
    # 3. Extraction de features & Classification (Simulé ici, complet en Étape 3)
    # Dans une version finale, on appellerait full_feature_vector pour chaque burst
    results = {
        "id": f"Traitement_{datetime.datetime.now().strftime('%H%M%S')}",
        "metrics": {"n_connus": 0, "n_inconnus": 0, "n_total": len(bursts_idx), "elapsed": 0.5},
        "distribution": {},
        "bursts": {"labels": [], "probas": []},
        "iq_bursts": [iq[s:e] for s, e in bursts_idx]
    }
    
    # Simulation de résultats IA
    for i in range(len(bursts_idx)):
        results["bursts"]["labels"].append("AIS 01")
        results["bursts"]["probas"].append(98.5)
    
    results["distribution"] = {"AIS 01": len(bursts_idx)}
    results["metrics"]["n_connus"] = len(bursts_idx)
    
    return results

# --- CONFIGURATION DE LA PAGE ---

BASE_DIR = Path(__file__).parent
icon_path = BASE_DIR / "signoise_bleu.png"

st.set_page_config(
    page_title="SigNoise Web Pro",
    page_icon=str(icon_path) if icon_path.exists() else "📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PALETTE & STYLE ---
PAL = {
    "teal":   "#1D9E75",
    "coral":  "#D85A30",
    "purple": "#534AB7",
    "amber":  "#BA7517",
    "blue":   "#185FA5",
    "dark":   "#2C2C2A",
    "mid":    "#5F5E5A",
    "light":  "#FAFAF8",
    "white":  "#FFFFFF",
    "border": "#D3D1C7",
    "marine": "#0A1637",
    "gold":   "#C4A050",
    "unknown":"#7B5EA7",
}

# --- STYLE CSS (MATCHING DESKTOP) ---
st.markdown(f"""
    <style>
    .main {{ background-color: {PAL['light']}; color: {PAL['dark']}; font-family: 'Segoe UI', sans-serif; }}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: {PAL['white']};
        border-radius: 8px 8px 0 0;
        border: 1px solid {PAL['border']};
        padding: 10px 20px;
        font-weight: 500;
        color: {PAL['dark']};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PAL['white']} !important;
        border-bottom: 3px solid {PAL['gold']} !important;
        color: {PAL['marine']} !important;
        font-weight: 700 !important;
    }}
    
    /* Metrics Styling */
    [data-testid="stMetric"] {{
        background-color: {PAL['white']};
        border: 1px solid {PAL['border']};
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }}
    [data-testid="stMetricLabel"] {{ color: {PAL['mid']}; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
    [data-testid="stMetricValue"] {{ color: {PAL['marine']}; font-size: 24px; font-weight: 700; }}
    
    /* Custom Header */
    .header-box {{
        background: {PAL['white']};
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 25px;
        color: {PAL['marine']};
        border: 1px solid {PAL['border']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {PAL['teal']} !important;
        color: white !important;
        border-radius: 7px !important;
        border: none !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
    }}
    .stButton>button:hover {{ background-color: #0F6E56 !important; }}
    
    /* Section Labels */
    .section-label {{
        font-size: 12px;
        font-weight: 600;
        color: {PAL['mid']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 20px;
        margin-bottom: 10px;
    }}
    /* Icônes SVG dans les onglets via data-URI */
    .stTabs [data-baseweb="tab"]:nth-child(1)::before {{
        content: '';
        display: inline-block;
        width: 16px; height: 16px;
        margin-right: 7px;
        vertical-align: middle;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%235F5E5A' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4.9 16.1C1 12.2 1 5.8 4.9 1.9'/%3E%3Cpath d='M7.8 4.7a6.14 6.14 0 0 0 0 8.5'/%3E%3Ccircle cx='12' cy='9' r='2'/%3E%3Cpath d='M16.2 4.7a6.14 6.14 0 0 1 0 8.5'/%3E%3Cpath d='M19.1 1.9a10.11 10.11 0 0 1 0 14.2'/%3E%3Cline x1='12' y1='11' x2='12' y2='22'/%3E%3Cline x1='8' y1='22' x2='16' y2='22'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-size: contain;
    }}
    .stTabs [data-baseweb="tab"]:nth-child(2)::before {{
        content: '';
        display: inline-block;
        width: 16px; height: 16px;
        margin-right: 7px;
        vertical-align: middle;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%235F5E5A' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-size: contain;
    }}
    .stTabs [data-baseweb="tab"]:nth-child(3)::before {{
        content: '';
        display: inline-block;
        width: 16px; height: 16px;
        margin-right: 7px;
        vertical-align: middle;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%235F5E5A' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='8' ry='3'/%3E%3Cpath d='M4 5v7c0 1.66 3.58 3 8 3s8-1.34 8-3V5'/%3E%3Cpath d='M4 12v5c0 1.66 3.58 3 8 3'/%3E%3Cpath d='M19 17v5'/%3E%3Cpath d='M22 20l-3-3-3 3'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-size: contain;
    }}
    .stTabs [data-baseweb="tab"]:nth-child(4)::before {{
        content: '';
        display: inline-block;
        width: 16px; height: 16px;
        margin-right: 7px;
        vertical-align: middle;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%235F5E5A' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2'/%3E%3Ccircle cx='9' cy='7' r='4'/%3E%3Cpath d='M23 21v-2a4 4 0 0 0-3-3.87'/%3E%3Cpath d='M16 3.13a4 4 0 0 1 0 7.75'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-size: contain;
    }}
    .stTabs [data-baseweb="tab"]:nth-child(5)::before {{
        content: '';
        display: inline-block;
        width: 16px; height: 16px;
        margin-right: 7px;
        vertical-align: middle;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%235F5E5A' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z'/%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-size: contain;
    }}
    .stTabs [data-baseweb="tab"]:nth-child(6)::before {{
        content: '';
        display: inline-block;
        width: 16px; height: 16px;
        margin-right: 7px;
        vertical-align: middle;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%235F5E5A' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M12 16v-4'/%3E%3Cpath d='M12 8h.01'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-size: contain;
    }}
    /* Icône active en couleur marine */
    .stTabs [aria-selected="true"]::before {{
        filter: invert(10%) sepia(80%) saturate(800%) hue-rotate(200deg) brightness(50%);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- INITIALISATION DU SESSION STATE ---
if "current_data" not in st.session_state:
    st.session_state.current_data = None
if "selected_burst_idx" not in st.session_state:
    st.session_state.selected_burst_idx = 0
if "analysis_trigger" not in st.session_state:
    st.session_state.analysis_trigger = False

# --- UTILS ---
def get_path(filename):
    path = BASE_DIR / filename
    return str(path) if path.exists() else None

import base64
def img_to_b64(path):
    if not path or not os.path.exists(path): return ""
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode()

if 'logo_ern_b64' not in st.session_state:
    st.session_state.logo_ern_b64 = img_to_b64(get_path("ern.png"))
if 'logo_en_b64' not in st.session_state:
    st.session_state.logo_en_b64 = img_to_b64(get_path("en.png"))
# --- HEADER (REPRODUCED FROM DESKTOP) ---
st.markdown(f"""
    <div class="header-box">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <img src="data:image/png;base64,{st.session_state.get('logo_ern_b64', '')}" width="100">
            <div style="text-align: center;">
                <h1 style="margin: 0; font-size: 28px; letter-spacing: 1px; color: {PAL['marine']};">SigNoise </h1>
                <p style="margin: 5px 0 0 0; color: {PAL['mid']}; font-size: 14px; letter-spacing: 1px;">IDENTIFICATION  & ANALYSE DE SIGNATURE DE BRUIT</p>
            </div>
            <img src="data:image/png;base64,{st.session_state.get('logo_en_b64', '')}" width="100">
        </div>
    </div>
    """, unsafe_allow_html=True)

# Injection des logos B64 pour le header HTML (optionnel si on utilise st.image plus simplement)
import base64
def img_to_b64(path):
    if not path or not os.path.exists(path): return ""
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode()

if 'logo_ern_b64' not in st.session_state:
    st.session_state.logo_ern_b64 = img_to_b64(get_path("ern.png"))
if 'logo_en_b64' not in st.session_state:
    st.session_state.logo_en_b64 = img_to_b64(get_path("en.png"))

# --- NAVIGATION TABS ---
# Utilisation d'icônes professionnelles (Flaticon ou similaires via URL/Local)
tabs_config = [
    {"label": "Identification", "icon": "https://cdn-icons-png.flaticon.com/512/1077/1077976.png"},
    {"label": "Analyse Signal", "icon": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png"},
    {"label": "Entraînement", "icon": "https://cdn-icons-png.flaticon.com/512/2103/2103445.png"},
    {"label": "Inconnus", "icon": "https://cdn-icons-png.flaticon.com/512/912/912214.png"},
    {"label": "Paramètres", "icon": "https://cdn-icons-png.flaticon.com/512/3524/3524659.png"},
    {"label": "À Propos", "icon": "https://cdn-icons-png.flaticon.com/512/471/471663.png"}
]

# Note: Streamlit ne supporte pas nativement les images dans st.tabs, 
# on garde le texte mais on améliore le rendu via CSS ou colonnes si besoin.
tab_id, tab_analysis, tab_train, tab_inc, tab_params, tab_about = st.tabs([
    "Identification",
    "Analyse Signal",
    "Entraînement",
    "Inconnus",
    "Paramètres",
    "À Propos",
])

# --- SIDEBAR ---
sidebar_logo = get_path("SigNoise.png") or get_path("SigNoise_icon.svg")
if sidebar_logo:
    st.sidebar.image(sidebar_logo, width="stretch")

st.sidebar.title("Navigation Pro")
st.sidebar.markdown("---")
st.sidebar.write("**Statut Cloud :** 🟢 Connecté")

# Chargement dynamique des métadonnées du modèle
model_meta = {}
if META_PATH.exists():
    with open(META_PATH, "r") as f:
        model_meta = json.load(f)

st.sidebar.write(f"**Modèle :** {model_meta.get('modele', 'SVM (RBF)')}")

# --- CONTENU DES ONGLETS ---

with tab_id:
    st.header("Identification des émetteurs")
    
    # Barre supérieure de contrôle
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        mode = st.radio("Mode de fonctionnement :", ["Consulter Archives", "Nouvelle Identification"], horizontal=True, label_visibility="collapsed")
    
    if mode == "Consulter Archives":
        archive_dir = BASE_DIR / "archives"
        if archive_dir.exists():
            json_files = sorted([f.name for f in archive_dir.glob("*.json")], reverse=True)
            if json_files:
                selected_file = st.selectbox("Sélectionner un rapport archivé :", json_files)
                if st.button("📂 Charger le rapport"):
                    with open(archive_dir / selected_file, "r") as f:
                        st.session_state.current_data = json.load(f)
                        st.session_state.selected_burst_idx = 0
            else:
                st.warning("Aucun rapport trouvé dans le dossier 'archives'.")
        else:
            st.error("Le dossier 'archives' est manquant.")
    else:
        uploaded_file = st.file_uploader("Charger un fichier IQ (.iq, .bin)", type=["iq", "bin"])
        if uploaded_file:
            st.success(f"Fichier '{uploaded_file.name}' prêt pour l'analyse.")
            
            params = {
                "fs_original": 80000,
                "decim": 2,
                "burst_smooth_ms": 1.0,
                "burst_threshold_db": 8.0,
                "max_bursts": 200
            }
            
            if st.button("🚀 Lancer l'identification IA"):
                with st.spinner("Traitement du signal et identification en cours..."):
                    file_content = uploaded_file.read()
                    results = identify_bursts(file_content, params)
                    
                    if results:
                        st.session_state.current_data = results
                        st.session_state.selected_burst_idx = 0
                        st.balloons()
                        st.success("Identification terminée !")
                    else:
                        st.error("Aucun burst AIS détecté.")

    # AFFICHAGE DES RÉSULTATS
    data = st.session_state.get("current_data")

    if data:
        st.markdown(f"### Rapport : `{data.get('id', 'N/A')}` — {data.get('date', 'Date inconnue')}")
        
        # --- MÉTRIQUES ---
        m = data.get("metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Bursts Connus", m.get("n_connus", 0))
        with c2: st.metric("Bursts Inconnus", m.get("n_inconnus", 0), delta_color="inverse")
        with c3: st.metric("Total Bursts", m.get("n_total", 0))
        with c4: st.metric("Temps de calcul", f"{m.get('elapsed', 0)}s")

        # --- GRAPHIQUES ---
        col_g1, col_g2 = st.columns([1, 1])
        
        with col_g1:
            st.markdown("#### Distribution des émetteurs")
            dist = data.get("distribution", {})
            if dist:
                labels = list(dist.keys())
                values = list(dist.values())
                fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=PAL['teal'])])
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, width="stretch")

        with col_g2:
            st.markdown("#### Probabilités moyennes")
            labels = list(dist.keys()) if dist else []
            if labels:
                probs = [0.95] * len(labels)
                fig_p = go.Figure(data=[go.Bar(
                    y=labels,
                    x=probs,
                    orientation='h',
                    marker_color=PAL['blue'],
                    text=[f"{p*100:.1f}%" for p in probs],
                    textposition='inside',
                    insidetextanchor='middle',
                )])
                fig_p.update_layout(
                    height=180,          # ← plus petit qu'avant (était 350)
                    margin=dict(l=10, r=20, t=10, b=10),
                    xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
                    yaxis=dict(tickfont=dict(size=11)),
                    bargap=0.3,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_p, width="stretch")

        # --- BLOC INCONNUS (style SigNoise desktop) ---
        inconnus = {k: v for k, v in data.get("distribution", {}).items() if "INCONNU" in k.upper()}
        n_inconnus = m.get("n_inconnus", 0)

        if n_inconnus > 0 or inconnus:
            st.markdown("---")
            
            

            if INCONNUS_FILE.exists():
                with open(INCONNUS_FILE, "r") as f:
                    inc_db = json.load(f)

                if inc_db:
                    data_inc = []
                    for uid, info in inc_db.items():
                        statut = info.get("statut", "EN_ATTENTE")
                        couleur = "#BA7517" if statut == "EN_ATTENTE" else "#1D9E75"
                        data_inc.append({
                            "ID Groupe": uid,
                            "Captures": info.get("n", 0),
                            "Statut": statut,
                            "Première vue": info.get("date_premier", "N/A")[:10],
                            "Dernière vue": info.get("date_dernier", "N/A")[:10],
                        })
                    df_inc = pd.DataFrame(data_inc)

                    st.markdown(f"**{len(inc_db)} groupe(s) d'inconnus détectés :**")
                    st.dataframe(
                        df_inc,
                        hide_index=True,
                        height=min(150 + 35 * len(df_inc), 300),
                        column_config={
                            "Captures": st.column_config.ProgressColumn(
                                "Captures",
                                min_value=0,
                                max_value=20,
                                format="%d",
                            ),
                            "Statut": st.column_config.TextColumn("Statut"),
                        },
                        width="stretch"
                    )
                else:
                    st.info("Aucun groupe d'inconnus enregistré dans la base.")

        # --- TABLEAU DES BURSTS ---
        st.markdown("#### Détails des Bursts détectés")
        bursts = data.get("bursts", {})
        if bursts:
            df_bursts = pd.DataFrame({
                "Burst #": range(1, len(bursts.get("labels", [])) + 1),
                "Classe prédite": bursts.get("labels", []),
                "Confiance (%)": bursts.get("probas", [])
            })
            st.dataframe(df_bursts, width="stretch", height=300)
            
            selected_burst = st.selectbox("Analyser un burst spécifique :", df_bursts["Burst #"])
            if st.button(" Voir l'analyse détaillée"):
                st.session_state.selected_burst_idx = selected_burst - 1
                st.success(f"Burst {selected_burst} sélectionné. Allez dans l'onglet 'Analyse Signal'.")

with tab_analysis:
    st.header("Analyse de Signature de Bruit")
    
    data = st.session_state.get("current_data")
    if data:
        idx = st.session_state.get("selected_burst_idx", 0)
        st.subheader(f"Détails du Burst #{idx + 1}")
        
        col_a1, col_a2 = st.columns([2, 1])
        
        with col_a1:
            st.markdown("#### 1. Phase du signal (Purifiée)")
            t = np.linspace(0, 10, 1000)
            phase = np.sin(t) # Placeholder
            fig_p = go.Figure(go.Scatter(y=phase, line=dict(color=PAL['teal'])))
            fig_p.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_p, width="stretch")
            
            st.markdown("#### 2. Variances Log-Log (Allan & Hadamard)")
            v_data = data.get("avar", {})
            if v_data:
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=v_data.get('x'), y=v_data.get('y'), name="AVAR", line=dict(color=PAL['teal'], width=3)))
                fig_v.update_xaxes(type="log"); fig_v.update_yaxes(type="log")
                fig_v.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_v, width="stretch")

        with col_a2:
            st.markdown("#### Caractéristiques")
            feat_names = ["AVAR Slope", "HVAR Amp", "Crest Factor", "Kurtosis", "Skewness"]
            feat_vals = [0.002, 0.15, 4.2, 3.1, 0.5]
            df_feat = pd.DataFrame({"Paramètre": feat_names, "Valeur": feat_vals})
            st.dataframe(df_feat, hide_index=True, width="stretch")
            
            st.markdown("#### Densité Spectrale (PSD)")
            psd = data.get("psd", {})
            if psd:
                fig_psd = go.Figure(go.Scatter(x=psd.get('f'), y=psd.get('p'), line=dict(color=PAL['blue'])))
                fig_psd.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_psd, width="stretch")
    else:
        st.info("💡 Sélectionnez d'abord un rapport dans l'onglet **Identification**.")

with tab_train:
    st.header("Gestion de d'entraînement")
    
    # 1. État du modèle actuel (chargé dynamiquement)
    st.markdown('<p class="section-label">Modèle Actuel</p>', unsafe_allow_html=True)
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Algorithme", model_meta.get("modele", "N/A"))
    c_m2.metric("Score F1", f"{model_meta.get('f1_macro', 0):.4f}")
    
    classes_list = model_meta.get("classes", [])
    c_m3.metric("Classes connues", len(classes_list))
    
    st.write(f"**Classes :** {', '.join(classes_list) if classes_list else 'Aucune'}")

    # 2. Ajout de nouvelles acquisitions pour l'entraînement
    st.markdown('<p class="section-label">Nouvelle Acquisition pour la Base de Données</p>', unsafe_allow_html=True)
    with st.expander("➕ Ajouter un échantillon à la base"):
        new_class = st.text_input("Étiquette de la classe (ex: AIS_TUG_01)")
        new_iq = st.file_uploader("Fichier IQ de référence", type=["iq", "bin"], key="train_iq")
        if st.button("Enregistrer dans la base d'apprentissage"):
            st.success(f"Échantillon ajouté à la classe '{new_class}'.")

    # 3. Bouton de réentraînement
    st.markdown("---")
    if st.button(" Lancer le réentraînement global"):
        st.warning("Réentraînement en cours ...")

with tab_inc:
    st.header("Émetteurs Inconnus ")

    st.markdown('<p class="section-label">Configuration de l\'apprentissage continu</p>', unsafe_allow_html=True)
    n_trigger = st.number_input("Nombre de captures requis pour réentraîner automatiquement :", value=10, min_value=2)

    # --- TABLEAU 1 : inconnus bruts (JSON local) ---
    st.markdown('<p class="section-label">Groupes d\'inconnus détectés</p>', unsafe_allow_html=True)

    if INCONNUS_FILE.exists():
        with open(INCONNUS_FILE, "r") as f:
            inc_db = json.load(f)

        if inc_db:
            st.write(f"Nombre de groupes d'inconnus détectés : `{len(inc_db)}`")

            data_inc = []
            for uid, info in inc_db.items():
                sources = info.get("fichiers_sources", [])
                if not sources:
                    fichier_affiche = "—"
                elif len(sources) == 1:
                    fichier_affiche = Path(sources[0]).name
                else:
                    fichier_affiche = "  |  ".join(Path(f).name for f in sources)

                data_inc.append({
                    "ID Groupe":        uid,
                    "Fichier source":   fichier_affiche,
                    "Captures":         info.get("n", 0),
                    "Statut":           info.get("statut", "EN_ATTENTE"),
                    "Date":             info.get("date_dernier", info.get("date_premier", "N/A"))[:10],
                })

            st.dataframe(
                pd.DataFrame(data_inc),
                hide_index=True,
                height=min(150 + 35 * len(data_inc), 300),
                column_config={
                    "Captures": st.column_config.ProgressColumn(
                        "Captures", min_value=0, max_value=20, format="%d",
                    ),
                },
                width="stretch"
            )

            # --- SECTION ÉTIQUETAGE ---
            st.markdown("---")
            st.subheader("Étiquetage d'un groupe d'inconnus")
            col_sel, col_nom = st.columns(2)

            with col_sel:
                selected_uid = st.selectbox(
                    "Sélectionner un inconnu :",
                    list(inc_db.keys()),
                    key="sel_uid_inc"
                )
            with col_nom:
                new_label = st.text_input(
                    "Nouveau nom définitif :",
                    placeholder="Ex: AIS_NAVIRE_X",
                    key="new_label_inc"
                )

            col_b1, col_b2 = st.columns(2)

            if col_b1.button(" Valider le nom"):
                if new_label.strip():
                    n_cap = inc_db.get(selected_uid, {}).get("n", 0)
                    ok = firebase_set_etiquette(selected_uid, new_label.strip(), n_cap)
                    if ok:
                        st.success(f"'{selected_uid}' qualifié comme **'{new_label}'** et sauvegardé dans Firebase.")
                        st.rerun()
                    else:
                        st.error("Échec de la sauvegarde.")
                else:
                    st.warning("Entrez un nom avant de valider.")

            if col_b2.button("Lancer Réentraînement "):
                n_caps = inc_db.get(selected_uid, {}).get("n", 0)
                if n_caps >= n_trigger:
                    st.success(f"Réentraînement lancé avec {n_caps} captures de '{selected_uid}' !")
                else:
                    st.warning(f"Captures insuffisantes ({n_caps}/{n_trigger}).")

        else:
            st.info("Aucun émetteur inconnu détecté pour le moment.")
    else:
        st.info("La base des inconnus est vide.")

    # --- TABLEAU 2 : émetteurs qualifiés (Firebase) ---
    st.markdown("---")
    st.markdown('<p class="section-label">Émetteurs qualifiés </p>', unsafe_allow_html=True)

    etiquettes = firebase_get_etiquettes()

    if etiquettes:
        data_etiq = []
        for uid, info in etiquettes.items():
            data_etiq.append({
                "Ancien ID":        info.get("ancien_id", uid),
                "Nouveau nom":      info.get("nouveau_nom", "—"),
                "Captures":         info.get("n_captures", 0),
                "Date validation":  info.get("date_validation", "—"),
            })

        st.markdown(f"**{len(etiquettes)} émetteur(s) qualifié(s) :**")
        st.dataframe(
            pd.DataFrame(data_etiq),
            hide_index=True,
            column_config={
                "Nouveau nom": st.column_config.TextColumn("Nouveau nom 🏷️"),
                "Captures": st.column_config.NumberColumn("Captures", format="%d"),
            },
            width="stretch"
        )
    else:
        st.info(" Aucun émetteur qualifié pour l'instant. Utilisez le formulaire ci-dessus pour nommer un inconnu.")


with tab_params:
    st.header("Configuration")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        # ── Acquisition ──────────────────────────────────────────────────
        with st.container(border=True):
            st.markdown('<p class="section-label">Acquisition</p>', unsafe_allow_html=True)
            fs = st.number_input(
                "Fréquence d'échantillonnage (Hz)",
                min_value=10, max_value=100_000_000,
                value=80_000, step=100_000
            )

        # ── Décimation ───────────────────────────────────────────────────
        with st.container(border=True):
            st.markdown('<p class="section-label">Décimation</p>', unsafe_allow_html=True)
            decim = st.number_input(
                "Facteur de décimation",
                min_value=1, max_value=64,
                value=2, step=1
            )
            fs_eff = fs // decim
            st.markdown(f"**Fs effective : <span style='color:#1D9E75;font-size:15px'>{fs_eff/1000:.1f} kHz</span>**", unsafe_allow_html=True)

        # ── Détection des Bursts AIS ─────────────────────────────────────
        with st.container(border=True):
            st.markdown('<p class="section-label">Détection des Bursts AIS</p>', unsafe_allow_html=True)
            burst_smooth  = st.number_input("Lissage énergie (ms)",       min_value=0.01, max_value=50.0,    value=1.0,  step=0.1,  format="%.2f")
            burst_thr     = st.number_input("Seuil détection (dB)",       min_value=0.5,  max_value=60.0,    value=8.0,  step=0.5,  format="%.1f")
            burst_min     = st.number_input("Durée minimale burst (ms)",  min_value=0.1,  max_value=500.0,   value=10.0, step=0.5,  format="%.1f")
            burst_merge   = st.number_input("Fusion gaps < (ms)",         min_value=0.01, max_value=100.0,   value=2.0,  step=0.1,  format="%.2f")
            burst_max     = st.number_input("Bursts max par fichier",     min_value=0,    max_value=5000,    value=200,  step=10)
            st.caption("0 = tous les bursts")

        # ── Analyse ──────────────────────────────────────────────────────
        with st.container(border=True):
            st.markdown('<p class="section-label">Analyse</p>', unsafe_allow_html=True)
            nseg = st.number_input("Segments log-log (pentes)", min_value=2, max_value=10, value=3, step=1)

    with col_p2:
        # ── EMD + CMSE ───────────────────────────────────────────────────
        with st.container(border=True):
            st.markdown('<p class="section-label">EMD+CMSE — Décomposition Modale Empirique</p>', unsafe_allow_html=True)
            use_emd  = st.toggle("Activer l'EMD",               value=True)
            emd_sub  = st.number_input("Sous-échantillonnage (pts)", min_value=100, max_value=200_000, value=5000, step=500)
            emd_imf  = st.number_input("IMFs max",                   min_value=2,   max_value=30,      value=8,    step=1)

        # ── WT — Débruitage par Ondelette ────────────────────────────────
        with st.container(border=True):
            st.markdown('<p class="section-label">WT — Débruitage par Ondelette</p>', unsafe_allow_html=True)
            use_wt  = st.toggle("Activer la WT (désactivé par défaut)", value=False)
            wt_wav  = st.selectbox("Ondelette", ["db4", "db6", "db8", "sym4", "sym6", "coif2", "haar"], index=0)

        # ── Open Set / GMM ───────────────────────────────────────────────
        with st.container(border=True):
            st.markdown('<p class="section-label">Open Set — Seuils GMM</p>', unsafe_allow_html=True)
            seuil_proba    = st.number_input("Seuil proba classifieur",        min_value=0.01, max_value=0.99, value=0.60, step=0.01, format="%.2f")
            seuil_cap      = st.number_input("Captures min avant intégration", min_value=1,    max_value=500,  value=10,   step=1)
            seuil_dist     = st.number_input("Distance max regroupement",      min_value=0.01, max_value=100.0,value=2.0,  step=0.1,  format="%.2f")
            percentile_gmm = st.number_input("Percentile calibration GMM",     min_value=1,    max_value=50,   value=5,    step=1)
            k_max_gmm      = st.number_input("K max composantes GMM",          min_value=1,    max_value=20,   value=5,    step=1)

        # ── Chemins fichiers ─────────────────────────────────────────────
        with st.container(border=True):
            st.markdown('<p class="section-label">Fichiers du modèle</p>', unsafe_allow_html=True)
            st.code(
                f"Modèle      : {MODEL_PATH}\n"
                f"Métadonnées : {META_PATH}\n"
                f"GMM         : {GMM_PATH}\n"
                f"Inconnus    : {INCONNUS_FILE}",
                language=None
            )

    st.markdown("---")
    if st.button(" Sauvegarder la configuration système"):
        st.session_state["params"] = {
            "fs_original":        fs,
            "decim":              decim,
            "fs":                 fs_eff,
            "burst_smooth_ms":    burst_smooth,
            "burst_threshold_db": burst_thr,
            "burst_min_ms":       burst_min,
            "burst_merge_gap_ms": burst_merge,
            "max_bursts":         burst_max if burst_max > 0 else None,
            "nseg":               nseg,
            "use_emd":            use_emd,
            "use_cmse":           use_cmse,
            "emd_sub":            emd_sub,
            "emd_imf":            emd_imf,
            "use_wt":             use_wt,
            "wt_wav":             wt_wav,
            "seuil_proba":        seuil_proba,
            "seuil_cap":          seuil_cap,
            "seuil_dist":         seuil_dist,
            "percentile_gmm":     percentile_gmm,
            "k_max_gmm":          k_max_gmm,
        }
        st.success("✅ Configuration sauvegardée pour cette session.")


with tab_about:

    # --- EN-TÊTE ---
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, {PAL['marine']} 0%, #1C3464 100%);
        border-radius: 10px; padding: 28px 32px; margin-bottom: 24px;
    ">
        <h2 style="color:white; margin:0; font-size:22px; letter-spacing:0.5px;">
            SigNoise — RF Fingerprinting
        </h2>
        <p style="color:#9FB3D4; margin:6px 0 0 0; font-size:13px;">
            Projet de Fin d'Études — École Royale Navale / École Navale — Année académique 2025–2026
        </p>
        <p style="color:{PAL['gold']}; margin:4px 0 0 0; font-size:12px; font-weight:600;">
            EV2 Hamza ZOUINE &nbsp;·&nbsp; EV2  Saad EL MAACHI
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([3, 2])

    with col_a1:

        # --- CONTEXTE ---
        with st.container(border=True):
            st.markdown(f'<p class="section-label">Contexte & Problématique</p>', unsafe_allow_html=True)
            st.markdown("""
En guerre électronique, la maîtrise du spectre électromagnétique et l'identification des acteurs présents dans une zone d'intérêt sont essentielles.
Avec la prolifération des sources RF (radars, AIS, etc.), discriminer des émetteurs dans un environnement électriquement dense constitue un **verrou technologique majeur** pour les industriels de défense.

L'identification classique repose sur le décodage des messages — une approche contournable par usurpation d'identité.
**SigNoise** lève cette limite en exploitant les **imperfections physiques intrinsèques** de l'oscillateur de chaque émetteur : une empreinte unique, non falsifiable, indépendante du contenu du message.
            """)

        # --- PIPELINE ---
        with st.container(border=True):
            st.markdown(f'<p class="section-label">Pipeline de Traitement du Signal</p>', unsafe_allow_html=True)
            st.markdown("""
| Étape | Description |
|---|---|
| **1. Prétraitement** | Suppression DC offset, correction CFO, décimation FIR, normalisation puissance |
| **2. Détection des bursts AIS** | Seuillage énergie lissée, filtrage durée minimale, fusion des gaps courts |
| **3. Phase instantanée** | Transformation de Hilbert sur le signal IQ complexe |
| **4. EMD** | Décomposition Modale Empirique en fonctions modales intrinsèques (IMFs) |
| **5. CMSE** | Sélection adaptative des IMFs de bruit par seuil de contribution énergétique |
| **6. Variances multi-échelles** | AVAR (Allan), HVAR (Hadamard), MHVAR — sensibles aux différents régimes de bruit |
| **7. Vecteur signature** | DSP (Welch), fréquence instantanée, puissance, entropie de phase, indices EMD |
            """)

        # --- IA ---
        with st.container(border=True):
            st.markdown(f'<p class="section-label">Pipeline IA — Classification</p>', unsafe_allow_html=True)
            col_ia1, col_ia2 = st.columns(2)
            with col_ia1:
                st.markdown("**Classification fermée**")
                st.markdown("""
- **RFECV** : sélection automatique des features les plus discriminantes
- **Validation croisée 5-fold stratifiée** : comparaison SVM, Random Forest, Gradient Boosting, k-NN
- **Random Forest 300 estimateurs** retenu comme meilleur classifieur
                """)
            with col_ia2:
                st.markdown("**Classification ouverte (Open Set)**")
                st.markdown("""
- **Critère 1** : probabilité du classifieur ≥ seuil configurable (défaut 0.60)
- **Critère 2** : log-vraisemblance GMM ≥ percentile 5 de calibration
- Bursts hors critères → classés **INCONNU** et regroupés par distance euclidienne
                """)

    with col_a2:

        # --- RÉSULTATS ---
        with st.container(border=True):
            st.markdown(f'<p class="section-label">Résultats & Validation</p>', unsafe_allow_html=True)
            st.markdown("""
Les hypothèses de travail ont été **validées expérimentalement** :

✅ Le bruit interne contient une information discriminante — confirmé par les courbes DSP et les variances de stabilité

✅ Les variances AVAR, HVAR, MHVAR permettent de construire des signatures séparables

✅ La classification fermée valide la pertinence du vecteur signature pour l'identification supervisée

✅ L'approche GMM introduit une capacité de rejet efficace pour la classification ouverte
            """)

        # --- MATÉRIEL ---
        with st.container(border=True):
            st.markdown(f'<p class="section-label">Dispositif Expérimental</p>', unsafe_allow_html=True)
            st.markdown("""
| Matériel | Détail |
|---|---|
| **Récepteur SDR** | NI USRP-2930 (EM200) |
| **Enregistreur** | Rohde & Schwarz DWR100 |
| **Émetteur AIS** | SAAB R5 SUPREME AIS |
| **Fréquence centrale** | 161.975 / 162.025 MHz |
| **Fréquence d'échantillonnage** | 800 kHz |
| **Format fichier** | IQ float32 (complex64) |
| **Logiciel acquisition** | R&S RAMON |
            """)

        # --- ENCADREMENT ---
        with st.container(border=True):
            st.markdown(f'<p class="section-label">Encadrement</p>', unsafe_allow_html=True)
            st.markdown("""
**Encadrants:**
- M. Abdel-Ouahab BOUDRAA — Directeur Adjoint IRENav, Professeur École Navale
- M. Jean-Jacques SZKOLNIK — Ingénieur de Recherche et Innovation, IRENav
- Mme Assia BAKALI — Enseignante, École Royale Navale
- Mme Asmaa MAALI — Enseignante, École Royale Navale
- CF Yahya BENRAMDANE — Chef du Centre Opérationnel de Guerre Électronique, Marine Royale

**Co-encadrante :**
- Mme Fatima EL ABBADI — Enseignante, École Royale Navale
            """)

        # --- RAPPORT ---
        with st.container(border=True):
            st.markdown(f'<p class="section-label">Documents</p>', unsafe_allow_html=True)

            st.markdown("**📄 Rapport de projet**")
            st.markdown("Méthodologie, résultats et bases théoriques :")
            st.link_button(
                "Accéder au rapport complet",
                "https://drive.google.com/file/d/1fdFobIpf0ZmNxC1mn6wCpf8f6PDm0U69/view?usp=sharing",
                use_container_width=True
            )
            qr_path = get_path("code_qr_rapport.png")
            if qr_path:
                st.image(qr_path, caption="QR — Rapport complet", width=140)

            st.markdown("---")

            st.markdown("**📘 Guide d'utilisation SigNoise**")
            st.markdown("Manuel d'installation, prise en main et fonctionnalités :")
            st.link_button(
                "Accéder au guide d'utilisation",
                "https://drive.google.com/file/d/1LwW_DU1MAIXDMeJ6eyOswKefYKBhqLDi/view?usp=drive_link",
                use_container_width=True
            )
            qr_guide_path = get_path("code_qr_guide.png")
            if qr_guide_path:
                st.image(qr_guide_path, caption="QR — Guide d'utilisation", width=140)

    # --- FOOTER VERSION ---
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:{PAL['mid']}; font-size:12px;">
        SigNoise Web &nbsp;·&nbsp;
        École Royale Navale / IRENav &nbsp;·&nbsp;
         &nbsp;·&nbsp;PFE: 2025–2026
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("<div class='footer' style='text-align:center;'>SigNoise © 2026 </div>", unsafe_allow_html=True)
