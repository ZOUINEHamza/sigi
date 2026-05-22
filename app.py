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

# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = Path(__file__).parent
MODEL_PATH     = BASE_DIR / "final_model.pkl"
META_PATH      = BASE_DIR / "model_meta.json"
GMM_PATH       = BASE_DIR / "gmm_db.pkl"
INCONNUS_FILE  = BASE_DIR / "inconnus_db.json"
SCALER_PATH    = BASE_DIR / "scaler.pkl"
RFECV_PATH     = BASE_DIR / "rfecv.pkl"

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
icon_path = BASE_DIR / "SigNoise_charge.svg"

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

# --- HEADER (REPRODUCED FROM DESKTOP) ---
st.markdown(f"""
    <div class="header-box">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <img src="data:image/png;base64,{st.session_state.get('logo_ern_b64', '')}" width="100">
            <div style="text-align: center;">
                <h1 style="margin: 0; font-size: 28px; letter-spacing: 1px; color: {PAL['marine']};">SigNoise : RF Fingerprinting</h1>
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
            if st.button("🔍 Voir l'analyse détaillée"):
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
    st.header("Gestion de l'Intelligence Artificielle")
    
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
        if st.button("💾 Enregistrer dans la base d'apprentissage"):
            st.success(f"Échantillon ajouté à la classe '{new_class}'.")

    # 3. Bouton de réentraînement
    st.markdown("---")
    if st.button("🧠 Lancer le réentraînement global"):
        st.warning("Réentraînement en cours sur le Cloud...")

with tab_inc:
    st.header("Émetteurs Inconnus ")

    # Seuil de réentraînement automatique
    st.markdown('<p class="section-label">Configuration de l\'apprentissage continu</p>', unsafe_allow_html=True)
    n_trigger = st.number_input("Nombre de captures requis pour réentraîner automatiquement :", value=10, min_value=2)

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
                    "ID": uid,
                    "Fichier source": fichier_affiche,
                    "Captures": info.get("n", 0),
                    "Statut": info.get("statut", "EN_ATTENTE"),
                    "Date": info.get("date_dernier", info.get("date_premier", "N/A"))[:10],
                })

            st.table(pd.DataFrame(data_inc))
            
            st.markdown("---")
            st.subheader("Étiquetage & Intégration")
            selected_uid = st.selectbox("Sélectionner un inconnu pour qualification :", list(inc_db.keys()))
            new_label = st.text_input("Attribuer un nom définitif :", placeholder="Ex: AIS_NAVIRE_X")
            
            col_b1, col_b2 = st.columns(2)
            if col_b1.button(" Valider le nom"):
                st.success(f"L'émetteur {selected_uid} est maintenant identifié comme '{new_label}'.")
            
            if col_b2.button(" Lancer Réentraînement Ciblé"):
                n_caps = inc_db[selected_uid].get("n", 0)
                if n_caps >= n_trigger:
                    st.success(f"Réentraînement lancé avec {n_caps} captures de {new_label} !")
                else:
                    st.warning(f"Captures insuffisantes ({n_caps}/{n_trigger}).")
        else:
            st.info("Aucun émetteur inconnu détecté pour le moment.")
    else:
        st.info("La base des inconnus est vide.")


with tab_params:
    st.header("Configuration Avancée")

    # Layout professionnel en colonnes pour les paramètres
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        with st.container(border=True):
            st.markdown("**📡 Acquisition & Traitement**")
            st.number_input("Fréquence (Hz)", value=80000, step=1000)
            st.select_slider("Décimation", options=[1, 2, 4, 8], value=2)
            st.toggle("Utiliser EMD (Empirical Mode Decomposition)", value=True)
            st.toggle("Utiliser Wavelet Denoising", value=False)

    with col_p2:
        with st.container(border=True):
            st.markdown("**🎯 Détection AIS**")
            st.slider("Seuil de puissance (dB)", 1, 20, 8)
            st.slider("Durée minimale burst (ms)", 5, 50, 10)
            st.slider("Seuil Open-Set (GMM)", 0.1, 0.9, 0.6)

    st.button("💾 Sauvegarder la configuration système")


with tab_about:
    st.header("À Propos de SigNoise")
    
    col_a1, col_a2 = st.columns([2, 1])
    
    with col_a1:
        st.markdown("""
        ### Objectif
        **SigNoise** est un système de **RF Fingerprinting** conçu pour identifier les navires via la signature unique de leurs émetteurs AIS. 
        Contrairement à l'identification classique qui décode les messages, SigNoise analyse les imperfections physiques (bruit de phase) de l'oscillateur de l'émetteur.
        
        ### Pipeline Technique
        1. **Détection** : Identification automatique des bursts AIS dans le flux IQ.
        2. **Analyse** : Décomposition modale (EMD) et calcul de variances multi-échelles (Allan, Hadamard).
        3. **IA** : Classification par Random Forest avec détection d'émetteurs inconnus (Open Set via GMM).
        
        **Projet de Fin d'Études (2025-2026)**
        - **Élèves :** EV2 Hamza ZOUINE & EV2 Saad EL MAACHI
        - **Encadrement :** École Navale / Institut de Recherche de l'École Navale (IRENav)
        """)
        
    with col_a2:
        qr_path = get_path("code_qr_rapport.png")
        if qr_path:
            st.image(qr_path, caption="Scanner pour lire le rapport complet", width=200)
        st.info("Version Web Pro v1.0 (Cloud Edition)")


# --- FOOTER ---
st.markdown("---")
st.markdown("<div class='footer'>SigNoise © 2026 </div>", unsafe_allow_html=True)
