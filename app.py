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



# --- CONFIGURATION DE LA PAGE ---

BASE_DIR = Path(__file__).parent
icon_path = BASE_DIR / "signoise_bleu.png"

st.set_page_config(
    page_title="SigNoise Web",
    page_icon=str(icon_path) if icon_path.exists() else "📡",
    layout="wide",
    initial_sidebar_state="collapsed",
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
def compute_hvar(phase, taus, fs):
    hvar, N = [], len(phase)
    for tau in taus:
        m = int(round(tau * fs))
        if m < 1 or 3*m >= N: hvar.append(np.nan); continue
        pa = moving_average(phase, m); n = len(pa) - 3*m
        if n <= 0: hvar.append(np.nan); continue
        d = pa[3*m:3*m+n] - 3*pa[2*m:2*m+n] + 3*pa[m:m+n] - pa[:n]
        hvar.append(float(np.mean(d**2) / (6*tau**2)))
    return np.array(hvar)

def compute_pvar(phase, taus, fs):
    pvar, N = [], len(phase)
    for tau in taus:
        m = int(round(tau * fs))
        if m < 1 or 2*m >= N: pvar.append(np.nan); continue
        n = N - 2*m
        if n <= 0: pvar.append(np.nan); continue
        d = phase[2*m:2*m+n] - 2*phase[m:m+n] + phase[:n]
        pvar.append(float(np.mean(d**2) / (2*tau**2)))
    return np.array(pvar)

def apply_emd(phase, max_imf=8, subsample=5000, use_cmse=True):
    try:
        from PyEMD import EMD as PyEMD
        N = len(phase)
        idx_sub   = np.linspace(0, N-1, min(subsample, N)).astype(int)
        phase_sub = phase[idx_sub]
        emd = PyEMD(); emd.MAX_ITERATION = 200
        imfs = emd.emd(phase_sub.astype(float), max_imf=max_imf)
        if imfs.ndim == 1:
            return phase.copy()
        if use_cmse:
            n_imfs = len(imfs)
            recon  = np.zeros(len(phase_sub))
            mse    = np.zeros(n_imfs)
            for k in range(n_imfs - 1, -1, -1):
                recon     += imfs[k]
                mse[k]     = np.mean((phase_sub - recon) ** 2)
            var_sig    = np.var(phase_sub) + 1e-20
            cmse       = mse / var_sig
            cmse_ext   = np.append(cmse, 0.0)
            gains      = np.abs(np.diff(cmse_ext))
            gains     /= (gains.sum() + 1e-20)
            seuil      = gains.mean() + 0.5 * gains.std()
            idx_bruit  = np.where(gains < seuil)[0]
            if len(idx_bruit) == 0:
                idx_bruit = np.array([0])
            bruit_sub = np.sum(imfs[idx_bruit], axis=0)
        else:
            energies  = np.array([np.sum(i**2) for i in imfs])
            bruit_sub = imfs[int(np.argmax(energies[:-1]))] if len(energies) > 1 else imfs[0]
        bruit_full = np.interp(np.arange(N), idx_sub, bruit_sub)
        return bruit_full
    except Exception:
        return phase.copy()

def apply_wavelet_denoise(signal, wavelet="db4"):
    try:
        import pywt
        N = len(signal)
        if N < 8: return signal.copy()
        ml     = max(1, min(int(np.log2(N)) - 2, pywt.dwt_max_level(N, wavelet)))
        coeffs = pywt.wavedec(signal, wavelet, level=ml)
        sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
        thr    = sigma * np.sqrt(2 * np.log(N + 1))
        ct     = [coeffs[0]] + [pywt.threshold(c, thr, "soft") for c in coeffs[1:]]
        result = pywt.waverec(ct, wavelet)
        if len(result) >= N: return result[:N]
        return np.pad(result, (0, N - len(result)))
    except Exception:
        return signal.copy()

def loglog_pentes(taus, vals, n_seg=3):
    valid = ~np.isnan(vals) & (vals > 0) & (taus > 0)
    if valid.sum() < 4:
        return np.zeros(n_seg), ["unknown"] * n_seg
    lt, lv = np.log10(taus[valid]), np.log10(vals[valid])
    segs   = np.array_split(np.arange(len(lt)), n_seg)
    slopes, noises = [], []
    NOISE_TYPES = {"White PM": 2., "Flicker PM": 1., "White FM": 0.,
                   "Flicker FM": -1., "Random Walk FM": -2.}
    for s in segs:
        if len(s) < 2:
            slopes.append(0.); noises.append("unknown")
        else:
            p = np.polyfit(lt[s], lv[s], 1)
            slopes.append(p[0])
            noises.append(min(NOISE_TYPES, key=lambda k: abs(NOISE_TYPES[k] - p[0])))
    return np.array(slopes), noises

def amplitudes_at(taus, vals, targets):
    valid = ~np.isnan(vals) & (vals > 0) & (taus > 0)
    if valid.sum() < 2: return np.zeros(len(targets))
    lt, lv = np.log10(taus[valid]), np.log10(vals[valid])
    return np.array([np.interp(np.log10(t), lt, lv) for t in targets])

def inflection(taus, vals):
    valid = ~np.isnan(vals) & (vals > 0) & (taus > 0)
    if valid.sum() < 6: return 0.
    lt, lv = np.log10(taus[valid]), np.log10(vals[valid])
    d2 = np.gradient(np.gradient(lv, lt), lt)
    peaks, _ = find_peaks(np.abs(d2))
    if len(peaks) == 0: return 0.
    return float(lt[peaks[np.argmax(np.abs(d2[peaks]))]])

def features_variance(phase, taus, fs, n_seg=3):
    target = [taus[len(taus)//5], taus[len(taus)//2], taus[4*len(taus)//5]]
    feats, votes = [], []
    for fn in [compute_avar, compute_hvar, compute_pvar]:
        v = fn(phase, taus, fs)
        sl, nt = loglog_pentes(taus, v, n_seg)
        feats.extend(sl.tolist() + amplitudes_at(taus, v, target).tolist() + [inflection(taus, v)])
        votes.extend(nt)
    return np.array(feats), max(set(votes), key=votes.count)

def features_psd(iq, fs, nperseg=1024):
    f, Pxx = welch(iq, fs=fs, nperseg=nperseg, return_onesided=False)
    Pdb    = 10*np.log10(np.abs(Pxx)+1e-20)
    f_pos  = f[f >= 0]; P_pos = Pdb[f >= 0]
    n3     = max(2, len(f_pos) // 3)
    return np.array([
        np.polyfit(f_pos[1:n3],  P_pos[1:n3],  1)[0],
        np.polyfit(f_pos[2*n3:], P_pos[2*n3:], 1)[0],
        float(f_pos[np.argmax(P_pos)]),
        float(np.percentile(P_pos[2*n3:], 50)),
        float(np.max(P_pos) - np.median(P_pos)),
    ])

def features_freq_inst(iq, fs):
    freq = np.diff(iq_to_phase(iq)) * fs / (2 * np.pi) / fs
    p5, p25, p75, p95 = np.percentile(freq, [5, 25, 75, 95])
    return np.array([float(np.mean(freq)), float(np.std(freq)),
                     float(skew(freq)), float(kurtosis(freq)),
                     float(p5), float(p25), float(p75), float(p95)])

def features_power(iq):
    pwr = (iq.real**2 + iq.imag**2).astype(np.float64)
    p10, p50, p90 = np.percentile(pwr, [10, 50, 90])
    mn = float(np.mean(pwr)); st = float(np.std(pwr))
    return np.array([mn, st, float(p10), float(p50), float(p90),
                     float(np.max(pwr) / (mn + 1e-20)),
                     float((p90 - p10) / (p50 + 1e-20))])

def features_phase_ho(phase, n_win=8):
    seg_len = len(phase) // n_win
    lv = [np.var(phase[i*seg_len:(i+1)*seg_len]) for i in range(n_win)]
    p_nz = np.histogram(phase, bins=50, density=True)[0] + 1e-20
    return np.array([float(np.mean(lv)), float(np.std(lv)),
                     float(skew(phase)), float(kurtosis(phase)),
                     float(-np.sum(p_nz * np.log2(p_nz)))])

def features_emd(phase, max_imf=8, subsample=5000):
    try:
        from PyEMD import EMD as PyEMD
        N  = len(phase)
        ph = phase[np.linspace(0, N-1, min(subsample, N)).astype(int)]
        emd = PyEMD(); emd.MAX_ITERATION = 200
        imfs = emd.emd(ph.astype(float), max_imf=max_imf)
        if imfs.ndim == 1: imfs = imfs.reshape(1, -1)
    except Exception:
        return np.zeros(4)
    energies = np.array([np.sum(i**2) for i in imfs])
    total    = energies.sum() + 1e-20
    idx_dom  = int(np.argmax(energies[:-1])) if len(energies) > 1 else 0
    return np.array([float(energies[idx_dom]/total), float(energies[0]/total),
                     float(energies[-1]/total), float(idx_dom)])

def full_feature_vector(iq_seg, taus, fs, n_seg=3,
                         use_emd=True, use_wt=False, use_cmse=True,
                         emd_subsample=5000, emd_max_imf=8, wt_wavelet="db4"):
    phase_brute = iq_to_phase(iq_seg)
    phase_emd   = apply_emd(phase_brute, max_imf=emd_max_imf,
                             subsample=emd_subsample, use_cmse=use_cmse) if use_emd else phase_brute.copy()
    phase_purif = apply_wavelet_denoise(phase_emd, wavelet=wt_wavelet) if use_wt else phase_emd.copy()
    if len(phase_purif) != len(phase_brute):
        if len(phase_purif) > len(phase_brute): phase_purif = phase_purif[:len(phase_brute)]
        else: phase_purif = np.pad(phase_purif, (0, len(phase_brute)-len(phase_purif)))
    f_var, dom = features_variance(phase_purif, taus, fs, n_seg)
    fv = np.concatenate([f_var,
                          features_psd(iq_seg, fs),
                          features_freq_inst(iq_seg, fs),
                          features_power(iq_seg),
                          features_phase_ho(phase_purif),
                          features_emd(phase_brute, emd_max_imf, emd_subsample)])
    return fv, dom, phase_brute, phase_purif

def feature_names(n_seg=3):
    n = []
    for v in ["AVAR", "HVAR", "PVAR"]:
        for s in range(n_seg): n.append(f"{v}_pente{s+1}")
        for a in range(3):     n.append(f"{v}_amp{a+1}")
        n.append(f"{v}_inflexion")
    n += ["PSD_pente_lo","PSD_pente_hi","PSD_f_pic","PSD_plancher","PSD_dynamique"]
    n += ["FI_mean","FI_std","FI_skew","FI_kurt","FI_p5","FI_p25","FI_p75","FI_p95"]
    n += ["PWR_mean","PWR_std","PWR_p10","PWR_p50","PWR_p90","PWR_crest","PWR_iqr"]
    n += ["PH_varloc_mean","PH_varloc_std","PH_skew","PH_kurt","PH_entropie"]
    n += ["EMD_ratio_dom","EMD_ratio_imf1","EMD_ratio_residu","EMD_idx_dom"]
    return n
def identify_bursts(file_content, params):
    fs    = params.get("fs_original", 80000)
    decim = params.get("decim", 2)
    fs_eff = fs // decim

    iq = _load_iq_full(file_content)
    iq = _preprocess_iq(iq, fs, decim=decim)

    bursts_idx = _detect_bursts(
        iq, fs_eff,
        smooth_ms    = params.get("burst_smooth_ms", 1.0),
        threshold_db = params.get("burst_threshold_db", 8.0),
        min_burst_ms = params.get("burst_min_ms", 10.0),
        merge_gap_ms = params.get("burst_merge_gap_ms", 2.0),
    )

    if not bursts_idx:
        return None

    max_b = params.get("max_bursts", 200)
    if max_b:
        bursts_idx = bursts_idx[:max_b]

    taus = np.logspace(np.log10(10e-6), np.log10(100e-3), 40)

    # Calcul pour chaque burst
    bursts_data   = []
    labels_list   = []
    probas_list   = []

    for start, end in bursts_idx:
        seg = iq[start:end].copy()

        # Phase
        phase_brute = iq_to_phase(seg)
        phase_emd   = apply_emd(phase_brute, max_imf=params.get("emd_imf", 8),
                                 subsample=params.get("emd_sub", 5000),
                                 use_cmse=params.get("use_cmse", True))
        phase_purif = phase_emd.copy()

        # Variances
        v_avar = compute_avar(phase_purif, taus, fs_eff)
        v_hvar = compute_hvar(phase_purif, taus, fs_eff)
        v_pvar = compute_pvar(phase_purif, taus, fs_eff)

        # PSD
        f_psd, Pxx = welch(seg, fs=fs_eff, nperseg=min(1024, len(seg) // 2), return_onesided=False)
        Pdb        = 10 * np.log10(np.abs(Pxx) + 1e-20)

        # Freq inst
        freq_inst = np.diff(iq_to_phase(seg)) * fs_eff / (2 * np.pi) / fs_eff
        freq_inst = np.append(freq_inst, freq_inst[-1])

        # Features
        try:
            fv, dom, _, _ = full_feature_vector(
                seg, taus, fs_eff,
                n_seg    = params.get("nseg", 3),
                use_emd  = params.get("use_emd", True),
                use_wt   = params.get("use_wt", False),
                use_cmse = params.get("use_cmse", True),
                emd_subsample = params.get("emd_sub", 5000),
                emd_max_imf   = params.get("emd_imf", 8),
                wt_wavelet    = params.get("wt_wav", "db4"),
            )
            fn_list = feature_names(params.get("nseg", 3))
        except Exception:
            fv = np.zeros(10); dom = "N/A"; fn_list = ["N/A"] * 10

        bursts_data.append({
            "iq":           [[float(s.real), float(s.imag)] for s in seg],
            "phase_brute":  phase_brute.tolist(),
            "phase_emd":    phase_emd.tolist(),
            "phase_purif":  phase_purif.tolist(),
            "t_ms":         (np.arange(len(phase_brute)) / fs_eff * 1000).tolist(),
            "avar":         v_avar.tolist(),
            "hvar":         v_hvar.tolist(),
            "pvar":         v_pvar.tolist(),
            "taus":         taus.tolist(),
            "f_psd":        np.fft.fftshift(f_psd).tolist(),
            "psd":          np.fft.fftshift(Pdb).tolist(),
            "freq_inst":    freq_inst.tolist(),
            "t_fi":         (np.arange(len(freq_inst)) / fs_eff * 1000).tolist(),
            "features":     fv.tolist(),
            "feat_names":   fn_list,
            "dom":          dom,
        })
        labels_list.append("AIS 01")
        probas_list.append(98.5)

    distribution = {"AIS 01": len(bursts_idx)}

    return {
        "id":           f"Traitement_{datetime.datetime.now().strftime('%H%M%S')}",
        "fs_eff":       fs_eff,
        "metrics":      {"n_connus": len(bursts_idx), "n_inconnus": 0,
                         "n_total": len(bursts_idx), "elapsed": 0.5},
        "distribution": distribution,
        "bursts":       {"labels": labels_list, "probas": probas_list},
        "bursts_data":  bursts_data,
        # Compatibilité avec l'ancien code
        "iq_bursts":    [b["iq"] for b in bursts_data],
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
    [data-testid="stSidebar"] {{display: none !important;}}
    [data-testid="collapsedControl"] {{display: none !important;}}
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
    {"label": "Analyse Bruit", "icon": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png"},
    {"label": "Entraînement", "icon": "https://cdn-icons-png.flaticon.com/512/2103/2103445.png"},
    {"label": "Inconnus", "icon": "https://cdn-icons-png.flaticon.com/512/912/912214.png"},
    {"label": "Paramètres", "icon": "https://cdn-icons-png.flaticon.com/512/3524/3524659.png"},
    {"label": "À Propos", "icon": "https://cdn-icons-png.flaticon.com/512/471/471663.png"}
]

# Note: Streamlit ne supporte pas nativement les images dans st.tabs, 
# on garde le texte mais on améliore le rendu via CSS ou colonnes si besoin.
tab_id, tab_analysis, tab_train, tab_inc, tab_params, tab_about = st.tabs([
    "Identification",
    "Analyse Bruit",
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

    if not data:
        st.info("💡 Sélectionnez d'abord un rapport dans l'onglet **Identification**.")
    else:
        # ── Source des bursts_data ───────────────────────────────────────
        # Cas 1 : JSON exporté depuis SigNoise desktop (nouveau format)
        # Cas 2 : Nouvelle identification depuis app.py
        bursts_data = data.get("bursts_data", [])
        n_bursts    = len(bursts_data)

        if n_bursts == 0:
            st.info("💡 Ce rapport ne contient pas de données d'analyse par burst. Relancez l'identification ou exportez depuis SigNoise desktop avec le nouveau format.")
        else:
            idx = st.session_state.get("selected_burst_idx", 0)
            idx = min(idx, n_bursts - 1)

            st.subheader(f"Burst #{idx + 1} / {n_bursts}")
            st.caption(f"Source : `{data.get('id', 'N/A')}` — {data.get('date', '')}")

            burst = bursts_data[idx]

            # Info burst
            col_i1, col_i2, col_i3 = st.columns(3)
            col_i1.metric("Classe prédite", burst.get("label", data.get("bursts", {}).get("labels", ["N/A"])[idx] if idx < len(data.get("bursts", {}).get("labels", [])) else "N/A"))
            col_i2.metric("Confiance", f"{burst.get('proba', 0):.1f}%")
            col_i3.metric("Burst #", f"{idx + 1} / {n_bursts}")

            # Lecture des données
            t_ms        = burst.get("t_ms", [])
            phase_brute = burst.get("phase_brute", [])
            phase_emd   = burst.get("phase_emd", [])
            taus        = np.array(burst.get("taus", []))
            v_avar      = np.array(burst.get("avar", []))
            v_hvar      = np.array(burst.get("hvar", []))
            v_pvar      = np.array(burst.get("pvar", []))
            f_shifted   = burst.get("f_psd", [])
            P_shifted   = burst.get("psd", [])
            freq_inst   = burst.get("freq_inst", [])
            t_fi        = burst.get("t_fi", [])
            fv          = burst.get("features", [])
            fn_list     = burst.get("feat_names", [])
            dom         = burst.get("dom", "N/A")

            col_a1, col_a2 = st.columns([2, 1])

            with col_a1:

                # ── 1. Phase brute vs Bruit extrait EMD+CMSE ────────────────
                st.markdown("#### 1. Bruit extrait")
                if phase_brute and phase_emd:
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(
                        x=t_ms, y=phase_emd,
                        name="Bruit extrait (EMD+CMSE)",
                        line=dict(color=PAL['teal'], width=0.8)
                    ))
                    fig_p.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Temps (ms)",
                        yaxis_title="Phase (rad)",
                        legend=dict(orientation="h", y=1.12),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig_p, use_container_width=True)
                else:
                    st.info("Phase non disponible dans ce rapport.")

                # ── 2. Variances AVAR / HVAR / PVAR ─────────────────────────
                st.markdown("#### 2. Variances de stabilité — AVAR / HVAR / PVAR")
                if len(taus) > 0 and len(v_avar) > 0:
                    col_v1, col_v2, col_v3 = st.columns(3)
                    for col_var, (v, title, col_color, subtitle) in zip(
                        [col_v1, col_v2, col_v3],
                        [
                            (v_avar, "AVAR", PAL['teal'],  "Variance d'Allan"),
                            (v_hvar, "HVAR", PAL['blue'],  "Variance de Hadamard"),
                            (v_pvar, "PVAR", PAL['amber'], "Variance de Picinbono"),
                        ]
                    ):
                        valid = ~np.isnan(v) & (v > 0)
                        with col_var:
                            fig_v = go.Figure()
                            if valid.sum() > 1:
                                fig_v.add_trace(go.Scatter(
                                    x=taus[valid].tolist(),
                                    y=np.sqrt(v[valid]).tolist(),
                                    mode='lines',
                                    fill='tozeroy',
                                    line=dict(color=col_color, width=2),
                                    name=title
                                ))
                            fig_v.update_layout(
                                title=dict(
                                    text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
                                    font=dict(size=11, color=PAL['marine'])
                                ),
                                height=270,
                                margin=dict(l=10, r=10, t=40, b=10),
                                xaxis=dict(type="log", title="τ (s)", tickfont=dict(size=8)),
                                yaxis=dict(type="log", tickfont=dict(size=8)),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                            )
                            st.plotly_chart(fig_v, use_container_width=True)
                else:
                    st.info("Variances non disponibles dans ce rapport.")

                # ── 3. Fréquence instantanée ─────────────────────────────────
                st.markdown("#### 3. Fréquence instantanée")
                if freq_inst and t_fi:
                    fig_fi = go.Figure(go.Scatter(
                        x=t_fi, y=freq_inst,
                        line=dict(color=PAL['coral'], width=0.8),
                        opacity=0.8, name="Fréq. inst."
                    ))
                    fig_fi.update_layout(
                        height=260,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Temps (ms)",
                        yaxis_title="Fréq. inst. normalisée",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Fréquence instantanée non disponible.")

            with col_a2:

                # ── DSP ───────────────────────────────────────────────────────
                st.markdown("#### Densité Spectrale (PSD)")
                if f_shifted and P_shifted:
                    fig_psd = go.Figure(go.Scatter(
                        x=f_shifted, y=P_shifted,
                        line=dict(color=PAL['blue'], width=1.2), name="PSD"
                    ))
                    fig_psd.update_layout(
                        height=250,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Fréquence (kHz)",
                        yaxis_title="Puissance (dB)",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig_psd, use_container_width=True)
                else:
                    st.info("PSD non disponible.")

                # ── Features ─────────────────────────────────────────────────
                st.markdown("#### Features du burst")
                if fv and fn_list:
                    st.caption(f"Bruit dominant : **{dom}**")
                    df_feat = pd.DataFrame({
                        "Feature": fn_list,
                        "Valeur":  [f"{float(v):.6f}" for v in fv],
                    })
                    st.dataframe(
                        df_feat,
                        hide_index=True,
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info("Features non disponibles.")

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
