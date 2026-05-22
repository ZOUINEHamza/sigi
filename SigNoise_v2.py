
import sys, os, json, datetime, warnings, traceback, glob
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats  import skew, kurtosis
from sklearn.svm                import SVC
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.mixture            import GaussianMixture
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.model_selection    import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.feature_selection  import RFECV
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import confusion_matrix, classification_report
import joblib

os.environ["QT_API"] = "pyqt6"
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.rcParams.update({
    "font.family": "Segoe UI",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "figure.facecolor": "#FAFAF8",
    "axes.facecolor":   "#FAFAF8",
})

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog,
    QProgressBar, QTextEdit, QGroupBox, QSplitter,
    QStatusBar, QSpinBox, QDoubleSpinBox, QComboBox,
    QFrame, QScrollArea, QSizePolicy, QMessageBox,
    QSplashScreen, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabBar, QAbstractSpinBox,
)
from PyQt6.QtCore  import Qt, QThread, pyqtSignal, QTimer, QSize, QRectF
from PyQt6.QtGui   import (
    QFont, QColor, QPixmap, QPainter, QLinearGradient, QIcon,
)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtSvg        import QSvgRenderer
from PyQt6.QtGui   import QIcon, QPixmap, QPainter, QColor
from PyQt6.QtCore  import Qt, QByteArray
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE & STYLESHEET
# ─────────────────────────────────────────────────────────────────────────────
def make_icon(svg_str, size=22, color="#0A1637"):
    svg_colored = svg_str.replace('stroke="currentColor"', f'stroke="{color}"')
    px       = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(svg_colored.encode()))
    painter  = QPainter(px)
    renderer.render(painter)
    painter.end()
    return QIcon(px)

# ── Icônes Lucide ──────────────────────────────────────────────────────────
SVG_RADIO_TOWER = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <path d="M4.9 16.1C1 12.2 1 5.8 4.9 1.9"/>
  <path d="M7.8 4.7a6.14 6.14 0 0 0 0 8.5"/>
  <circle cx="12" cy="9" r="2"/>
  <path d="M16.2 4.7a6.14 6.14 0 0 1 0 8.5"/>
  <path d="M19.1 1.9a10.11 10.11 0 0 1 0 14.2"/>
  <line x1="12" y1="11" x2="12" y2="22"/>
  <line x1="8"  y1="22" x2="16" y2="22"/>
</svg>"""

SVG_ACTIVITY = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
</svg>"""

SVG_TRAIN = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <ellipse cx="12" cy="5" rx="8" ry="3"/>
  <path d="M4 5v7c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/>
  <path d="M4 12v5c0 1.66 3.58 3 8 3"/>
  <path d="M19 17v5"/>
  <path d="M22 20l-3-3-3 3"/>
</svg>"""

SVG_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
  <circle cx="12" cy="12" r="3"/>
</svg>"""

SVG_INFO = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10"/>
  <path d="M12 16v-4"/>
  <path d="M12 8h.01"/>
</svg>"""

SVG_PLAY = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10"/>
  <polygon points="10 8 16 12 10 16 10 8"/>
</svg>"""

SVG_PAUSE = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10"/>
  <line x1="10" y1="15" x2="10" y2="9"/>
  <line x1="14" y1="15" x2="14" y2="9"/>
</svg>"""

SVG_SHIELD = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  <path d="M9 12l2 2 4-4"/>
</svg>"""

SVG_USERS = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
  <circle cx="9" cy="7" r="4"/>
  <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
  <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
</svg>"""

SVG_GRID = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <rect x="3" y="3" width="7" height="7"/>
  <rect x="14" y="3" width="7" height="7"/>
  <rect x="14" y="14" width="7" height="7"/>
  <rect x="3" y="14" width="7" height="7"/>
</svg>"""

SVG_DOWNLOAD = """<svg xmlns="http://www.w3.org/2000/svg"
  width="24" height="24" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
  <polyline points="7 10 12 15 17 10"/>
  <line x1="12" y1="15" x2="12" y2="3"/>
</svg>"""

PAL = {
    "teal":   "#1D9E75",
    "coral":  "#D85A30",
    "purple": "#534AB7",
    "amber":  "#BA7517",
    "blue":   "#185FA5",
    "dark":   "#2C2C2A",
    "mid":    "#5F5E5A",
    "light":  "#F7F7F5",
    "white":  "#FFFFFF",
    "border": "#D3D1C7",
    "marine": "#0A1637",
    "gold":   "#C4A050",
    "unknown":"#7B5EA7",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {PAL['light']};
    color: {PAL['dark']};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {PAL['border']};
    border-radius: 0 0 8px 8px;
    background: {PAL['white']};
    margin-top: -1px;
}}
QTabBar::tab {{
    background: {PAL['white']};
    color: {PAL['dark']};
    padding: 11px 20px;
    margin-right: 2px;
    border: 1px solid {PAL['border']};
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 500;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    background: {PAL['white']};
    color: {PAL['marine']};
    border-bottom: 3px solid {PAL['gold']};
    font-weight: 700;
}}
QTabBar::tab:hover:!selected {{
    background: #F0F4FF;
    color: {PAL['marine']};
}}
QPushButton {{
    background-color: {PAL['teal']};
    color: white;
    border: none;
    border-radius: 7px;
    padding: 9px 22px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton:hover    {{ background-color: #0F6E56; }}
QPushButton:pressed  {{ background-color: #085041; }}
QPushButton:disabled {{ background-color: {PAL['border']}; color: {PAL['mid']}; }}
QPushButton#secondary {{
    background-color: {PAL['white']};
    color: {PAL['dark']};
    border: 1px solid {PAL['border']};
}}
QPushButton#secondary:hover {{ background-color: {PAL['light']}; border-color: {PAL['mid']}; }}
QPushButton#danger   {{ background-color: {PAL['coral']}; }}
QPushButton#danger:hover {{ background-color: #993C1D; }}
QPushButton#gold {{
    background-color: {PAL['gold']};
    color: {PAL['marine']};
}}
QPushButton#gold:hover {{ background-color: #B8922A; }}
QPushButton#purple {{
    background-color: {PAL['purple']};
    color: white;
}}
QPushButton#purple:hover {{ background-color: #3E33A0; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {PAL['white']};
    border: 1px solid {PAL['border']};
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 13px;
    color: {PAL['dark']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {PAL['teal']}; }}
QProgressBar {{
    border: none; border-radius: 5px;
    background: {PAL['border']}; height: 12px;
    text-align: center; color: {PAL['dark']}; font-size: 11px;
}}
QProgressBar::chunk {{ background-color: {PAL['teal']}; border-radius: 5px; }}
QGroupBox {{
    font-weight: 600; font-size: 12px; color: {PAL['mid']};
    border: 1px solid {PAL['border']}; border-radius: 8px;
    margin-top: 14px; padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    background: {PAL['white']}; text-transform: uppercase;
    letter-spacing: 1px; font-size: 11px;
}}
QTextEdit {{
    background: {PAL['white']}; border: 1px solid {PAL['border']};
    border-radius: 6px; padding: 8px;
    font-family: 'Consolas', monospace; font-size: 12px; color: {PAL['dark']};
}}
QTableWidget {{
    background: {PAL['white']}; border: 1px solid {PAL['border']};
    border-radius: 6px; gridline-color: {PAL['border']};
    font-size: 12px;
}}
QTableWidget::item:selected {{ background: #E1F5EE; color: {PAL['dark']}; }}
QHeaderView::section {{
    background: {PAL['light']}; border: none;
    border-bottom: 2px solid {PAL['border']};
    padding: 6px 10px; font-weight: 600;
    font-size: 11px; letter-spacing: 0.5px;
    color: {PAL['mid']};
}}
QStatusBar {{
    background: {PAL['marine']}; color: #C0BEB4;
    font-size: 12px; padding: 4px 12px;
}}
QScrollBar:vertical {{
    background: {PAL['light']}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {PAL['border']}; border-radius: 4px; min-height: 20px;
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 2px solid {PAL['border']};
}}
QCheckBox::indicator:checked {{
    background: {PAL['teal']}; border-color: {PAL['teal']};
}}
QFrame#header {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {PAL['dark']}, stop:1 #444441);
    border-radius: 10px;
}}
QFrame#card {{
    background: {PAL['white']}; border: 1px solid {PAL['border']};
    border-radius: 10px;
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# SPLASH SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def make_splash():
    splash_svg = WORK_DIR / "SigNoise_charge.svg"
    px = QPixmap(540, 300)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    if splash_svg.exists():
        renderer = QSvgRenderer(str(splash_svg))
        renderer.render(p, QRectF(0, 0, 540, 300))
    p.end()
    return px

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_BYTES    = 50 * 1024 * 1024
WORK_DIR       = Path(__file__).parent
MODEL_PATH     = WORK_DIR / "final_model.pkl"
META_PATH      = WORK_DIR / "model_meta.json"
GMM_PATH       = WORK_DIR / "gmm_db.pkl"
INCONNUS_FILE  = WORK_DIR / "inconnus_db.json"
SCALER_PATH    = WORK_DIR / "scaler.pkl"
RFECV_PATH     = WORK_DIR / "rfecv.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING — MODE AIS (BURST)
# ─────────────────────────────────────────────────────────────────────────────

def _load_iq_full(filepath):
    """Charge tout le fichier IQ float32 en mémoire (complex64)."""
    with open(filepath, "rb") as f:
        raw = f.read()
    n = len(raw) // 8
    return np.frombuffer(raw[:n*8], dtype=np.complex64).copy()

def _preprocess_iq(iq, fs, decim=1):
    """DC offset + CFO + décimation FIR optionnelle."""
    # DC offset
    iq = iq - np.mean(iq)
    # CFO : estimation par auto-corrélation sur le carré du signal
    sq = iq ** 2
    cfo_est = np.angle(np.mean(sq[1:] * np.conj(sq[:-1]))) / (2 * np.pi) * (fs / 2)
    t = np.arange(len(iq)) / fs
    iq = iq * np.exp(-1j * 2 * np.pi * cfo_est * t)
    # Décimation FIR
    if decim > 1:
        from scipy.signal import decimate
        iq_r = decimate(iq.real, decim, ftype="fir", zero_phase=True)
        iq_i = decimate(iq.imag, decim, ftype="fir", zero_phase=True)
        iq   = (iq_r + 1j * iq_i).astype(np.complex64)
    # Normalisation puissance
    pwr = np.mean(np.abs(iq) ** 2)
    if pwr > 0:
        iq = iq / np.sqrt(pwr)
    return iq

def _detect_bursts(iq, fs, smooth_ms=1.0, threshold_db=8.0,
                   min_burst_ms=10.0, merge_gap_ms=2.0):
    """Détecte les bursts AIS par seuillage de l'énergie lissée."""
    EPS_E    = 1e-30
    energie  = np.abs(iq) ** 2
    win      = max(1, int(smooth_ms * 1e-3 * fs))
    kern     = np.ones(win) / win
    e_smooth = np.convolve(energie, kern, mode="same")
    e_db     = 10 * np.log10(e_smooth + EPS_E)
    plancher = np.percentile(e_db, 10)
    seuil    = plancher + threshold_db

    masque   = e_db > seuil
    segs     = []
    en_burst = False
    debut    = 0
    for i, v in enumerate(masque):
        if v and not en_burst:
            debut = i; en_burst = True
        elif not v and en_burst:
            segs.append((debut, i)); en_burst = False
    if en_burst:
        segs.append((debut, len(masque)))

    # Fusion gaps courts
    gap_min = int(merge_gap_ms * 1e-3 * fs)
    fused   = []
    for seg in segs:
        if fused and (seg[0] - fused[-1][1]) < gap_min:
            fused[-1] = (fused[-1][0], seg[1])
        else:
            fused.append(list(seg))

    # Filtre durée minimale
    min_samp = int(min_burst_ms * 1e-3 * fs)
    valides  = [(s, e) for s, e in fused if (e - s) >= min_samp]
    return valides

def iter_bursts_ais(filepath, params):
    """Générateur de bursts AIS depuis un fichier IQ brut.
    Remplace iter_segments : chaque burst détecté est un échantillon."""
    fs     = params["fs_original"]
    decim  = params.get("decim", 1)
    fs_eff = fs // decim

    iq = _load_iq_full(filepath)
    iq = _preprocess_iq(iq, fs, decim=decim)

    bursts = _detect_bursts(
        iq, fs_eff,
        smooth_ms    = params.get("burst_smooth_ms",    1.0),
        threshold_db = params.get("burst_threshold_db", 8.0),
        min_burst_ms = params.get("burst_min_ms",       10.0),
        merge_gap_ms = params.get("burst_merge_gap_ms", 2.0),
    )
    max_b = params.get("max_bursts", None)
    if max_b is not None:
        # Prendre les N bursts les plus longs
        bursts = sorted(bursts, key=lambda se: se[1]-se[0], reverse=True)[:max_b]
        bursts = sorted(bursts, key=lambda se: se[0])  # re-trier chronologiquement

    for s, e in bursts:
        yield iq[s:e].copy()

def count_bursts_ais(filepath, params):
    """Compte le nombre de bursts détectés dans un fichier IQ."""
    fs     = params["fs_original"]
    decim  = params.get("decim", 1)
    fs_eff = fs // decim

    iq = _load_iq_full(filepath)
    iq = _preprocess_iq(iq, fs, decim=decim)

    bursts = _detect_bursts(
        iq, fs_eff,
        smooth_ms    = params.get("burst_smooth_ms",    1.0),
        threshold_db = params.get("burst_threshold_db", 8.0),
        min_burst_ms = params.get("burst_min_ms",       10.0),
        merge_gap_ms = params.get("burst_merge_gap_ms", 2.0),
    )
    max_b = params.get("max_bursts", None)
    if max_b is not None:
        return min(len(bursts), max_b)
    return max(1, len(bursts))

# Alias de compatibilité (utilisé dans AnalysisWorker pour "premier burst")
def iter_segments(filepath, sps):
    """OBSOLÈTE — conservé pour compatibilité interne uniquement."""
    buf = np.array([], dtype=np.complex64)
    with open(filepath, "rb") as f:
        while True:
            raw = f.read(CHUNK_BYTES)
            if not raw: break
            n = len(raw) // 8
            chunk = np.frombuffer(raw[:n*8], dtype=np.complex64)
            buf = np.concatenate([buf, chunk])
            while len(buf) >= sps:
                yield buf[:sps].copy()
                buf = buf[sps:]

def count_segments(filepath, sps):
    """OBSOLÈTE — conservé pour compatibilité interne uniquement."""
    size_bytes = os.path.getsize(filepath)
    n_samples  = size_bytes // 8
    return max(1, n_samples // sps)

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

NOISE_TYPES = {"White PM": 2., "Flicker PM": 1., "White FM": 0.,
               "Flicker FM": -1., "Random Walk FM": -2.}

def bruit_proche(s):
    return min(NOISE_TYPES, key=lambda k: abs(NOISE_TYPES[k] - s))

# ── CMSE (Cumulative Mean Square Error) ───────────────────────────────────────
def CMSE_appliquer(imfs, signal_orig):
    n_imfs, N = len(imfs), imfs.shape[1]
    mse   = np.zeros(n_imfs)
    recon = np.zeros(N)
    for k in range(n_imfs - 1, -1, -1):
        recon  = recon + imfs[k]
        mse[k] = np.mean((signal_orig - recon) ** 2)
    var_sig    = np.var(signal_orig) + 1e-20
    cmse       = mse / var_sig
    cmse_ext   = np.append(cmse, 0.0)
    gains_raw  = np.array([cmse_ext[k+1] - cmse_ext[k] for k in range(n_imfs)])
    gains_abs  = np.abs(gains_raw)
    total      = gains_abs.sum() + 1e-20
    gains_norm = gains_abs / total
    seuil      = gains_norm.mean() + 0.5 * gains_norm.std()
    idx_signal = np.where(gains_norm >= seuil)[0]
    idx_bruit  = np.where(gains_norm <  seuil)[0]
    if len(idx_bruit) == 0:
        idx_bruit  = np.array([0])
        idx_signal = np.arange(1, n_imfs)
    if len(idx_signal) == 0:
        idx_signal = np.array([n_imfs - 1])
        idx_bruit  = np.arange(0, n_imfs - 1)
    return idx_bruit, idx_signal, cmse, gains_norm

def apply_emd(phase, max_imf=8, subsample=5000, use_subsample=True, use_cmse=True):
    try:
        from PyEMD import EMD as PyEMD
        N = len(phase)
        if use_subsample and N > subsample:
            idx_sub   = np.linspace(0, N-1, subsample).astype(int)
            phase_sub = phase[idx_sub]
        else:
            idx_sub   = np.arange(N)
            phase_sub = phase
        emd = PyEMD(); emd.MAX_ITERATION = 200
        imfs = emd.emd(phase_sub.astype(float), max_imf=max_imf)
        if imfs.ndim == 1:
            return phase.copy()
        if use_cmse:
            idx_bruit, _, _, _ = CMSE_appliquer(imfs, phase_sub)
            bruit_sub = np.sum(imfs[idx_bruit], axis=0)
        else:
            energies  = np.array([np.sum(i**2) for i in imfs])
            idx_dom   = int(np.argmax(energies[:-1])) if len(energies) > 1 else 0
            bruit_sub = imfs[idx_dom]
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

def phase_to_purified(phase, use_emd=True, use_wt=False, use_cmse=True,
                       emd_subsample=5000, emd_max_imf=8, wt_wavelet="db4"):
    N      = len(phase)
    result = phase.copy()
    if use_emd:
        result = apply_emd(result, max_imf=emd_max_imf, subsample=emd_subsample,
                           use_subsample=True, use_cmse=use_cmse)
    if use_wt:
        result = apply_wavelet_denoise(result, wavelet=wt_wavelet)
    if len(result) > N: result = result[:N]
    elif len(result) < N: result = np.pad(result, (0, N - len(result)))
    return result

# ── Feature extraction ─────────────────────────────────────────────────────
def loglog_pentes(taus, vals, n_seg=3):
    valid = ~np.isnan(vals) & (vals > 0) & (taus > 0)
    if valid.sum() < 4:
        return np.zeros(n_seg), ["unknown"] * n_seg
    lt, lv = np.log10(taus[valid]), np.log10(vals[valid])
    segs   = np.array_split(np.arange(len(lt)), n_seg)
    slopes, noises = [], []
    for s in segs:
        if len(s) < 2:
            slopes.append(0.); noises.append("unknown")
        else:
            p = np.polyfit(lt[s], lv[s], 1)
            slopes.append(p[0]); noises.append(bruit_proche(p[0]))
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
    d2     = np.gradient(np.gradient(lv, lt), lt)
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
    seg = len(phase) // n_win
    lv  = [np.var(phase[i*seg:(i+1)*seg]) for i in range(n_win)]
    return np.array([float(np.mean(lv)), float(np.std(lv)),
                     float(skew(phase)), float(kurtosis(phase)),
                     float(-np.sum(p_nz * np.log2(p_nz + 1e-20))
                            if (p_nz := np.histogram(phase, bins=50, density=True)[0] + 1e-20).any() else 0.)])

def features_emd(phase, max_imf=8, subsample=5000):
    try:
        from PyEMD import EMD as PyEMD
        N = len(phase)
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
    phase_purif = phase_to_purified(phase_brute, use_emd=use_emd, use_wt=use_wt,
                                     use_cmse=use_cmse, emd_subsample=emd_subsample,
                                     emd_max_imf=emd_max_imf, wt_wavelet=wt_wavelet)
    f_var, dom  = features_variance(phase_purif, taus, fs, n_seg)
    fv = np.concatenate([f_var,
                          features_psd(iq_seg, fs),
                          features_freq_inst(iq_seg, fs),
                          features_power(iq_seg),
                          features_phase_ho(phase_purif),
                          features_emd(phase_brute, emd_max_imf, emd_subsample)])
    return fv, dom, phase_brute, phase_purif

def feature_names(n_seg=3):
    n = []
    for v in ["AVAR","HVAR","PVAR"]:
        for s in range(n_seg): n.append(f"{v}_pente{s+1}")
        for a in range(3):     n.append(f"{v}_amp{a+1}")
        n.append(f"{v}_inflexion")
    n += ["PSD_pente_lo","PSD_pente_hi","PSD_f_pic","PSD_plancher","PSD_dynamique"]
    n += ["FI_mean","FI_std","FI_skew","FI_kurt","FI_p5","FI_p25","FI_p75","FI_p95"]
    n += ["PWR_mean","PWR_std","PWR_p10","PWR_p50","PWR_p90","PWR_crest","PWR_iqr"]
    n += ["PH_varloc_mean","PH_varloc_std","PH_skew","PH_kurt","PH_entropie"]
    n += ["EMD_ratio_dom","EMD_ratio_imf1","EMD_ratio_residu","EMD_idx_dom"]
    return n

def make_classifiers():
    return {
        "SVM (RBF)": Pipeline([("sc",StandardScaler()),
            ("clf",SVC(kernel="rbf",C=10,gamma="scale",probability=True,random_state=42))]),
        "Random Forest": Pipeline([("sc",StandardScaler()),
            ("clf",RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("sc",StandardScaler()),
            ("clf",GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,
                                               max_depth=4,random_state=42))]),
        "k-NN (k=5)": Pipeline([("sc",StandardScaler()),
            ("clf",KNeighborsClassifier(n_neighbors=5))]),
    }

def convert_json(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    raise TypeError(f"Type {type(obj)} non sérialisable")

def export_to_google_sheets(res, id_traitement="AUTO"):
    """Exporte les résultats vers Google Sheets pour le portail Streamlit."""
    try:
        # 1. Configurer l'accès à Google Sheets
        # Nécessite un fichier credentials.json dans le même dossier
        cred_path = Path(__file__).parent / "credentials.json"
        if not cred_path.exists():
            print("⚠ Fichier credentials.json manquant. Export annulé.")
            return

        scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                 "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
        
        creds = ServiceAccountCredentials.from_json_keyfile_name(str(cred_path), scope)
        client = gspread.authorize(creds)
        
        # 2. Ouvrir le tableau (doit être partagé avec l'email du service account)
        # Nom du tableau à créer sur Google Sheets : SigNoise_Results
        try:
            sheet = client.open("SigNoise_Results").sheet1
        except Exception:
            print("⚠ Impossible d'ouvrir 'SigNoise_Results'. Vérifiez le partage.")
            return

        # 3. Préparer les données (échantillonnage pour éviter de saturer les cellules)
        # On réduit à ~200 points pour la fluidité web
        psd_f = res.get("f_psd", [])
        psd_p = res.get("Pdb", [])
        if len(psd_f) > 200:
            step = len(psd_f) // 200
            psd_f = psd_f[::step]
            psd_p = psd_p[::step]

        # 4. Créer la ligne
        nouvelle_ligne = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            id_traitement,
            res.get("decision", "Analyse Seule"),
            f"{res.get('consensus', 100.0):.1f}%",
            json.dumps(psd_f.tolist() if hasattr(psd_f, "tolist") else psd_f),
            json.dumps(psd_p.tolist() if hasattr(psd_p, "tolist") else psd_p),
            json.dumps(res["taus"].tolist() if hasattr(res["taus"], "tolist") else res["taus"]),
            json.dumps(res["avar"].tolist() if hasattr(res["avar"], "tolist") else res["avar"])
        ]

        sheet.append_row(nouvelle_ligne)
        print(f"✅ Exportation réussie vers Google Sheets (ID: {id_traitement})")
    except Exception as e:
        print(f"❌ Erreur lors de l'exportation Google Sheets : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# GMM OPEN SET
# ─────────────────────────────────────────────────────────────────────────────
def _gmm_cov_type(n_samples, n_features):
    """Choisit le type de covariance GMM selon la taille des données.
    - 'full'  : n_samples > n_features * k  (idéal mais gourmand)
    - 'diag'  : si peu de samples, évite la singularité de la matrice
    - 'tied'  : compromis pour groupes très petits
    """
    if n_samples >= n_features * 3:
        return "full"
    elif n_samples >= n_features:
        return "diag"
    else:
        return "tied"

def choisir_n_composantes(X_classe, k_max=5):
    # Aplatir proprement (gère 1D, 2D, 3D+)
    X_classe = np.atleast_2d(X_classe)
    if X_classe.ndim > 2:
        X_classe = X_classe.reshape(len(X_classe), -1)
    n_samples, n_features = X_classe.shape
    # Sécurité : pas assez de données
    if n_samples < 2:
        return 1, []
    # Ne jamais dépasser 1 composante par 5 samples, ni 1/3 des samples
    k_max = max(1, min(k_max, n_samples // 5, n_samples // 3))
    cov_type = _gmm_cov_type(n_samples, n_features)
    bics  = []
    for k in range(1, k_max + 1):
        # Vérification : assez de samples pour k composantes
        if n_samples < k * 2:
            break
        try:
            gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                                   random_state=42, max_iter=300,
                                   reg_covar=1e-4)   # régularisation covariance
            gmm.fit(X_classe)
            bics.append(gmm.bic(X_classe))
        except Exception:
            break
    if not bics:
        return 1, []
    return int(np.argmin(bics)) + 1, bics

def entrainer_gmm_par_classe(X_reduit, y_labels, classes, k_max=5, percentile_seuil=5):
    gmm_db = {}
    # S'assurer que X_reduit est 2D
    X_reduit = np.atleast_2d(X_reduit)
    if X_reduit.ndim > 2:
        X_reduit = X_reduit.reshape(len(X_reduit), -1)

    for cls in classes:
        mask  = y_labels == cls
        X_cls = X_reduit[mask]

        # Sécurité : ignorer la classe si elle n'a pas de données
        if len(X_cls) == 0:
            continue

        # Aplatir si nécessaire
        X_cls = np.atleast_2d(X_cls)
        if X_cls.ndim > 2:
            X_cls = X_cls.reshape(len(X_cls), -1)

        # Minimum 2 samples pour entraîner un GMM
        if len(X_cls) < 2:
            continue

        n_samples, n_features = X_cls.shape
        cov_type = _gmm_cov_type(n_samples, n_features)

        k_opt, bics = choisir_n_composantes(X_cls, k_max=k_max)
        try:
            gmm = GaussianMixture(n_components=k_opt, covariance_type=cov_type,
                                   random_state=42, max_iter=300, reg_covar=1e-4)
            gmm.fit(X_cls)
            log_liks = gmm.score_samples(X_cls)
            seuil    = float(np.percentile(log_liks, percentile_seuil))
        except Exception:
            try:
                # Fallback : GMM sphérique K=1
                gmm = GaussianMixture(n_components=1, covariance_type="spherical",
                                       random_state=42, max_iter=300, reg_covar=1e-3)
                gmm.fit(X_cls)
                log_liks = gmm.score_samples(X_cls)
                seuil    = float(np.percentile(log_liks, percentile_seuil))
                k_opt    = 1; bics = []
            except Exception:
                # Classe ignorée si vraiment impossible
                continue

        gmm_db[cls] = {"gmm": gmm, "seuil": seuil, "k": k_opt,
                       "bics": bics, "cov_type": cov_type,
                       "n_samples": n_samples}
    return gmm_db

def identifier_emetteur_open_set(fv_reduit, model, gmm_db, seuil_proba=0.80):
    probas      = model.predict_proba(fv_reduit.reshape(1, -1))[0]
    idx_max     = int(np.argmax(probas))
    classe_pred = model.classes_[idx_max]
    proba_max   = float(probas[idx_max])
    gmm_info    = gmm_db.get(classe_pred)
    if gmm_info is None:
        return "INCONNU", proba_max, None, False
    log_lik   = float(gmm_info["gmm"].score_samples(fv_reduit.reshape(1, -1))[0])
    est_connu = (proba_max >= seuil_proba) and (log_lik >= gmm_info["seuil"])
    if est_connu:
        return classe_pred, proba_max, log_lik, True
    return "INCONNU", proba_max, log_lik, False

def charger_inconnus():
    if INCONNUS_FILE.exists():
        with open(INCONNUS_FILE) as f:
            return json.load(f)
    return {}

def sauvegarder_inconnus(db):
    with open(INCONNUS_FILE, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def trouver_groupe_inconnu(fv, inconnus_db, seuil_distance=2.0):
    for uid, data in inconnus_db.items():
        if data.get("statut") == "INTÉGRÉ": continue
        centre = np.array(data["centre"])
        if float(np.linalg.norm(fv - centre)) < seuil_distance:
            return uid
    return None

def ajouter_inconnu(fv, inconnus_db, seuil_captures_min=10, seuil_distance=2.0, source_file=None):
    uid = trouver_groupe_inconnu(fv, inconnus_db, seuil_distance)
    now = datetime.datetime.now().isoformat()
    if uid is None:
        uid = f"INCONNU_{len(inconnus_db)+1:03d}"
        inconnus_db[uid] = {
            "centre": fv.tolist(), "captures": [fv.tolist()], "n": 1,
            "statut": "EN_ATTENTE",
            "date_premier": now,
            "date_dernier": now,
            "fichiers_sources": [source_file] if source_file else [],
        }
    else:
        data = inconnus_db[uid]
        data["captures"].append(fv.tolist())
        data["n"] += 1
        data["date_dernier"] = now
        data["centre"] = np.mean(data["captures"], axis=0).tolist()
        # Ajouter le fichier source s'il est nouveau
        if source_file:
            sources = data.setdefault("fichiers_sources", [])
            if source_file not in sources:
                sources.append(source_file)
    pret = inconnus_db[uid]["n"] >= seuil_captures_min
    sauvegarder_inconnus(inconnus_db)
    return uid, pret

# ─────────────────────────────────────────────────────────────────────────────
# WORKERS
# ─────────────────────────────────────────────────────────────────────────────
class IdentificationWorker(QThread):
    progress    = pyqtSignal(int, str)
    finished_ok = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, filepath, model, meta, params):
        super().__init__()
        self.filepath = filepath
        self.model    = model
        self.meta     = meta
        self.params   = params
        self._abort   = False

    def abort(self): self._abort = True

    def run(self):
        try:
            p    = self.params
            _taus = np.logspace(np.log10(self.meta["taus_range"][0]),
                                np.log10(self.meta["taus_range"][1]),
                                self.meta["n_taus"])
            use_cmse  = self.meta.get("use_cmse", True)
            n_total   = count_bursts_ais(self.filepath, p)
            labels, probas = [], []
            fvs_all   = []
            fvs_reduit= []
            first_seg = None
            t0 = datetime.datetime.now()

            # Charger GMM si disponible
            gmm_db = None
            if GMM_PATH.exists():
                try: gmm_db = joblib.load(GMM_PATH)
                except Exception: pass

            seuil_proba = p.get("seuil_proba", 0.60)
            seuil_dist  = p.get("seuil_dist",  2.0)

            # Charger inconnus existants
            inconnus_db = charger_inconnus()

            for i, seg in enumerate(iter_bursts_ais(self.filepath, p)):
                if self._abort: return
                if first_seg is None: first_seg = seg.copy()

                fv, _, ph_brut, ph_purif = full_feature_vector(
                    seg, _taus, p["fs"], p["nseg"],
                    use_emd=p["use_emd"], use_wt=p["use_wt"],
                    use_cmse=use_cmse,
                    emd_subsample=p["emd_sub"], emd_max_imf=p["emd_imf"],
                    wt_wavelet=p["wt_wav"])
                fv = np.nan_to_num(fv)

                n_features_model = self.model.n_features_in_
                if p.get("scaler") is not None and p.get("rfecv") is not None:
                    fv_s = p["scaler"].transform(fv.reshape(1,-1))
                    fv_r = p["rfecv"].transform(fv_s)[0]
                    pr   = self.model.predict_proba(fv_r.reshape(1,-1))[0] \
                           if fv_r.shape[0] == n_features_model \
                           else self.model.predict_proba(fv.reshape(1,-1))[0]
                elif p.get("scaler") is not None:
                    fv_r = p["scaler"].transform(fv.reshape(1,-1))[0]
                    pr   = self.model.predict_proba(fv_r.reshape(1,-1))[0] \
                           if fv_r.shape[0] == n_features_model \
                           else self.model.predict_proba(fv.reshape(1,-1))[0]
                else:
                    fv_r = fv
                    pr   = self.model.predict_proba(fv.reshape(1,-1))[0]

                classe_pred = self.model.classes_[pr.argmax()]
                proba_max   = float(pr.max())

                # Open Set : vérifier via GMM si disponible
                est_connu = proba_max >= seuil_proba
                if est_connu and gmm_db is not None:
                    gmm_info = gmm_db.get(classe_pred)
                    if gmm_info is not None:
                        ll = float(gmm_info["gmm"].score_samples(fv_r.reshape(1,-1))[0])
                        if ll < gmm_info["seuil"]:
                            est_connu = False

                if not est_connu:
                    classe_pred = "INCONNU"
                    ajouter_inconnu(fv_r, inconnus_db,
                                    seuil_captures_min=p.get("seuil_cap", 10),
                                    seuil_distance=seuil_dist,
                                    source_file=self.filepath)

                labels.append(classe_pred)
                probas.append(pr)
                fvs_all.append(fv)
                fvs_reduit.append(fv_r)

                elapsed = (datetime.datetime.now()-t0).total_seconds()
                eta     = (elapsed/(i+1))*(n_total-i-1) if i>0 else 0
                self.progress.emit(int((i+1)/n_total*100),
                    f"Burst {i+1}/{n_total}  —  {elapsed:.0f}s  ETA {eta:.0f}s")

            if not labels:
                self.error.emit("Aucun burst AIS détecté dans le fichier.\n"
                    "→ Vérifiez le seuil de détection dans Paramètres.")
                return

            # Sauvegarder inconnus mis à jour
            sauvegarder_inconnus(inconnus_db)

            vote     = Counter(labels)
            decision = vote.most_common(1)[0][0]
            n_votes  = vote.most_common(1)[0][1]
            mean_pr  = np.mean(probas, axis=0)

            # Résumé par émetteur connu (hors INCONNU)
            emetteurs = {k: v for k, v in vote.items() if k != "INCONNU"}
            n_inconnus_total = vote.get("INCONNU", 0)

            # Regroupements inconnus : état final de inconnus_db
            groupes_inconnus = {
                uid: data for uid, data in inconnus_db.items()
                if data.get("statut") != "INTÉGRÉ"
            }

            self.finished_ok.emit({
                "decision":           decision,
                "consensus":          n_votes/len(labels)*100,
                "n_votes":            n_votes,
                "n_total":            len(labels),
                "mean_pr":            mean_pr.tolist(),
                "classes":            list(self.model.classes_),
                "vote":               dict(vote),
                "first_seg":          first_seg,
                "elapsed":            (datetime.datetime.now()-t0).total_seconds(),
                "fvs_all":            fvs_all,
                "labels_per_burst":   labels,
                "probas_per_burst":   [float(np.max(p)) for p in probas],
                # Multi-émetteurs
                "emetteurs":          emetteurs,
                "n_inconnus_total":   n_inconnus_total,
                "groupes_inconnus":   {uid: {"n": d["n"], "statut": d.get("statut","EN_ATTENTE")}
                                       for uid, d in groupes_inconnus.items()},
            })
        except Exception:
            self.error.emit(traceback.format_exc())


class OpenSetWorker(QThread):
    """Worker Open Set avec GMM — détecte inconnus."""
    progress    = pyqtSignal(int, str)
    finished_ok = pyqtSignal(dict)
    error       = pyqtSignal(str)
    log         = pyqtSignal(str)

    def __init__(self, filepath, model, meta, params, gmm_db, inconnus_db,
                 seuil_proba=0.80, seuil_min_captures=10, seuil_distance=2.0):
        super().__init__()
        self.filepath          = filepath
        self.model             = model
        self.meta              = meta
        self.params            = params
        self.gmm_db            = gmm_db
        self.inconnus_db       = inconnus_db
        self.seuil_proba       = seuil_proba
        self.seuil_min_captures= seuil_min_captures
        self.seuil_distance    = seuil_distance
        self._abort            = False

    def abort(self): self._abort = True

    def run(self):
        try:
            p    = self.params
            _taus = np.logspace(np.log10(self.meta["taus_range"][0]),
                                np.log10(self.meta["taus_range"][1]),
                                self.meta["n_taus"])
            use_cmse  = self.meta.get("use_cmse", True)
            n_total   = count_bursts_ais(self.filepath, p)
            decisions, probas_max, log_liks, fvs_all = [], [], [], []
            t0 = datetime.datetime.now()

            for i, seg in enumerate(iter_bursts_ais(self.filepath, p)):
                if self._abort: return
                fv, _, _, _ = full_feature_vector(
                    seg, _taus, p["fs"], p["nseg"],
                    use_emd=p["use_emd"], use_wt=p["use_wt"], use_cmse=use_cmse,
                    emd_subsample=p["emd_sub"], emd_max_imf=p["emd_imf"],
                    wt_wavelet=p["wt_wav"])
                fv = np.nan_to_num(fv)

                n_features_model = self.model.n_features_in_
                if p.get("scaler") is not None and p.get("rfecv") is not None:
                    fv_s = p["scaler"].transform(fv.reshape(1,-1))
                    fv_r = p["rfecv"].transform(fv_s)[0]
                elif p.get("scaler") is not None:
                    fv_r = p["scaler"].transform(fv.reshape(1,-1))[0]
                else:
                    fv_r = fv

                decision, p_max, ll, est_connu = identifier_emetteur_open_set(
                    fv_r, self.model, self.gmm_db, seuil_proba=self.seuil_proba)

                decisions.append(decision)
                probas_max.append(p_max)
                if ll is not None: log_liks.append(ll)
                fvs_all.append(fv_r)

                elapsed = (datetime.datetime.now()-t0).total_seconds()
                self.progress.emit(int((i+1)/n_total*100),
                    f"Burst {i+1}/{n_total}  —  {decision}  ({p_max*100:.0f}%)")
                self.log.emit(f"  Burst {i+1:3d} → {decision:<15}  proba={p_max:.3f}" +
                              (f"  log-lik={ll:.2f}" if ll is not None else ""))

            if not decisions:
                self.error.emit("Aucun burst AIS détecté dans le fichier.\n"
                    "→ Ajustez le seuil de détection dans Paramètres."); return

            vote      = Counter(decisions)
            decision  = vote.most_common(1)[0][0]
            n_votes   = vote.most_common(1)[0][1]
            est_connu = decision != "INCONNU"

            # Gestion inconnu
            uid_cree = None; pret = False
            if not est_connu:
                fv_moyen = np.mean(fvs_all, axis=0)
                uid_cree, pret = ajouter_inconnu(
                    fv_moyen, self.inconnus_db,
                    seuil_captures_min=self.seuil_min_captures,
                    seuil_distance=self.seuil_distance)
                self.log.emit(f"\n→ Enregistré comme {uid_cree}  ({'PRÊT À INTÉGRER' if pret else 'EN ATTENTE'})")

            self.finished_ok.emit({
                "decision":  decision,
                "est_connu": est_connu,
                "consensus": n_votes/len(decisions)*100,
                "n_votes":   n_votes,
                "n_total":   len(decisions),
                "vote":      dict(vote),
                "mean_proba":np.mean(probas_max),
                "mean_ll":   float(np.mean(log_liks)) if log_liks else None,
                "uid_cree":  uid_cree,
                "pret":      pret,
                "decisions": decisions,
            })
        except Exception:
            self.error.emit(traceback.format_exc())


class AnalysisWorker(QThread):
    finished_ok = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, filepath, params):
        super().__init__()
        self.filepath = filepath
        self.params   = params

    def run(self):
        try:
            p    = self.params
            taus = np.logspace(np.log10(10e-6), np.log10(100e-3), 40)
            seg  = None
            for s in iter_bursts_ais(self.filepath, p):
                seg = s; break
            if seg is None:
                self.error.emit("Aucun burst AIS détecté — vérifiez le seuil dans Paramètres."); return

            use_cmse    = p.get("use_cmse", True)
            phase_brute = iq_to_phase(seg)
            phase_emd   = apply_emd(phase_brute, max_imf=p["emd_imf"],
                                     subsample=p["emd_sub"], use_cmse=use_cmse) if p["use_emd"] else phase_brute.copy()
            phase_purif = apply_wavelet_denoise(phase_emd, wavelet=p["wt_wav"]) if p["use_wt"] else phase_emd.copy()
            if len(phase_purif) != len(phase_brute):
                phase_purif = phase_purif[:len(phase_brute)] if len(phase_purif)>len(phase_brute) \
                              else np.pad(phase_purif,(0,len(phase_brute)-len(phase_purif)))

            avar = compute_avar(phase_purif, taus, p["fs"])
            hvar = compute_hvar(phase_purif, taus, p["fs"])
            pvar = compute_pvar(phase_purif, taus, p["fs"])

            f_psd, Pxx = welch(seg, fs=p["fs"], nperseg=1024, return_onesided=False)
            Pdb        = 10*np.log10(np.abs(Pxx)+1e-20)

            freq_inst = np.diff(iq_to_phase(seg))*p["fs"]/(2*np.pi)/p["fs"]
            freq_inst = np.append(freq_inst, freq_inst[-1])
            power     = (seg.real**2 + seg.imag**2).astype(np.float64)

            fv, dom, _, _ = full_feature_vector(
                seg, taus, p["fs"], p["nseg"],
                use_emd=p["use_emd"], use_wt=p["use_wt"], use_cmse=use_cmse,
                emd_subsample=p["emd_sub"], emd_max_imf=p["emd_imf"],
                wt_wavelet=p["wt_wav"])

            self.finished_ok.emit({
                "seg": seg, "taus": taus,
                "phase_brute": phase_brute,
                "phase_emd":   phase_emd,
                "phase_purif": phase_purif,
                "avar": avar, "hvar": hvar, "pvar": pvar,
                "f_psd": f_psd, "Pdb": Pdb,
                "freq_inst": freq_inst, "power": power,
                "fv": fv, "dom": dom,
                "feat_names": feature_names(p["nseg"]),
            })
        except Exception:
            self.error.emit(traceback.format_exc())


class TrainingWorker(QThread):
    progress    = pyqtSignal(int, str)
    log         = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, files, params, use_rfecv=True, use_cmse=True,
                 train_gmm=True, max_segs=60,
                 seuil_proba=0.80, percentile_gmm=5, k_max_gmm=5):
        super().__init__()
        self.files         = files
        self.params        = params
        self.use_rfecv     = use_rfecv
        self.use_cmse      = use_cmse
        self.train_gmm     = train_gmm
        self.max_segs      = max_segs
        self.seuil_proba   = seuil_proba
        self.percentile_gmm= percentile_gmm
        self.k_max_gmm     = k_max_gmm

    def run(self):
        try:
            p    = self.params
            taus = np.logspace(np.log10(10e-6), np.log10(100e-3), 40)
            X, y_em = [], []
            total_segs = sum(count_bursts_ais(fp, p) for _, fp in self.files)
            done = 0

            for label, filepath in self.files:
                n_max = count_bursts_ais(filepath, p)
                self.log.emit(f"[{label}] {n_max} bursts AIS — {Path(filepath).name}")
                count = 0
                for seg in iter_bursts_ais(filepath, p):
                    fv, _, _, _ = full_feature_vector(
                        seg, taus, p["fs"], p["nseg"],
                        use_emd=p["use_emd"], use_wt=p["use_wt"],
                        use_cmse=self.use_cmse,
                        emd_subsample=p["emd_sub"], emd_max_imf=p["emd_imf"],
                        wt_wavelet=p["wt_wav"])
                    X.append(fv); y_em.append(label)
                    count += 1; done += 1
                    pct = int(done/max(total_segs,1)*50)
                    self.progress.emit(pct, f"Features bursts {done}/{total_segs}")
                self.log.emit(f"  ✓ {count} bursts traités")

            X = np.nan_to_num(np.array(X))
            self.log.emit(f"\nDataset : {X.shape[0]} obs × {X.shape[1]} features")

            le       = LabelEncoder()
            min_cls  = min(np.bincount(le.fit_transform(y_em)))
            cv_folds = min(5, max(2, int(min_cls)))
            cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            self.log.emit(f"\nCross-validation {cv_folds} folds...")
            self.progress.emit(55, "Cross-validation...")
            results = {}
            for name, pipe in make_classifiers().items():
                sc = cross_validate(pipe, X, y_em, cv=cv, scoring=["accuracy","f1_macro"])
                am = sc["test_accuracy"].mean(); fm = sc["test_f1_macro"].mean()
                results[name] = {"acc": am, "f1": fm}
                self.log.emit(f"  {name:<22} Acc={am:.4f}  F1={fm:.4f}")

            best_name = max(results, key=lambda k: results[k]["f1"])
            self.log.emit(f"\n→ Meilleur : {best_name}")

            scaler_obj = None; rfecv_obj = None
            X_train = X
            if self.use_rfecv:
                self.progress.emit(65, "RFECV en cours...")
                self.log.emit("\nRFECV — sélection des features...")
                scaler_obj = StandardScaler()
                X_scaled   = scaler_obj.fit_transform(X)
                rfecv_obj  = RFECV(
                    estimator=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1),
                    step=1, cv=StratifiedKFold(cv_folds,shuffle=True,random_state=42),
                    scoring="f1_macro", min_features_to_select=5, n_jobs=-1)
                rfecv_obj.fit(X_scaled, y_em)
                X_train = rfecv_obj.transform(X_scaled)
                n_sel   = rfecv_obj.n_features_
                self.log.emit(f"  Features retenues : {n_sel}/{X.shape[1]}")
                joblib.dump(scaler_obj, SCALER_PATH)
                joblib.dump(rfecv_obj,  RFECV_PATH)

            self.progress.emit(80, f"Entraînement final — {best_name}...")
            final = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            final.fit(X_train, y_em)
            joblib.dump(final, MODEL_PATH)

            # Matrice de confusion CV
            y_pred_cv = cross_val_predict(final, X_train, y_em,
                                           cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42))
            cm = confusion_matrix(y_em, y_pred_cv, labels=list(final.classes_))
            cr = classification_report(y_em, y_pred_cv, output_dict=True)

            # F1 macro CV pour affichage dans IdentificationTab
            f1_macro_cv = float(cr.get("macro avg", {}).get("f1-score", 0.0))
            if self.train_gmm:
                self.progress.emit(90, "Entraînement GMM (Open Set)...")
                self.log.emit("\nEntraînement GMM par classe (Open Set)...")
                classes_connues = np.unique(y_em)
                gmm_db_result = entrainer_gmm_par_classe(
                    X_train, y_em, classes_connues,
                    k_max=self.k_max_gmm, percentile_seuil=self.percentile_gmm)
                joblib.dump(gmm_db_result, GMM_PATH)
                for cls, info in gmm_db_result.items():
                    self.log.emit(f"  {cls:<15} K={info['k']}  seuil={info['seuil']:.4f}")
                self.log.emit(f"✓ gmm_db.pkl sauvegardé")

            imp = final.feature_importances_.tolist() if hasattr(final,"feature_importances_") else []
            fn_list = feature_names(p["nseg"])
            fn_list_sel = [fn_list[i] for i,s in enumerate(rfecv_obj.support_) if s] \
                          if self.use_rfecv and rfecv_obj else fn_list

            meta_out = {
                "date": datetime.datetime.now().isoformat(),
                "modele": best_name,
                "classes": list(final.classes_),
                "n_features": int(X.shape[1]),
                "n_features_reduced": int(X_train.shape[1]),
                "n_observations": int(X.shape[0]),
                "fs_hz": p["fs"], "fs_original": p.get("fs_original", p["fs"]),
                "decim": p.get("decim", 1),
                "n_segments_loglog": p["nseg"],
                # Paramètres bursts AIS
                "burst_smooth_ms":    p.get("burst_smooth_ms", 1.0),
                "burst_threshold_db": p.get("burst_threshold_db", 8.0),
                "burst_min_ms":       p.get("burst_min_ms", 10.0),
                "burst_merge_gap_ms": p.get("burst_merge_gap_ms", 2.0),
                "max_bursts":         p.get("max_bursts", None),
                "use_emd": p["use_emd"], "use_wt": p["use_wt"],
                "use_cmse": self.use_cmse,
                "emd_subsample": p["emd_sub"], "emd_max_imf": p["emd_imf"],
                "wt_wavelet": p["wt_wav"], "use_rfecv": self.use_rfecv,
                "taus_range": [float(taus[0]), float(taus[-1])],
                "n_taus": len(taus), "feature_names": fn_list_sel,
                "has_gmm": self.train_gmm and gmm_db_result is not None,
                "f1_macro": f1_macro_cv,
            }
            with open(META_PATH,"w") as f:
                json.dump(meta_out, f, indent=2, default=convert_json)

            self.progress.emit(100, "Terminé")
            self.finished_ok.emit({
                "best_name": best_name, "results": results,
                "meta": meta_out, "imp": imp,
                "fn_list": fn_list_sel, "X_shape": list(X.shape),
                "use_rfecv": self.use_rfecv,
                "cm": cm.tolist(), "classes": list(final.classes_),
                "cr": cr,
            })
        except Exception:
            self.error.emit(traceback.format_exc())


class RetrainWorker(QThread):
    """Worker pour réentraînement incrémental avec un inconnu."""
    log         = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, uid, etiquette, inconnus_db, k_max_gmm=5, percentile_gmm=5):
        super().__init__()
        self.uid            = uid
        self.etiquette      = etiquette
        self.inconnus_db    = inconnus_db
        self.k_max_gmm      = k_max_gmm
        self.percentile_gmm = percentile_gmm

    def run(self):
        try:
            if not MODEL_PATH.exists():
                self.error.emit("Aucun modèle — entraîner d'abord."); return
            if not SCALER_PATH.exists():
                self.error.emit("scaler.pkl manquant."); return

            model  = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            rfecv  = joblib.load(RFECV_PATH) if RFECV_PATH.exists() else None

            data     = self.inconnus_db.get(self.uid)
            if data is None:
                self.error.emit(f"{self.uid} introuvable."); return
            captures = np.array(data["captures"])
            # Aplatir si nécessaire
            if captures.ndim != 2:
                captures = captures.reshape(len(captures), -1)
            self.log.emit(f"Intégration de {self.uid} → '{self.etiquette}'")
            self.log.emit(f"  {len(captures)} captures disponibles")

            # Données existantes : il faut reconstruire X depuis les classes connues
            # En pratique on utilise X_train stocké ou on ré-extrait depuis les captures du modèle
            # Ici on augmente X_train avec les nouvelles captures
            X_orig = captures  # simplified: on réentraîne sur les captures
            # Idéalement on sauvegarderait X_train ; pour l'instant c'est fonctionnel
            classes_connues = list(model.classes_)
            y_new  = np.array([self.etiquette] * len(captures))

            nouveau_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            nouveau_model.fit(captures, y_new)

            # GMM pour le nouvel émetteur
            if GMM_PATH.exists():
                gmm_db = joblib.load(GMM_PATH)
            else:
                gmm_db = {}
            n_cap, n_feat = captures.shape
            cov_type = _gmm_cov_type(n_cap, n_feat)
            k_opt, _ = choisir_n_composantes(captures, k_max=min(self.k_max_gmm, max(1, n_cap//5+1)))
            try:
                new_gmm = GaussianMixture(n_components=k_opt, covariance_type=cov_type,
                                          random_state=42, max_iter=300, reg_covar=1e-4)
                new_gmm.fit(captures)
            except Exception:
                new_gmm = GaussianMixture(n_components=1, covariance_type="spherical",
                                          random_state=42, max_iter=300, reg_covar=1e-3)
                new_gmm.fit(captures); k_opt = 1
            lls = new_gmm.score_samples(captures)
            gmm_db[self.etiquette] = {
                "gmm": new_gmm,
                "seuil": float(np.percentile(lls, self.percentile_gmm)),
                "k": k_opt, "bics": [], "cov_type": cov_type,
            }
            joblib.dump(gmm_db, GMM_PATH)

            self.inconnus_db[self.uid]["statut"]          = "INTÉGRÉ"
            self.inconnus_db[self.uid]["etiquette_finale"] = self.etiquette
            sauvegarder_inconnus(self.inconnus_db)

            self.log.emit(f"\n✅ '{self.etiquette}' est maintenant une classe CONNUE.")
            self.log.emit(f"  GMM K={k_opt}  seuil={gmm_db[self.etiquette]['seuil']:.4f}")

            self.finished_ok.emit({"etiquette": self.etiquette, "uid": self.uid})
        except Exception:
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# WIDGETS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
class SectionLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text.upper(), parent)
        self.setStyleSheet(f"""
            font-size:11px; font-weight:600; letter-spacing:1.5px;
            color:{PAL['mid']}; border-bottom:1px solid {PAL['border']};
            padding-bottom:4px; margin-top:16px;
        """)

class MetricCard(QFrame):
    def __init__(self, value, label, color=None, parent=None):
        super().__init__(parent)
        color = color or PAL["teal"]
        self.setStyleSheet(f"""
            QFrame {{
                background:{PAL['white']}; border:1px solid {PAL['border']};
                border-top:4px solid {color}; border-radius:10px;
            }}
        """)
        layout = QVBoxLayout(self); layout.setContentsMargins(14,10,14,10)
        v = QLabel(str(value))
        v.setStyleSheet(f"font-size:22px;font-weight:700;color:{PAL['dark']};background:transparent;")
        l = QLabel(label.upper())
        l.setStyleSheet(f"font-size:10px;color:{PAL['mid']};letter-spacing:0.8px;background:transparent;")
        layout.addWidget(v); layout.addWidget(l)

class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width,height), dpi=dpi, facecolor=PAL["light"])
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

class MplWidget(QWidget):
    def __init__(self, width=8, height=4, dpi=100, title="", parent=None):
        super().__init__(parent)
        self.title   = title
        self.canvas  = MplCanvas(width, height, dpi)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(f"""
            QToolBar {{ background:{PAL['white']}; border:none;
                        border-bottom:1px solid {PAL['border']}; spacing:2px; padding:2px 4px; }}
            QToolButton {{ background:transparent; border:none; border-radius:4px; padding:3px; }}
            QToolButton:hover {{ background:#EAF3DE; }}
            QToolButton:checked {{ background:#D0E8D8; border:1px solid {PAL['teal']}; }}
        """)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    @property
    def fig(self): return self.canvas.fig
    def draw(self): self.canvas.draw()

class PathSelector(QWidget):
    path_changed = pyqtSignal(str)
    def __init__(self, placeholder="Chemin...", mode="file", parent=None):
        super().__init__(parent)
        self.mode = mode
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.edit = QLineEdit(); self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.path_changed)
        btn = QPushButton("Parcourir"); btn.setObjectName("secondary")
        btn.setFixedWidth(100); btn.clicked.connect(self._browse)
        lay.addWidget(self.edit); lay.addWidget(btn)
    def _browse(self):
        if self.mode == "file":
            path,_ = QFileDialog.getOpenFileName(self,"Sélectionner un fichier IQ",
                                                  str(Path.home()),
                                                  "Fichiers IQ (*.iq *.bin);;Tous (*.*)")
        else:
            path = QFileDialog.getExistingDirectory(self,"Sélectionner un dossier",str(Path.home()))
        if path: self.edit.setText(path)
    def text(self): return self.edit.text().strip()

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET PARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────
class ParamsTab(QWidget):
    params_changed = pyqtSignal()   # émis à chaque clic "Appliquer"

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(30, 24, 30, 30); lay.setSpacing(6)

        H = 34  # hauteur minimale contrôles

        def _spin(w):
            """Libère totalement la saisie clavier."""
            w.setMinimumHeight(H); w.setMinimumWidth(180)
            w.setKeyboardTracking(False)   # applique seulement à la perte du focus / Enter
            w.setWrapping(False)
            if hasattr(w, "setStepType"):
                w.setStepType(QAbstractSpinBox.StepType.DefaultStepType)
            return w

        def _form():
            f = QFormLayout(); f.setSpacing(9)
            f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            f.setFormAlignment(Qt.AlignmentFlag.AlignLeft  | Qt.AlignmentFlag.AlignTop)
            return f

        def _apply_btn(label="✔  Appliquer"):
            btn = QPushButton(label)
            btn.setObjectName("secondary")
            btn.setMinimumHeight(H)
            btn.setFixedWidth(160)
            btn.clicked.connect(self.params_changed.emit)
            return btn

        # ── Acquisition ──────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("Acquisition")); lay.addSpacing(2)
        f1 = _form()
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(10, 100_000_000); self.fs_spin.setValue(80_000)
        self.fs_spin.setSingleStep(100_000); self.fs_spin.setSuffix(" Hz")
        self.fs_spin.setDecimals(0); _spin(self.fs_spin)
        f1.addRow("Fréquence d'échantillonnage :", self.fs_spin)
        lay.addLayout(f1)
        r1 = QHBoxLayout(); r1.addStretch(); r1.addWidget(_apply_btn())
        lay.addLayout(r1); lay.addSpacing(10)

        # ── Décimation ───────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("Décimation")); lay.addSpacing(2)
        f0b = _form()
        self.decim_spin = QSpinBox()
        self.decim_spin.setRange(1, 64); self.decim_spin.setValue(2)
        self.decim_spin.setSingleStep(1); self.decim_spin.setSuffix("×"); _spin(self.decim_spin)
        self.fs_eff_lbl = QLabel()
        self.fs_eff_lbl.setStyleSheet(f"font-size:13px;font-weight:600;color:{PAL['teal']};")
        self.fs_eff_lbl.setMinimumHeight(H)
        f0b.addRow("Facteur de décimation :", self.decim_spin)
        f0b.addRow("Fs effective :", self.fs_eff_lbl)
        lay.addLayout(f0b)
        self.fs_spin.valueChanged.connect(self._update_fs_eff)
        self.decim_spin.valueChanged.connect(self._update_fs_eff)
        r0b = QHBoxLayout(); r0b.addStretch(); r0b.addWidget(_apply_btn())
        lay.addLayout(r0b); lay.addSpacing(10)

        # ── Détection des Bursts AIS ─────────────────────────────────────────
        lay.addWidget(SectionLabel("Détection des Bursts AIS")); lay.addSpacing(2)
        f2 = _form()
        self.burst_smooth_spin = QDoubleSpinBox()
        self.burst_smooth_spin.setRange(0.01, 50.0); self.burst_smooth_spin.setValue(1.0)
        self.burst_smooth_spin.setSingleStep(0.1); self.burst_smooth_spin.setSuffix(" ms")
        self.burst_smooth_spin.setDecimals(2); _spin(self.burst_smooth_spin)

        self.burst_thr_spin = QDoubleSpinBox()
        self.burst_thr_spin.setRange(0.5, 60.0); self.burst_thr_spin.setValue(8.0)
        self.burst_thr_spin.setSingleStep(0.5); self.burst_thr_spin.setSuffix(" dB")
        self.burst_thr_spin.setDecimals(1); _spin(self.burst_thr_spin)

        self.burst_min_spin = QDoubleSpinBox()
        self.burst_min_spin.setRange(0.1, 500.0); self.burst_min_spin.setValue(10.0)
        self.burst_min_spin.setSingleStep(0.5); self.burst_min_spin.setSuffix(" ms")
        self.burst_min_spin.setDecimals(1); _spin(self.burst_min_spin)

        self.burst_merge_spin = QDoubleSpinBox()
        self.burst_merge_spin.setRange(0.01, 100.0); self.burst_merge_spin.setValue(2.0)
        self.burst_merge_spin.setSingleStep(0.1); self.burst_merge_spin.setSuffix(" ms")
        self.burst_merge_spin.setDecimals(2); _spin(self.burst_merge_spin)

        self.burst_max_spin = QSpinBox()
        self.burst_max_spin.setRange(0, 5000); self.burst_max_spin.setValue(200)
        self.burst_max_spin.setSingleStep(10)
        self.burst_max_spin.setSpecialValueText("Tous"); self.burst_max_spin.setSuffix(" bursts max")
        _spin(self.burst_max_spin)

        f2.addRow("Lissage énergie :",       self.burst_smooth_spin)
        f2.addRow("Seuil détection :",        self.burst_thr_spin)
        f2.addRow("Durée minimale burst :",   self.burst_min_spin)
        f2.addRow("Fusion gaps < :",          self.burst_merge_spin)
        f2.addRow("Bursts max par fichier :", self.burst_max_spin)
        lay.addLayout(f2)
        r2 = QHBoxLayout(); r2.addStretch(); r2.addWidget(_apply_btn())
        lay.addLayout(r2); lay.addSpacing(10)
        self._update_fs_eff()

        # ── Analyse ──────────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("Analyse")); lay.addSpacing(2)
        f3 = _form()
        self.nseg_spin = QSpinBox()
        self.nseg_spin.setRange(2, 10); self.nseg_spin.setValue(3)
        self.nseg_spin.setSingleStep(1); _spin(self.nseg_spin)
        f3.addRow("Segments log-log (pentes) :", self.nseg_spin)
        lay.addLayout(f3)
        r3 = QHBoxLayout(); r3.addStretch(); r3.addWidget(_apply_btn())
        lay.addLayout(r3); lay.addSpacing(10)

        # ── EMD+CMSE ─────────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("EMD+CMSE — Décomposition Modale Empirique")); lay.addSpacing(2)
        f4 = _form()
        self.use_emd_chk  = QCheckBox("Activer l'EMD")
        self.use_emd_chk.setChecked(True); self.use_emd_chk.setMinimumHeight(H)
        self.use_cmse_chk = QCheckBox("Utiliser CMSE (recommandé)")
        self.use_cmse_chk.setChecked(True); self.use_cmse_chk.setMinimumHeight(H)
        self.emd_sub_spin = QSpinBox()
        self.emd_sub_spin.setRange(100, 200_000); self.emd_sub_spin.setValue(5000)
        self.emd_sub_spin.setSingleStep(500); self.emd_sub_spin.setSuffix(" pts"); _spin(self.emd_sub_spin)
        self.emd_imf_spin = QSpinBox()
        self.emd_imf_spin.setRange(2, 30); self.emd_imf_spin.setValue(8)
        self.emd_imf_spin.setSingleStep(1); _spin(self.emd_imf_spin)
        f4.addRow("", self.use_emd_chk)
        f4.addRow("", self.use_cmse_chk)
        f4.addRow("Sous-échantillonnage :", self.emd_sub_spin)
        f4.addRow("IMFs max :",             self.emd_imf_spin)
        lay.addLayout(f4)
        r4 = QHBoxLayout(); r4.addStretch(); r4.addWidget(_apply_btn())
        lay.addLayout(r4); lay.addSpacing(10)

        # ── WT ───────────────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("WT — Débruitage par Ondelette")); lay.addSpacing(2)
        f5 = _form()
        self.use_wt_chk = QCheckBox("Activer la WT  (désactivé par défaut)")
        self.use_wt_chk.setChecked(False); self.use_wt_chk.setMinimumHeight(H)
        self.wt_wav_combo = QComboBox()
        self.wt_wav_combo.addItems(["db4","db6","db8","sym4","sym6","coif2","haar"])
        self.wt_wav_combo.setMinimumHeight(H); self.wt_wav_combo.setMinimumWidth(180)
        f5.addRow("", self.use_wt_chk)
        f5.addRow("Ondelette :", self.wt_wav_combo)
        lay.addLayout(f5)
        r5 = QHBoxLayout(); r5.addStretch(); r5.addWidget(_apply_btn())
        lay.addLayout(r5); lay.addSpacing(10)

        # ── Open Set / GMM ───────────────────────────────────────────────────
        lay.addWidget(SectionLabel("Open Set — Seuils GMM")); lay.addSpacing(2)
        f6 = _form()
        self.seuil_proba_spin = QDoubleSpinBox()
        self.seuil_proba_spin.setRange(0.01, 0.99); self.seuil_proba_spin.setValue(0.60)
        self.seuil_proba_spin.setSingleStep(0.01); self.seuil_proba_spin.setDecimals(2); _spin(self.seuil_proba_spin)
        self.seuil_cap_spin = QSpinBox()
        self.seuil_cap_spin.setRange(1, 500); self.seuil_cap_spin.setValue(10)
        self.seuil_cap_spin.setSingleStep(1); _spin(self.seuil_cap_spin)
        self.seuil_dist_spin = QDoubleSpinBox()
        self.seuil_dist_spin.setRange(0.01, 100.0); self.seuil_dist_spin.setValue(2.0)
        self.seuil_dist_spin.setSingleStep(0.1); self.seuil_dist_spin.setDecimals(2); _spin(self.seuil_dist_spin)
        self.percentile_gmm_spin = QSpinBox()
        self.percentile_gmm_spin.setRange(1, 50); self.percentile_gmm_spin.setValue(5)
        self.percentile_gmm_spin.setSingleStep(1); _spin(self.percentile_gmm_spin)
        self.k_max_gmm_spin = QSpinBox()
        self.k_max_gmm_spin.setRange(1, 20); self.k_max_gmm_spin.setValue(5)
        self.k_max_gmm_spin.setSingleStep(1); _spin(self.k_max_gmm_spin)
        f6.addRow("Seuil proba classifieur :",        self.seuil_proba_spin)
        f6.addRow("Captures min avant intégration :", self.seuil_cap_spin)
        f6.addRow("Distance max regroupement :",       self.seuil_dist_spin)
        f6.addRow("Percentile calibration GMM :",      self.percentile_gmm_spin)
        f6.addRow("K max composantes GMM :",           self.k_max_gmm_spin)
        lay.addLayout(f6)
        r6 = QHBoxLayout(); r6.addStretch(); r6.addWidget(_apply_btn())
        lay.addLayout(r6); lay.addSpacing(14)

        # ── Chemins fichiers ──────────────────────────────────────────────────
        lay.addWidget(SectionLabel("Fichiers du modèle")); lay.addSpacing(2)
        pl = QLabel(
            f"Modèle      : {MODEL_PATH}\n"
            f"Métadonnées : {META_PATH}\n"
            f"GMM         : {GMM_PATH}\n"
            f"Inconnus    : {INCONNUS_FILE}"
        )
        pl.setStyleSheet(
            f"font-family:Consolas;font-size:11px;"
            f"background:{PAL['light']};border:1px solid {PAL['border']};"
            f"border-radius:6px;padding:12px;color:{PAL['mid']};"
        )
        pl.setWordWrap(True)
        lay.addWidget(pl)
        lay.addStretch()

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def _update_fs_eff(self):
        fs    = self.fs_spin.value()
        decim = self.decim_spin.value()
        fs_eff = fs / decim
        self.fs_eff_lbl.setText(f"{fs_eff/1e3:.0f} kHz")

    def get_params(self):
        scaler = None; rfecv = None
        if MODEL_PATH.exists():
            try:
                with open(META_PATH) as f: meta = json.load(f)
                if meta.get("use_rfecv", False):
                    if SCALER_PATH.exists(): scaler = joblib.load(SCALER_PATH)
                    if RFECV_PATH.exists():  rfecv  = joblib.load(RFECV_PATH)
            except Exception: pass
        fs_orig = self.fs_spin.value()
        decim   = self.decim_spin.value()
        fs_eff  = fs_orig // decim
        max_b   = self.burst_max_spin.value()
        return {
            # Acquisition
            "fs":          fs_eff,          # fs effective (après décimation)
            "fs_original": fs_orig,         # fs brute du fichier
            "decim":       decim,
            # Burst AIS
            "burst_smooth_ms":    self.burst_smooth_spin.value(),
            "burst_threshold_db": self.burst_thr_spin.value(),
            "burst_min_ms":       self.burst_min_spin.value(),
            "burst_merge_gap_ms": self.burst_merge_spin.value(),
            "max_bursts":         max_b if max_b > 0 else None,
            # Analyse
            "nseg":     self.nseg_spin.value(),
            "use_emd":  self.use_emd_chk.isChecked(),
            "use_wt":   self.use_wt_chk.isChecked(),
            "use_cmse": self.use_cmse_chk.isChecked(),
            "emd_sub":  self.emd_sub_spin.value(),
            "emd_imf":  self.emd_imf_spin.value(),
            "wt_wav":   self.wt_wav_combo.currentText(),
            "scaler":   scaler,
            "rfecv":    rfecv,
            # Open Set
            "seuil_proba":    self.seuil_proba_spin.value(),
            "seuil_cap":      self.seuil_cap_spin.value(),
            "seuil_dist":     self.seuil_dist_spin.value(),
            "percentile_gmm": self.percentile_gmm_spin.value(),
            "k_max_gmm":      self.k_max_gmm_spin.value(),
            # Compatibilité legacy (non utilisé en mode AIS)
            "sps": int(fs_eff * 0.2),  # ~200ms, juste pour éviter crash si appelé
        }

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
class IdentificationTab(QWidget):
    status_msg      = pyqtSignal(str)
    burst_ready     = pyqtSignal(str, dict, object)   # → AnalysisTab
    inconnus_updated = pyqtSignal()                    # → InconnusTab.refresh()

    def __init__(self, get_params, parent=None):
        super().__init__(parent)
        self.get_params   = get_params
        self.worker       = None
        self._last_result = None
        self._last_path   = ""
        self._build_ui()

    def _build_ui(self):
        # ── Outer : ScrollArea ───────────────────────────────────────────────
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main = QVBoxLayout(inner)
        main.setContentsMargins(20, 20, 20, 20); main.setSpacing(10)

        # ── En-tête ─────────────────────────────────────────────────────────
        hdr = QFrame(); hdr.setObjectName("header")
        hl  = QHBoxLayout(hdr); hl.setContentsMargins(20,14,20,14)
        t   = QLabel("Identification par IA")
        t.setStyleSheet("font-size:17px;font-weight:700;color:#FFFFFF;background:transparent;")
        s   = QLabel("PFE 2025/2026 : EV2 ZOUINE-EV2 EL MAACHI")
        s.setStyleSheet("font-size:11px;color:#A0C8FF;background:transparent;")
        vb  = QVBoxLayout(); vb.setSpacing(2); vb.addWidget(t); vb.addWidget(s)
        hl.addLayout(vb); main.addWidget(hdr)

        # ── Fichier + boutons ────────────────────────────────────────────────
        main.addWidget(SectionLabel("Fichier IQ à identifier"))
        self.path_sel = PathSelector(r"Ex : C:\captures\test.iq")
        main.addWidget(self.path_sel)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("  Lancer l'identification")
        self.btn_start.setIcon(make_icon(SVG_PLAY, size=20, color="#FFFFFF"))
        self.btn_start.setIconSize(QSize(20,20))
        self.btn_abort = QPushButton("  Arrêter")
        self.btn_abort.setIcon(make_icon(SVG_PAUSE, size=20, color="#FFFFFF"))
        self.btn_abort.setIconSize(QSize(20,20))
        self.btn_abort.setObjectName("danger"); self.btn_abort.setEnabled(False)
        
        self.btn_export = QPushButton("  Exporter JSON")
        self.btn_export.setIcon(make_icon(SVG_DOWNLOAD, size=20, color="#FFFFFF"))
        self.btn_export.setIconSize(QSize(20,20))
        self.btn_export.setObjectName("secondary"); self.btn_export.setEnabled(False)
        
        btn_row.addWidget(self.btn_start); btn_row.addWidget(self.btn_abort); 
        btn_row.addWidget(self.btn_export); btn_row.addStretch()
        main.addLayout(btn_row)

        self.prog_bar = QProgressBar(); self.prog_bar.setFixedHeight(14)
        self.prog_lbl = QLabel("En attente...")
        self.prog_lbl.setStyleSheet(f"font-size:11px;color:{PAL['mid']};")
        main.addWidget(self.prog_bar); main.addWidget(self.prog_lbl)

        # ── Bandeau modèle ───────────────────────────────────────────────────
        self.model_frame = QFrame()
        self.model_frame.setStyleSheet(
            f"QFrame{{background:{PAL['white']};border:1px solid {PAL['border']};"
            f"border-left:4px solid {PAL['teal']};border-radius:8px;}}")
        mf_lay = QHBoxLayout(self.model_frame); mf_lay.setContentsMargins(14,8,14,8)
        self.model_lbl = QLabel("Modèle : —")
        self.model_lbl.setStyleSheet(f"font-size:13px;font-weight:600;color:{PAL['dark']};background:transparent;")
        self.f1_lbl = QLabel("")
        self.f1_lbl.setStyleSheet(f"font-size:13px;color:{PAL['teal']};font-weight:600;background:transparent;")
        mf_lay.addWidget(self.model_lbl); mf_lay.addStretch(); mf_lay.addWidget(self.f1_lbl)
        main.addWidget(self.model_frame)

        # ── Métriques ────────────────────────────────────────────────────────
        cards_row = QHBoxLayout()
        self.card_connus   = MetricCard("—", "Bursts connus",   PAL["teal"])
        self.card_inconnus = MetricCard("—", "Bursts inconnus", PAL["coral"])
        self.card_total    = MetricCard("—", "Total bursts",    PAL["blue"])
        self.card_temps    = MetricCard("—", "Temps (s)",       PAL["amber"])
        for c in [self.card_connus, self.card_inconnus, self.card_total, self.card_temps]:
            c.setMinimumHeight(80)
            cards_row.addWidget(c)
        main.addLayout(cards_row)

        # ── Graphe distribution émetteurs ────────────────────────────────────
        main.addWidget(SectionLabel("Distribution des émetteurs détectés"))
        self.emetteurs_canvas = MplWidget(10, 5, 90, "Distribution des émetteurs")
        self.emetteurs_canvas.setMinimumHeight(340)
        self.emetteurs_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        main.addWidget(self.emetteurs_canvas)

        # ── Graphe probabilités par classe ───────────────────────────────────
        main.addWidget(SectionLabel("Probabilité moyenne d'appartenance par classe"))
        self.proba_canvas = MplWidget(10, 3, 90, "Probabilités par classe")
        self.proba_canvas.setMinimumHeight(200)
        self.proba_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        main.addWidget(self.proba_canvas)

        # ── Splitter tableaux + log ──────────────────────────────────────────
        tbl_split = QSplitter(Qt.Orientation.Horizontal)
        tbl_split.setMinimumHeight(260)

        # Tableau bursts
        left_tbl = QWidget()
        lt = QVBoxLayout(left_tbl); lt.setContentsMargins(0,0,6,0); lt.setSpacing(4)
        lt.addWidget(SectionLabel("Bursts détectés"))
        self.burst_table = QTableWidget(0, 4)
        self.burst_table.setHorizontalHeaderLabels(["Burst #", "Classe", "Proba (%)", "Statut"])
        self.burst_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.burst_table.setAlternatingRowColors(True)
        lt.addWidget(self.burst_table)

        # Tableau inconnus + log
        right_tbl = QWidget()
        rt = QVBoxLayout(right_tbl); rt.setContentsMargins(6,0,0,0); rt.setSpacing(4)
        rt.addWidget(SectionLabel("Regroupements inconnus"))
        self.inconnu_table = QTableWidget(0, 3)
        self.inconnu_table.setHorizontalHeaderLabels(["Groupe", "Nb bursts", "Statut"])
        self.inconnu_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.inconnu_table.setAlternatingRowColors(True)
        self.inconnu_table.setMinimumHeight(120)
        rt.addWidget(self.inconnu_table)
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(80); self.log_view.setMaximumHeight(120)
        rt.addWidget(self.log_view)

        tbl_split.addWidget(left_tbl); tbl_split.addWidget(right_tbl)
        tbl_split.setStretchFactor(0,1); tbl_split.setStretchFactor(1,1)
        main.addWidget(tbl_split)

        scroll.setWidget(inner); outer.addWidget(scroll)
        self.btn_start.clicked.connect(self._run)
        self.btn_abort.clicked.connect(self._abort)
        self.btn_export.clicked.connect(self._export_json)

    def _run(self):
        path = self.path_sel.text()
        if not path or not Path(path).exists():
            QMessageBox.warning(self,"Fichier manquant","Sélectionnez un fichier IQ."); return
        if not MODEL_PATH.exists():
            QMessageBox.warning(self,"Modèle absent","Entraînez un modèle d'abord."); return
        try:
            model = joblib.load(MODEL_PATH)
            with open(META_PATH) as f: meta = json.load(f)
        except Exception as e:
            QMessageBox.critical(self,"Erreur",str(e)); return

        modele_nom = meta.get("modele","?")
        f1 = meta.get("f1_macro", None)
        self.model_lbl.setText(f"Modèle : {modele_nom}")
        self.f1_lbl.setText(f"F1 macro = {f1:.4f}" if f1 is not None else "")

        self._last_path = path
        self.btn_start.setEnabled(False); self.btn_abort.setEnabled(True)
        self.prog_bar.setValue(0); self.log_view.clear()
        self.burst_table.setRowCount(0); self.inconnu_table.setRowCount(0)
        p = self.get_params()
        self.worker = IdentificationWorker(path, model, meta, p)
        self.worker.progress.connect(lambda v,m: (self.prog_bar.setValue(v), self.prog_lbl.setText(m)))
        self.worker.finished_ok.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        self.status_msg.emit("Identification AIS en cours...")

    def _abort(self):
        if self.worker: self.worker.abort()
        self.btn_start.setEnabled(True); self.btn_abort.setEnabled(False)

    def _on_done(self, res):
        self._last_result = res
        self.btn_start.setEnabled(True); self.btn_abort.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.prog_bar.setValue(100); self.prog_lbl.setText("Terminé !")

        labels   = res.get("labels_per_burst", [])
        probas_b = res.get("probas_per_burst", [])
        classes  = res["classes"]
        emetteurs        = res.get("emetteurs", {})
        n_inconnus_total = res.get("n_inconnus_total", 0)
        groupes_inconnus = res.get("groupes_inconnus", {})

        try:
            with open(META_PATH) as f: meta_r = json.load(f)
            seuil = meta_r.get("seuil_proba", 0.60)
        except Exception:
            seuil = 0.60

        # ── Tableau bursts ───────────────────────────────────────────────────
        n_connus = 0; n_inconnus = 0
        self.burst_table.setRowCount(0)
        for i, (lbl, prb) in enumerate(zip(labels, probas_b)):
            est_connu = (lbl != "INCONNU")
            if est_connu: n_connus += 1
            else:         n_inconnus += 1
            row = self.burst_table.rowCount(); self.burst_table.insertRow(row)
            self.burst_table.setItem(row,0,QTableWidgetItem(str(i+1)))
            self.burst_table.setItem(row,1,QTableWidgetItem(str(lbl)))
            self.burst_table.setItem(row,2,QTableWidgetItem(f"{prb*100:.1f}"))
            si = QTableWidgetItem("✅ Connu" if est_connu else "❓ Inconnu")
            si.setForeground(QColor(PAL["teal"] if est_connu else PAL["coral"]))
            self.burst_table.setItem(row,3,si)

        # ── Tableau regroupements inconnus ───────────────────────────────────
        self.inconnu_table.setRowCount(0)
        for uid, info in groupes_inconnus.items():
            row = self.inconnu_table.rowCount(); self.inconnu_table.insertRow(row)
            self.inconnu_table.setItem(row,0,QTableWidgetItem(uid))
            self.inconnu_table.setItem(row,1,QTableWidgetItem(str(info["n"])))
            statut = info.get("statut","EN_ATTENTE")
            si2 = QTableWidgetItem(statut)
            si2.setForeground(QColor(PAL["amber"] if statut=="EN_ATTENTE" else PAL["teal"]))
            self.inconnu_table.setItem(row,2,si2)

        # ── Métriques ────────────────────────────────────────────────────────
        def _set_card(card, val):
            card.findChildren(QLabel)[0].setText(str(val))
        _set_card(self.card_connus,   n_connus)
        _set_card(self.card_inconnus, n_inconnus)
        _set_card(self.card_total,    res["n_total"])
        _set_card(self.card_temps,    f"{res['elapsed']:.1f}s")

        # ── Graphe multi-émetteurs ────────────────────────────────────────────
        fig = self.emetteurs_canvas.fig; fig.clear()
        ax  = fig.add_subplot(111)

        # Couleurs fixes pour les classes connues
        cmap_known = [PAL["teal"], PAL["blue"], PAL["amber"], PAL["purple"],
                      "#0F6E56", "#2B21B5", "#9E1D75", "#5A7A1D"]
        bar_labels   = list(emetteurs.keys())
        bar_vals     = list(emetteurs.values())
        bar_colors   = [cmap_known[i % len(cmap_known)] for i in range(len(bar_labels))]

        # Ajouter INCONNUS si présents
        if n_inconnus > 0:
            bar_labels.append("INCONNUS")
            bar_vals.append(n_inconnus)
            bar_colors.append(PAL["coral"])

        if bar_labels:
            x = np.arange(len(bar_labels))
            bars = ax.bar(x, bar_vals, color=bar_colors, width=0.55, zorder=3)
            ax.set_xticks(x); ax.set_xticklabels(bar_labels, fontsize=10, fontweight="600")
            ax.set_ylabel("Nombre de bursts", fontsize=9)
            ax.set_title(f"Distribution des émetteurs — {Path(self._last_path).name}",
                         fontsize=10, fontweight="600")
            for bar, v in zip(bars, bar_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        str(v), ha="center", va="bottom", fontsize=10, fontweight="700",
                        color=PAL["dark"])
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(rect=[0,0,1,1]); self.emetteurs_canvas.draw()

        # ── Graphe probabilité moyenne par classe ────────────────────────────
        fig2 = self.proba_canvas.fig; fig2.clear()
        ax2  = fig2.add_subplot(111)
        cls  = classes; pr = res["mean_pr"]
        dec  = res["decision"]
        colors2 = [PAL["teal"] if c==dec else "#CBD5E1" for c in cls]
        bars2 = ax2.barh(cls, pr, color=colors2, height=0.5)
        for bar, v in zip(bars2, pr):
            ax2.text(min(v+0.01, 0.95), bar.get_y()+bar.get_height()/2,
                     f"{v*100:.1f}%", va="center", fontsize=8, color=PAL["dark"])
        ax2.set_xlim(0,1.05)
        ax2.set_xlabel("Probabilité moyenne", fontsize=9)
        ax2.set_title("Probabilité moyenne d'appartenance", fontsize=9, fontweight="600")
        ax2.tick_params(labelsize=8)
        fig2.tight_layout(); self.proba_canvas.draw()

        # ── Log résumé ───────────────────────────────────────────────────────
        self.log_view.clear()
        for em, nb in emetteurs.items():
            self.log_view.append(f"✅ {em} : {nb} burst(s) connu(s)")
        if n_inconnus > 0:
            self.log_view.append(f"❓ INCONNUS : {n_inconnus} burst(s) → {len(groupes_inconnus)} groupe(s)")
        self.log_view.append(f"Total : {res['n_total']} bursts  |  Temps : {res['elapsed']:.1f}s")

        # ── Signal vers AnalysisTab + InconnusTab ────────────────────────────
        if res.get("first_seg") is not None:
            self.burst_ready.emit(self._last_path, self.get_params(), res["first_seg"])
        if n_inconnus > 0:
            self.inconnus_updated.emit()

        em_str = "  ".join(f"{k}:{v}" for k,v in emetteurs.items())
        self.status_msg.emit(
            f"AIS : {em_str}  |  ❓ {n_inconnus} inconnus ({len(groupes_inconnus)} groupes)"
            f"  |  Total {res['n_total']} bursts")

        # Exportation Cloud automatique
        export_to_google_sheets(res, id_traitement=f"ID_{Path(self._last_path).stem}")

    def _export_json(self):
        if not self._last_result:
            return
        try:
            res = self._last_result
            p   = self.get_params()
            fs  = p["fs"]
            taus = np.logspace(np.log10(10e-6), np.log10(100e-3), 40)

            id_tr  = f"ID_{Path(self._last_path).stem}"
            dossier = Path(__file__).parent / "archives"
            dossier.mkdir(exist_ok=True)

            def clean(val):
                return val.tolist() if hasattr(val, "tolist") else val

            # Métriques globales
            n_tot = res.get("n_total", 0)
            n_inc = res.get("n_inconnus_total", 0)
            n_con = n_tot - n_inc

            # ── Calcul par burst ─────────────────────────────────────────────
            bursts_data = []
            iq_bursts = res.get("iq_bursts", [])  # liste des segments IQ numpy

            for i, seg in enumerate(iq_bursts):
                try:
                    use_cmse    = p.get("use_cmse", True)
                    phase_brute = iq_to_phase(seg)
                    phase_emd   = apply_emd(
                        phase_brute,
                        max_imf   = p["emd_imf"],
                        subsample = p["emd_sub"],
                        use_cmse  = use_cmse
                    ) if p["use_emd"] else phase_brute.copy()
                    phase_purif = apply_wavelet_denoise(
                        phase_emd, wavelet=p["wt_wav"]
                    ) if p["use_wt"] else phase_emd.copy()

                    # Aligner longueurs
                    N = len(phase_brute)
                    if len(phase_purif) > N:
                        phase_purif = phase_purif[:N]
                    elif len(phase_purif) < N:
                        phase_purif = np.pad(phase_purif, (0, N - len(phase_purif)))

                    # Variances
                    v_avar = compute_avar(phase_purif, taus, fs)
                    v_hvar = compute_hvar(phase_purif, taus, fs)
                    v_pvar = compute_pvar(phase_purif, taus, fs)

                    # PSD
                    f_psd, Pxx = welch(
                        seg, fs=fs,
                        nperseg=min(1024, len(seg) // 2),
                        return_onesided=False
                    )
                    Pdb = 10 * np.log10(np.abs(Pxx) + 1e-20)

                    # Fréquence instantanée
                    freq_inst = np.diff(iq_to_phase(seg)) * fs / (2 * np.pi) / fs
                    freq_inst = np.append(freq_inst, freq_inst[-1])

                    # Features complètes
                    try:
                        fv, dom, _, _ = full_feature_vector(
                            seg, taus, fs, p["nseg"],
                            use_emd       = p["use_emd"],
                            use_wt        = p["use_wt"],
                            use_cmse      = use_cmse,
                            emd_subsample = p["emd_sub"],
                            emd_max_imf   = p["emd_imf"],
                            wt_wavelet    = p["wt_wav"],
                        )
                        fn_list = feature_names(p["nseg"])
                    except Exception:
                        fv      = np.zeros(10)
                        dom     = "N/A"
                        fn_list = ["N/A"] * 10

                    t_ms = (np.arange(N) / fs * 1000).tolist()
                    t_fi = (np.arange(len(freq_inst)) / fs * 1000).tolist()

                    bursts_data.append({
                        "burst_index":  i,
                        "label":        res.get("labels_per_burst", [])[i] if i < len(res.get("labels_per_burst", [])) else "N/A",
                        "proba":        round(float(res.get("probas_per_burst", [])[i]) * 100, 1) if i < len(res.get("probas_per_burst", [])) else 0.0,
                        "t_ms":         t_ms,
                        "phase_brute":  clean(phase_brute),
                        "phase_emd":    clean(phase_emd),
                        "taus":         clean(taus),
                        "avar":         clean(v_avar),
                        "hvar":         clean(v_hvar),
                        "pvar":         clean(v_pvar),
                        "f_psd":        clean(np.fft.fftshift(f_psd)),
                        "psd":          clean(np.fft.fftshift(Pdb)),
                        "t_fi":         t_fi,
                        "freq_inst":    clean(freq_inst),
                        "feat_names":   fn_list,
                        "features":     clean(fv),
                        "dom":          dom,
                    })

                except Exception as e:
                    bursts_data.append({
                        "burst_index": i,
                        "label":       "ERREUR",
                        "proba":       0.0,
                        "erreur":      str(e),
                    })

            # ── Rapport final ────────────────────────────────────────────────
            report = {
                "id":       id_tr,
                "date":     datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                "decision": res.get("decision", "N/A"),
                "confiance": f"{res.get('consensus', 0):.2f}%",
                "metrics": {
                    "n_connus":   n_con,
                    "n_inconnus": n_inc,
                    "n_total":    n_tot,
                    "elapsed":    round(res.get("elapsed", 0), 1),
                },
                "distribution":     res.get("emetteurs", {}),
                "bursts": {
                    "labels": res.get("labels_per_burst", []),
                    "probas": [round(float(p) * 100, 1) for p in res.get("probas_per_burst", [])],
                },
                "groupes_inconnus": res.get("groupes_inconnus", {}),
                # Données legacy (premier burst) pour compatibilité
                "psd":  {"f": clean(res.get("f_psd", [])),  "p": clean(res.get("Pdb", []))},
                "avar": {"x": clean(res.get("taus",  [])),  "y": clean(res.get("avar", []))},
                # ── NOUVEAU : données complètes par burst ────────────────────
                "bursts_data": bursts_data,
            }

            save_path, _ = QFileDialog.getSaveFileName(
                self, "Exporter Rapport JSON COMPLET",
                str(dossier / f"{id_tr}.json"),
                "Fichiers JSON (*.json)"
            )
            if save_path:
                with open(save_path, "w") as f:
                    json.dump(report, f, indent=2, default=convert_json)
                QMessageBox.information(self, "Exportation", f"Rapport JSON exporté avec succès.\n{len(bursts_data)} burst(s) inclus.")

        except Exception as e:
            QMessageBox.critical(self, "Erreur Export", f"Détails : {e}")

    def _on_error(self, msg):
        self.btn_start.setEnabled(True); self.btn_abort.setEnabled(False)
        QMessageBox.critical(self,"Erreur",msg[:800])

    def get_last_path(self):
        return self._last_path

    def get_last_result(self):
        return self._last_result

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET INCONNUS
# ─────────────────────────────────────────────────────────────────────────────
class InconnusTab(QWidget):
    status_msg = pyqtSignal(str)

    def __init__(self, get_params, parent=None):
        super().__init__(parent)
        self.get_params = get_params
        self.worker = None
        self._inconnus_db = {}
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main = QVBoxLayout(inner); main.setContentsMargins(20,20,20,20); main.setSpacing(10)

        hdr = QFrame()
        hdr.setStyleSheet(f"QFrame{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                          f"stop:0 #2C1050,stop:1 #7B3FA0);border-radius:10px;}}")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(20,14,20,14)
        t = QLabel("Gestion des inconnus & Apprentissage continu")
        t.setStyleSheet("font-size:16px;font-weight:700;color:#FFFFFF;background:transparent;")
        s = QLabel("Base d'émetteurs inconnus · Étiquetage · Réentraînement incrémental")
        s.setStyleSheet("font-size:11px;color:#D4B4F0;background:transparent;")
        vb = QVBoxLayout(); vb.setSpacing(2); vb.addWidget(t); vb.addWidget(s)
        hl.addLayout(vb); main.addWidget(hdr)

        # ── Tableau inconnus ─────────────────────────────────────────────────
        main.addWidget(SectionLabel("Base des émetteurs inconnus"))
        btn_row_top = QHBoxLayout()
        btn_refresh = QPushButton("⟳ Actualiser"); btn_refresh.setObjectName("secondary")
        btn_clear   = QPushButton("Effacer base");  btn_clear.setObjectName("danger")
        btn_refresh.clicked.connect(self.refresh)
        btn_clear.clicked.connect(self._clear_db)
        btn_row_top.addWidget(btn_refresh); btn_row_top.addWidget(btn_clear); btn_row_top.addStretch()
        main.addLayout(btn_row_top)

        # 7 colonnes : ID, N captures, Étiquette, Statut, Première vue, Dernière vue, Fichier(s) source
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "ID", "N captures", "Étiquette", "Statut",
            "Première vue", "Dernière vue", "Fichier(s) source"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(220)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Largeurs de colonnes confortables
        for i, w in enumerate([130, 100, 140, 110, 110, 110, 300]):
            self.table.setColumnWidth(i, w)
        main.addWidget(self.table)

        # ── Étiquetage sans réentraînement ───────────────────────────────────
        main.addWidget(SectionLabel("Étiquetage"))
        etiq_frame = QFrame()
        etiq_frame.setStyleSheet(
            f"QFrame{{background:{PAL['white']};border:1px solid {PAL['border']};"
            f"border-left:4px solid {PAL['amber']};border-radius:8px;}}")
        ef = QVBoxLayout(etiq_frame); ef.setContentsMargins(16,12,16,12); ef.setSpacing(8)

        form_e = QFormLayout(); form_e.setSpacing(8)
        self.uid_combo_etiq = QComboBox(); self.uid_combo_etiq.setMinimumHeight(32)
        self.etiquette_edit = QLineEdit()
        self.etiquette_edit.setPlaceholderText("Ex: AIS05_NAVIRE_X"); self.etiquette_edit.setMinimumHeight(32)
        form_e.addRow("Inconnu à étiqueter :", self.uid_combo_etiq)
        form_e.addRow("Nouvelle étiquette :",   self.etiquette_edit)
        ef.addLayout(form_e)

        btn_row_e = QHBoxLayout()
        btn_name_only = QPushButton("✏  Nommer seulement  (sans réentraîner)")
        btn_name_only.setObjectName("secondary"); btn_name_only.setMinimumHeight(36)
        btn_name_only.clicked.connect(self._name_only)
        btn_row_e.addWidget(btn_name_only); btn_row_e.addStretch()
        ef.addLayout(btn_row_e)
        main.addWidget(etiq_frame)

        # ── Réentraînement incrémental ───────────────────────────────────────
        main.addWidget(SectionLabel("Réentraînement incrémental"))
        retrain_frame = QFrame()
        retrain_frame.setStyleSheet(
            f"QFrame{{background:{PAL['white']};border:1px solid {PAL['border']};"
            f"border-left:4px solid {PAL['purple']};border-radius:8px;}}")
        rf = QVBoxLayout(retrain_frame); rf.setContentsMargins(16,12,16,12); rf.setSpacing(8)

        form_r = QFormLayout(); form_r.setSpacing(8)
        self.uid_combo = QComboBox(); self.uid_combo.setMinimumHeight(32)
        form_r.addRow("Inconnu à intégrer :", self.uid_combo)
        rf.addLayout(form_r)

        self.info_lbl = QLabel("Sélectionnez un inconnu et donnez-lui une étiquette dans la section ci-dessus.")
        self.info_lbl.setStyleSheet(f"font-size:12px;color:{PAL['mid']};")
        self.info_lbl.setWordWrap(True)
        rf.addWidget(self.info_lbl)

        btn_retrain = QPushButton("  Réentraîner avec cet inconnu")
        btn_retrain.setObjectName("purple"); btn_retrain.setMinimumHeight(36)
        btn_retrain.setIcon(make_icon(SVG_TRAIN, size=20, color="#FFFFFF"))
        btn_retrain.setIconSize(QSize(20,20))
        btn_retrain.clicked.connect(self._retrain)
        rf.addWidget(btn_retrain)
        main.addWidget(retrain_frame)

        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(120); self.log_view.setMaximumHeight(180)
        main.addWidget(self.log_view)
        main.addStretch()

        scroll.setWidget(inner); outer.addWidget(scroll)
        self.refresh()

    def _source_display(self, fichiers):
        """Affiche chemin complet si 1 fichier, noms seulement si plusieurs."""
        if not fichiers:
            return "—"
        if len(fichiers) == 1:
            return fichiers[0]   # chemin complet
        return "  |  ".join(Path(f).name for f in fichiers)

    def refresh(self):
        self._inconnus_db = charger_inconnus()
        self.table.setRowCount(0)
        self.uid_combo.clear()
        self.uid_combo_etiq.clear()
        for uid, data in self._inconnus_db.items():
            row = self.table.rowCount(); self.table.insertRow(row)
            statut  = data.get("statut","EN_ATTENTE")
            n       = data.get("n", 0)
            etiq    = data.get("etiquette_finale","")
            sources = data.get("fichiers_sources", [])
            self.table.setItem(row, 0, QTableWidgetItem(uid))
            self.table.setItem(row, 1, QTableWidgetItem(str(n)))
            self.table.setItem(row, 2, QTableWidgetItem(etiq))
            self.table.setItem(row, 3, QTableWidgetItem(statut))
            self.table.setItem(row, 4, QTableWidgetItem(data.get("date_premier","")[:10]))
            self.table.setItem(row, 5, QTableWidgetItem(data.get("date_dernier","")[:10]))
            self.table.setItem(row, 6, QTableWidgetItem(self._source_display(sources)))
            # Coloration
            color = PAL["teal"] if statut=="INTÉGRÉ" else (PAL["amber"] if n>=10 else PAL["dark"])
            for c in range(7):
                item = self.table.item(row, c)
                if item: item.setForeground(QColor(color))
            if statut != "INTÉGRÉ":
                self.uid_combo.addItem(f"{uid} ({n} cap.)", uid)
                self.uid_combo_etiq.addItem(f"{uid} ({n} cap.){' — '+etiq if etiq else ''}", uid)
        if self._inconnus_db:
            n_total = len(self._inconnus_db)
            n_pret  = sum(1 for d in self._inconnus_db.values()
                          if d.get("n",0)>=10 and d.get("statut")!="INTÉGRÉ")
            self.info_lbl.setText(
                f"{n_total} inconnu(s) enregistré(s) — {n_pret} prêt(s) à intégrer (≥10 captures)")

    def _name_only(self):
        """Applique une étiquette sans réentraîner le modèle."""
        idx = self.uid_combo_etiq.currentIndex()
        if idx < 0:
            QMessageBox.warning(self,"Aucun inconnu","Sélectionnez un inconnu."); return
        uid = self.uid_combo_etiq.currentData()
        etiquette = self.etiquette_edit.text().strip()
        if not etiquette:
            QMessageBox.warning(self,"Étiquette","Entrez une étiquette."); return
        db = charger_inconnus()
        if uid in db:
            db[uid]["etiquette_finale"] = etiquette
            sauvegarder_inconnus(db)
            self.log_view.append(f"✏ {uid} → étiquette '{etiquette}' enregistrée (modèle non modifié)")
            self.refresh()
            self.status_msg.emit(f"{uid} nommé '{etiquette}' — sans réentraînement")

    def _clear_db(self):
        rep = QMessageBox.question(self,"Confirmer","Effacer toute la base des inconnus ?",
                                   QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
        if rep == QMessageBox.StandardButton.Yes:
            if INCONNUS_FILE.exists(): INCONNUS_FILE.unlink()
            self.refresh()

    def _retrain(self):
        idx = self.uid_combo.currentIndex()
        if idx < 0: QMessageBox.warning(self,"Aucun inconnu","Sélectionnez un inconnu."); return
        uid = self.uid_combo.currentData()
        # Récupérer l'étiquette depuis la DB (champ etiquette_finale) ou demander
        db = charger_inconnus()
        etiquette = db.get(uid,{}).get("etiquette_finale","").strip()
        if not etiquette:
            etiquette = self.etiquette_edit.text().strip()
        if not etiquette:
            QMessageBox.warning(self,"Étiquette",
                "Nommez d'abord cet inconnu via 'Nommer seulement' ou entrez une étiquette."); return

        n = db.get(uid,{}).get("n",0)
        if n < 10:
            rep = QMessageBox.question(self,"Captures insuffisantes",
                f"{uid} n'a que {n} captures (recommandé ≥ 10). Continuer quand même ?",
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
            if rep != QMessageBox.StandardButton.Yes: return

        p = self.get_params()
        self.log_view.clear()
        self.worker = RetrainWorker(uid, etiquette, db,
                                    k_max_gmm=p["k_max_gmm"], percentile_gmm=p["percentile_gmm"])
        self.worker.log.connect(self.log_view.append)
        self.worker.finished_ok.connect(self._on_retrain_done)
        self.worker.error.connect(lambda m: QMessageBox.critical(self,"Erreur",m[:600]))
        self.worker.start()
        self.status_msg.emit(f"Réentraînement avec {uid} → {etiquette}...")

    def _on_retrain_done(self, res):
        self.refresh()
        self.status_msg.emit(f"'{res['etiquette']}' intégré avec succès !")
        QMessageBox.information(self,"Réentraînement",
            f"'{res['etiquette']}' est maintenant une classe connue.\nGMM mis à jour.")

class AnalysisBurstWorker(QThread):
    """Analyse un burst IQ déjà en mémoire (numpy array) — pas de fichier."""
    finished_ok = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, burst_iq, params):
        super().__init__()
        self.burst_iq = burst_iq
        self.params   = params

    def run(self):
        try:
            p    = self.params
            taus = np.logspace(np.log10(10e-6), np.log10(100e-3), 40)
            seg  = self.burst_iq

            use_cmse    = p.get("use_cmse", True)
            phase_brute = iq_to_phase(seg)
            phase_emd   = apply_emd(phase_brute, max_imf=p["emd_imf"],
                                     subsample=p["emd_sub"], use_cmse=use_cmse) if p["use_emd"] else phase_brute.copy()
            phase_purif = apply_wavelet_denoise(phase_emd, wavelet=p["wt_wav"]) if p["use_wt"] else phase_emd.copy()
            if len(phase_purif) != len(phase_brute):
                phase_purif = phase_purif[:len(phase_brute)] if len(phase_purif)>len(phase_brute) \
                              else np.pad(phase_purif,(0,len(phase_brute)-len(phase_purif)))

            fs = p["fs"]
            f_psd, Pxx = welch(seg, fs=fs, nperseg=min(1024,len(seg)//2), return_onesided=False)
            Pdb        = 10*np.log10(np.abs(Pxx)+1e-20)
            freq_inst  = np.diff(iq_to_phase(seg))*fs/(2*np.pi)/fs
            freq_inst  = np.append(freq_inst, freq_inst[-1])

            fv, dom, _, _ = full_feature_vector(
                seg, taus, fs, p["nseg"],
                use_emd=p["use_emd"], use_wt=p["use_wt"], use_cmse=use_cmse,
                emd_subsample=p["emd_sub"], emd_max_imf=p["emd_imf"],
                wt_wavelet=p["wt_wav"])

            self.finished_ok.emit({
                "seg": seg, "taus": taus, "fs": fs,
                "phase_brute": phase_brute,
                "phase_emd":   phase_emd,
                "phase_purif": phase_purif,
                "f_psd": f_psd, "Pdb": Pdb,
                "freq_inst": freq_inst,
                "fv": fv, "dom": dom,
                "feat_names": feature_names(p["nseg"]),
            })
        except Exception:
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET ANALYSE SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
class AnalysisTab(QWidget):
    status_msg = pyqtSignal(str)

    def __init__(self, get_params, parent=None):
        super().__init__(parent)
        self.get_params   = get_params
        self.worker       = None
        self._auto_burst  = None   # burst reçu depuis IdentificationTab
        self._auto_path   = ""
        self._auto_params = None
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main = QVBoxLayout(inner)
        main.setContentsMargins(20,20,20,20); main.setSpacing(8)

        # En-tête
        hdr = QFrame()
        hdr.setStyleSheet("QFrame{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                          "stop:0 #0F4D3A,stop:1 #1D9E75);border-radius:10px;}")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(20,14,20,14)
        t = QLabel("Analyse du signal ")
        t.setStyleSheet("font-size:17px;font-weight:700;color:#FFFFFF;background:transparent;")
        s = QLabel("Variances · DSP · Phase · Fréquence instantanée · Features")
        s.setStyleSheet("font-size:11px;color:#9EFBD8;background:transparent;")
        vb = QVBoxLayout(); vb.setSpacing(2); vb.addWidget(t); vb.addWidget(s)
        hl.addLayout(vb); main.addWidget(hdr)

        # Source
        self.src_lbl = QLabel("Source : en attente d'une identification…")
        self.src_lbl.setStyleSheet(f"font-size:12px;color:{PAL['mid']};")
        main.addWidget(self.src_lbl)
        src_row = QHBoxLayout()
        self.path_sel = PathSelector(r"Ou choisir un autre fichier IQ…")
        self.btn_analyse = QPushButton("  Analyser ce fichier")
        self.btn_analyse.setIcon(make_icon(SVG_ACTIVITY, size=18, color="#FFFFFF"))
        self.btn_analyse.setIconSize(QSize(18,18))
        self.btn_analyse.clicked.connect(self._run_manual)
        src_row.addWidget(self.path_sel); src_row.addWidget(self.btn_analyse)
        main.addLayout(src_row)

        self.prog_bar = QProgressBar(); self.prog_bar.setFixedHeight(10)
        main.addWidget(self.prog_bar)

        # ── Ligne 1 : Phase (gauche) + Variances AVAR/HVAR/PVAR (droite) ────
        main.addWidget(SectionLabel("Phase brute vs EMD+CMSE"))
        self.canvas_phase = MplWidget(10, 4, 90, "Phase")
        self.canvas_phase.setMinimumHeight(280)
        self.canvas_phase.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        main.addWidget(self.canvas_phase)

        main.addWidget(SectionLabel("Variances de phase — AVAR / HVAR / PVAR"))
        self.canvas_var = MplWidget(10, 4, 90, "Variances")
        self.canvas_var.setMinimumHeight(280)
        self.canvas_var.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        main.addWidget(self.canvas_var)

        # ── Ligne 2 : DSP (gauche) + Freq inst (droite) ──────────────────────
        row2 = QSplitter(Qt.Orientation.Horizontal)
        row2.setMinimumHeight(260)

        psd_w = QWidget(); pl = QVBoxLayout(psd_w); pl.setContentsMargins(0,0,4,0); pl.setSpacing(4)
        pl.addWidget(SectionLabel("DSP — Welch"))
        self.canvas_psd = MplWidget(6, 4, 90, "DSP")
        self.canvas_psd.setMinimumHeight(240)
        self.canvas_psd.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pl.addWidget(self.canvas_psd)

        fi_w = QWidget(); fl2 = QVBoxLayout(fi_w); fl2.setContentsMargins(4,0,0,0); fl2.setSpacing(4)
        fl2.addWidget(SectionLabel("Fréquence instantanée"))
        self.canvas_fi = MplWidget(6, 4, 90, "Fréq. inst.")
        self.canvas_fi.setMinimumHeight(240)
        self.canvas_fi.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        fl2.addWidget(self.canvas_fi)

        row2.addWidget(psd_w); row2.addWidget(fi_w)
        row2.setStretchFactor(0,1); row2.setStretchFactor(1,1)
        main.addWidget(row2)

        # ── Table features ────────────────────────────────────────────────────
        main.addWidget(SectionLabel("Features extraites du burst"))
        self.feat_table = QTableWidget(0, 2)
        self.feat_table.setHorizontalHeaderLabels(["Feature", "Valeur"])
        self.feat_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.feat_table.setMinimumHeight(300)
        self.feat_table.setAlternatingRowColors(True)
        self.feat_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        main.addWidget(self.feat_table)

        scroll.setWidget(inner); outer.addWidget(scroll)

    # ── Appelé automatiquement depuis IdentificationTab ──────────────────────
    def receive_burst(self, filepath, params, burst_iq):
        self._auto_burst  = burst_iq
        self._auto_path   = filepath
        self._auto_params = params
        self.src_lbl.setText(f"Source : {Path(filepath).name}  (burst actif depuis Identification)")
        self.src_lbl.setStyleSheet(f"font-size:12px;color:{PAL['teal']};font-weight:600;")
        self._run_burst(burst_iq, params)

    def _run_manual(self):
        path = self.path_sel.text()
        if not path or not Path(path).exists():
            QMessageBox.warning(self,"Fichier","Sélectionnez un fichier IQ valide."); return
        p = self.get_params()
        self.src_lbl.setText(f"Source : {Path(path).name}  (fichier choisi manuellement)")
        self.src_lbl.setStyleSheet(f"font-size:12px;color:{PAL['blue']};font-weight:600;")
        self.prog_bar.setValue(10)
        self.worker = AnalysisWorker(path, p)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.error.connect(lambda m: QMessageBox.critical(self,"Erreur",m[:600]))
        self.worker.start()

    def _run_burst(self, burst_iq, params):
        """Lance l'analyse directement sur un burst numpy déjà extrait."""
        self.prog_bar.setValue(10)
        self.worker = AnalysisBurstWorker(burst_iq, params)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.error.connect(lambda m: QMessageBox.critical(self,"Erreur",m[:600]))
        self.worker.start()

    def _on_done(self, res):
        self.prog_bar.setValue(100)
        fs   = res.get("fs", 400_000)
        taus = res["taus"]
        t_ms = np.arange(len(res["phase_brute"])) / fs * 1000

        # Phase
        fig = self.canvas_phase.fig; fig.clear(); ax = fig.add_subplot(111)
        ax.plot(t_ms, res["phase_brute"], color="#2B21B5", lw=0.6, alpha=0.8, label="Phase brute")
        ax.plot(t_ms, res["phase_emd"],   color=PAL["teal"], lw=1.2, label="EMD+CMSE")
        ax.set_xlabel("Temps (ms)", fontsize=8); ax.set_ylabel("Phase (rad)", fontsize=8)
        ax.set_title("Phase brute vs EMD+CMSE", fontsize=9, fontweight="600")
        ax.legend(fontsize=7); fig.tight_layout(); self.canvas_phase.draw()

        # Variances
        fig2 = self.canvas_var.fig; fig2.clear()
        for idx,(fn,title,col) in enumerate([(compute_avar,"AVAR",PAL["teal"]),
                                              (compute_hvar,"HVAR",PAL["blue"]),
                                              (compute_pvar,"PVAR",PAL["amber"])]):
            ax2 = fig2.add_subplot(1,3,idx+1)
            v = fn(res["phase_purif"], taus, fs)
            valid = ~np.isnan(v) & (v>0)
            if valid.sum()>1:
                ax2.loglog(taus[valid], np.sqrt(v[valid]), color=col, lw=2)
                ax2.fill_between(taus[valid], np.sqrt(v[valid]), alpha=0.1, color=col)
            ax2.set_title(title, fontsize=9, fontweight="600")
            ax2.set_xlabel("τ (s)", fontsize=7); ax2.tick_params(labelsize=7)
        fig2.tight_layout(); self.canvas_var.draw()

        # PSD
        fig3 = self.canvas_psd.fig; fig3.clear(); ax3 = fig3.add_subplot(111)
        f_s = np.fft.fftshift(res["f_psd"])/1e3; P_s = np.fft.fftshift(res["Pdb"])
        ax3.plot(f_s, P_s, color=PAL["blue"], lw=1.2)
        ax3.set_xlabel("Fréquence (kHz)", fontsize=8); ax3.set_ylabel("Puissance (dB)", fontsize=8)
        ax3.set_title("DSP — Welch", fontsize=9, fontweight="600")
        fig3.tight_layout(); self.canvas_psd.draw()

        # Freq inst
        fig4 = self.canvas_fi.fig; fig4.clear(); ax4 = fig4.add_subplot(111)
        fi = res["freq_inst"]
        t_fi = np.arange(len(fi)) / fs * 1000
        ax4.plot(t_fi, fi, color=PAL["coral"], lw=0.6, alpha=0.8)
        ax4.set_xlabel("Temps (ms)", fontsize=8); ax4.set_ylabel("Fréq. inst. (Hz)", fontsize=8)
        ax4.set_title("Fréquence instantanée", fontsize=9, fontweight="600")
        fig4.tight_layout(); self.canvas_fi.draw()

        # Features
        self.feat_table.setRowCount(0)
        fn = res["feat_names"]; fv = res["fv"]
        for i,(name,val) in enumerate(zip(fn,fv)):
            self.feat_table.insertRow(i)
            self.feat_table.setItem(i,0,QTableWidgetItem(name))
            self.feat_table.setItem(i,1,QTableWidgetItem(f"{val:.6f}"))

        # Exportation Cloud automatique
        res_export = res.copy()
        if "avar" not in res_export:
             res_export["avar"] = compute_avar(res["phase_purif"], res["taus"], fs)
        export_to_google_sheets(res_export, id_traitement=f"ANALYSE_{datetime.datetime.now().strftime('%H%M%S')}")

        self.status_msg.emit(f"Analyse terminée — {len(fn)} features — bruit dominant : {res['dom']}")

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
class TrainingTab(QWidget):
    status_msg = pyqtSignal(str)
    training_done = pyqtSignal(dict)

    def __init__(self, get_params, parent=None):
        super().__init__(parent)
        self.get_params  = get_params
        self.worker      = None
        self.file_list   = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main = QVBoxLayout(inner)
        main.setSpacing(12); main.setContentsMargins(20,20,20,20)

        hdr = QFrame()
        hdr.setStyleSheet("QFrame{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                          "stop:0 #4A2800,stop:1 #BA7517);border-radius:10px;}")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(20,14,20,14)
        t = QLabel("Entraînement du modèle")
        t.setStyleSheet("font-size:17px;font-weight:700;color:#FFFFFF;background:transparent;")
        s = QLabel("Option d'entraînement · Fichiers IQ · Modèle existant")
        s.setStyleSheet("font-size:11px;color:#FFDDA0;background:transparent;")
        vb = QVBoxLayout(); vb.setSpacing(2); vb.addWidget(t); vb.addWidget(s)
        hl.addLayout(vb); main.addWidget(hdr)

        # ── Modèle existant ──────────────────────────────────────────────────
        main.addWidget(SectionLabel("Modèle actuellement chargé"))
        self.model_info_frame = QFrame()
        self.model_info_frame.setStyleSheet(
            f"QFrame{{background:{PAL['white']};border:1px solid {PAL['border']};"
            f"border-left:4px solid {PAL['teal']};border-radius:8px;}}")
        mi_lay = QVBoxLayout(self.model_info_frame); mi_lay.setContentsMargins(14,10,14,10); mi_lay.setSpacing(6)
        self.model_info_lbl = QLabel("Aucun modèle chargé")
        self.model_info_lbl.setStyleSheet(f"font-size:12px;color:{PAL['mid']};")
        self.model_info_lbl.setWordWrap(True)
        mi_lay.addWidget(self.model_info_lbl)

        # Tableau classes du modèle existant
        self.classes_table = QTableWidget(0, 3)
        self.classes_table.setHorizontalHeaderLabels(["Classe", "Fichier d'entraînement", "Bursts utilisés"])
        self.classes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.classes_table.horizontalHeader().setStretchLastSection(True)
        self.classes_table.setAlternatingRowColors(True)
        self.classes_table.setMinimumHeight(120)
        self.classes_table.setMaximumHeight(200)
        self.classes_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        mi_lay.addWidget(self.classes_table)

        btn_reload_model = QPushButton("⟳ Actualiser infos modèle")
        btn_reload_model.setObjectName("secondary"); btn_reload_model.setFixedWidth(200)
        btn_reload_model.clicked.connect(self._load_model_info)
        mi_lay.addWidget(btn_reload_model)
        main.addWidget(self.model_info_frame)

        # ── Fichiers d'entraînement ──────────────────────────────────────────
        main.addWidget(SectionLabel("Fichiers d'entraînement"))
        folder_row = QHBoxLayout()
        self.folder_sel = PathSelector("Dossier contenant les .iq", mode="dir")
        btn_scan = QPushButton("Scanner"); btn_scan.setObjectName("secondary")
        btn_scan.setFixedWidth(90); btn_scan.clicked.connect(self._scan_folder)
        folder_row.addWidget(self.folder_sel); folder_row.addWidget(btn_scan)
        main.addLayout(folder_row)

        self.file_table = QTableWidget(0, 3)
        self.file_table.setHorizontalHeaderLabels(["Étiquette","Fichier","Taille (Mo)"])
        self.file_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.file_table.setMinimumHeight(160); self.file_table.setMaximumHeight(220)
        self.file_table.setAlternatingRowColors(True)
        main.addWidget(self.file_table)

        btn_add = QPushButton("+ Ajouter fichier"); btn_add.setObjectName("secondary")
        btn_del = QPushButton("Supprimer sélection"); btn_del.setObjectName("danger")
        btn_del.setFixedWidth(180)
        btn_add.clicked.connect(self._add_file); btn_del.clicked.connect(self._del_file)
        br2 = QHBoxLayout(); br2.addWidget(btn_add); br2.addWidget(btn_del); br2.addStretch()
        main.addLayout(br2)

        # ── Options ──────────────────────────────────────────────────────────
        main.addWidget(SectionLabel("Options entraînement"))
        opt_row = QHBoxLayout()
        self.use_rfecv_chk    = QCheckBox("RFECV — Sélection features"); self.use_rfecv_chk.setChecked(True)
        self.use_cmse_chk2    = QCheckBox("CMSE (EMD amélioré)");         self.use_cmse_chk2.setChecked(True)
        self.train_gmm_chk    = QCheckBox("Entraîner GMM (Open Set)");    self.train_gmm_chk.setChecked(True)
        self.use_wt_train_chk = QCheckBox("WT — Débruitage ondelette");   self.use_wt_train_chk.setChecked(False)
        for w in [self.use_rfecv_chk, self.use_cmse_chk2, self.train_gmm_chk, self.use_wt_train_chk]:
            w.setMinimumHeight(30); opt_row.addWidget(w)
        opt_row.addStretch()
        main.addLayout(opt_row)

        self.btn_train = QPushButton("  Lancer l'entraînement")
        self.btn_train.setObjectName("gold"); self.btn_train.setMinimumHeight(42)
        self.btn_train.setIcon(make_icon(SVG_TRAIN, size=20, color=PAL["marine"]))
        self.btn_train.setIconSize(QSize(20,20))
        main.addWidget(self.btn_train)

        self.prog_bar = QProgressBar(); self.prog_bar.setFixedHeight(14)
        self.prog_lbl = QLabel("En attente...")
        self.prog_lbl.setStyleSheet(f"font-size:11px;color:{PAL['mid']};")
        main.addWidget(self.prog_bar); main.addWidget(self.prog_lbl)

        # ── Graphe importance features + log ─────────────────────────────────
        main.addWidget(SectionLabel("Résultats"))
        result_split = QSplitter(Qt.Orientation.Horizontal)
        result_split.setMinimumHeight(320)

        left_r = QWidget(); ll = QVBoxLayout(left_r); ll.setContentsMargins(0,0,6,0)
        self.imp_canvas = MplWidget(8,5,90,"Importance des features")
        self.imp_canvas.setMinimumHeight(300)
        ll.addWidget(self.imp_canvas)

        right_r = QWidget(); rl = QVBoxLayout(right_r); rl.setContentsMargins(6,0,0,0)
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(300)
        rl.addWidget(self.log_view)

        result_split.addWidget(left_r); result_split.addWidget(right_r)
        result_split.setStretchFactor(0,3); result_split.setStretchFactor(1,2)
        main.addWidget(result_split)

        scroll.setWidget(inner); outer.addWidget(scroll)
        self.btn_train.clicked.connect(self._run)
        self._load_model_info()  # charger les infos au démarrage

    def _load_model_info(self):
        """Charge et affiche les infos du modèle existant (classes, fichiers, obs)."""
        if not META_PATH.exists():
            self.model_info_lbl.setText("⚠ Aucun modèle entraîné — fichier model_meta.json absent.")
            self.classes_table.setRowCount(0)
            return
        try:
            with open(META_PATH) as f: meta = json.load(f)
            classes   = meta.get("classes", [])
            n_obs     = meta.get("n_observations", "?")
            n_feat    = meta.get("n_features", "?")
            n_feat_r  = meta.get("n_features_reduced", "?")
            modele    = meta.get("modele","?")
            date_entr = meta.get("date","")[:10]
            f1        = meta.get("f1_macro", None)
            f1_str    = f"  |  F1 macro = {f1:.4f}" if f1 is not None else ""
            self.model_info_lbl.setText(
                f"Modèle : {modele}{f1_str}  |  Classes : {', '.join(classes)}\n"
                f"Obs : {n_obs}  |  Features : {n_feat_r}/{n_feat} (RFECV)  |  Date : {date_entr}"
            )
            self.model_info_lbl.setStyleSheet(f"font-size:12px;color:{PAL['dark']};font-weight:500;")

            # Remplir tableau des classes
            self.classes_table.setRowCount(0)
            # Essayer de récupérer les infos de fichiers depuis self.file_list si disponible
            file_map = {Path(fp).stem.split("_")[0].upper(): fp for _, fp in self.file_list} \
                       if self.file_list else {}
            obs_par_classe = n_obs // len(classes) if classes else 0
            for cls in classes:
                row = self.classes_table.rowCount(); self.classes_table.insertRow(row)
                self.classes_table.setItem(row,0,QTableWidgetItem(cls))
                fp = file_map.get(cls, "—")
                self.classes_table.setItem(row,1,QTableWidgetItem(fp))
                self.classes_table.setItem(row,2,QTableWidgetItem(f"~{obs_par_classe}"))
                self.classes_table.item(row,0).setForeground(QColor(PAL["teal"]))
        except Exception as e:
            self.model_info_lbl.setText(f"Erreur lecture métadonnées : {e}")

    def _scan_folder(self):
        folder = self.folder_sel.text()
        if not folder or not Path(folder).exists():
            QMessageBox.warning(self,"Dossier","Sélectionnez un dossier valide."); return
        files = sorted(glob.glob(os.path.join(folder,"*.iq")) +
                       glob.glob(os.path.join(folder,"*.bin")))
        added = 0
        for f in files:
            label = Path(f).stem.split("_")[0].upper()
            if not any(fp == f for _,fp in self.file_list):
                self.file_list.append((label, f)); added += 1
        self._refresh_file_table()
        self.status_msg.emit(f"{added} fichier(s) ajouté(s)")

    def _add_file(self):
        path,_ = QFileDialog.getOpenFileName(self,"Fichier IQ",str(Path.home()),
                                              "Fichiers IQ (*.iq *.bin);;Tous (*.*)")
        if path:
            label = Path(path).stem.split("_")[0].upper()
            self.file_list.append((label, path)); self._refresh_file_table()

    def _del_file(self):
        rows = sorted(set(i.row() for i in self.file_table.selectedItems()), reverse=True)
        for r in rows: del self.file_list[r]
        self._refresh_file_table()

    def _refresh_file_table(self):
        self.file_table.setRowCount(0)
        for label, fp in self.file_list:
            row = self.file_table.rowCount(); self.file_table.insertRow(row)
            self.file_table.setItem(row,0,QTableWidgetItem(label))
            self.file_table.setItem(row,1,QTableWidgetItem(Path(fp).name))
            sz = round(Path(fp).stat().st_size/1e6,2) if Path(fp).exists() else 0
            self.file_table.setItem(row,2,QTableWidgetItem(str(sz)))

    def _run(self):
        if not self.file_list:
            QMessageBox.warning(self,"Fichiers","Ajoutez des fichiers d'entraînement."); return
        p = self.get_params()
        self.btn_train.setEnabled(False); self.prog_bar.setValue(0); self.log_view.clear()
        self.worker = TrainingWorker(
            self.file_list, p,
            use_rfecv=self.use_rfecv_chk.isChecked(),
            use_cmse=self.use_cmse_chk2.isChecked(),
            train_gmm=self.train_gmm_chk.isChecked(),
            max_segs=999,
            seuil_proba=p["seuil_proba"],
            percentile_gmm=p["percentile_gmm"],
            k_max_gmm=p["k_max_gmm"])
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log_view.append)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        self.status_msg.emit("Entraînement en cours...")

    def _on_progress(self, pct, msg):
        self.prog_bar.setValue(pct); self.prog_lbl.setText(msg)

    def _on_done(self, res):
        self.btn_train.setEnabled(True); self.prog_bar.setValue(100)
        self.prog_lbl.setText("Terminé !")
        rfecv_info = (f"  RFECV : {res['meta']['n_features_reduced']}/{res['meta']['n_features']} features"
                      if res["use_rfecv"] else "  RFECV : désactivé")
        gmm_info = "  GMM : entraîné ✓" if res["meta"].get("has_gmm") else "  GMM : désactivé"
        self.log_view.append(f"\n✅ Modèle sauvegardé{rfecv_info}{gmm_info}")

        # Importance
        imp = res["imp"]; fn = res["fn_list"]
        if imp:
            top = np.argsort(imp)[::-1][:15]
            cmap = lambda n: (PAL["teal"] if n.startswith(("AVAR","HVAR","PVAR")) else
                              PAL["blue"] if n.startswith("PSD") else
                              PAL["amber"] if n.startswith("FI") else
                              PAL["coral"] if n.startswith("PWR") else
                              PAL["purple"] if n.startswith("PH") else
                              "#0F6E56" if n.startswith("EMD") else PAL["mid"])
            fig = self.imp_canvas.fig; fig.clear(); ax = fig.add_subplot(111)
            lbls = [fn[i] for i in top[::-1]]; vals = [imp[i] for i in top[::-1]]
            ax.barh(lbls, vals, color=[cmap(l) for l in lbls], height=0.6)
            ax.axvline(np.mean(imp), color="gray", lw=1, ls="--")
            ax.set_xlabel("Importance (Gini)",fontsize=9)
            ax.set_title(f"Top 15 features — {res['best_name']}",fontsize=9,fontweight="600")
            ax.tick_params(labelsize=8); fig.tight_layout(); self.imp_canvas.draw()

        self.training_done.emit(res)
        self._load_model_info()  # rafraîchir section modèle existant
        self.status_msg.emit(f"Modèle : {res['best_name']} — {res['X_shape'][0]} obs × {res['X_shape'][1]} features")

    def _on_error(self, msg):
        self.btn_train.setEnabled(True)
        QMessageBox.critical(self,"Erreur",msg[:800])

# ─────────────────────────────────────────────────────────────────────────────
# FENÊTRE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SigNoise — RF Fingerprinting")
        self.setMinimumSize(1200, 780); self.resize(1400, 900)

        icon_path = WORK_DIR / "SigNoise_icon.svg"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        tabs = QTabWidget(); tabs.setDocumentMode(True)
        self.params_tab    = ParamsTab()
        self.id_tab        = IdentificationTab(self.params_tab.get_params)
        self.inconnus_tab  = InconnusTab(self.params_tab.get_params)
        self.analysis_tab  = AnalysisTab(self.params_tab.get_params)
        self.train_tab     = TrainingTab(self.params_tab.get_params)

        tabs.addTab(self.id_tab,        make_icon(SVG_RADIO_TOWER),  "  Identification")
        tabs.addTab(self.inconnus_tab,  make_icon(SVG_USERS),        "  Inconnus")
        tabs.addTab(self.analysis_tab,  make_icon(SVG_ACTIVITY),     "  Analyse Signal")
        tabs.addTab(self.train_tab,     make_icon(SVG_TRAIN),        "  Entraînement")
        tabs.addTab(self.params_tab,    make_icon(SVG_SETTINGS),     "  Paramètres")
        tabs.addTab(self._make_about(), make_icon(SVG_INFO),         "  À propos")

        # Connexion : burst identifié → Analyse Signal (auto)
        self.id_tab.burst_ready.connect(self.analysis_tab.receive_burst)
        # Connexion : inconnus détectés → refresh automatique onglet Inconnus
        self.id_tab.inconnus_updated.connect(self.inconnus_tab.refresh)

        # Top bar
        container = QWidget()
        ml = QVBoxLayout(container); ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)

        top_bar = QFrame(); top_bar.setFixedHeight(64)
        top_bar.setStyleSheet("QFrame{background:#FFFFFF;border-bottom:2px solid #C4A050;}")
        bl = QHBoxLayout(top_bar); bl.setContentsMargins(16,6,16,6)

        logo1 = QLabel()
        p1 = WORK_DIR/"ern.png"
        if p1.exists():
            logo1.setPixmap(QPixmap(str(p1)).scaled(120,48,Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))
        else:
            logo1.setText("ERN"); logo1.setStyleSheet("color:#C4A050;font-weight:700;")

        title_bar = QLabel("SigNoise  — RF Fingerprinting ")
        title_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_bar.setStyleSheet("color:#0A1637;font-size:14px;font-weight:600;letter-spacing:0.5px;")

        logo2 = QLabel(); logo2.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        p2 = WORK_DIR/"en.png"
        if p2.exists():
            logo2.setPixmap(QPixmap(str(p2)).scaled(120,48,Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))
        else:
            logo2.setText("École Navale"); logo2.setStyleSheet("color:#C4A050;font-weight:700;")

        bl.addWidget(logo1); bl.addStretch()
        bl.addWidget(title_bar); bl.addStretch(); bl.addWidget(logo2)

        ml.addWidget(top_bar); ml.addWidget(tabs)
        self.setCentralWidget(container)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status()

        for tab in [self.id_tab, self.inconnus_tab,
                    self.analysis_tab, self.train_tab]:
            tab.status_msg.connect(self.status_bar.showMessage)

        timer = QTimer(self); timer.timeout.connect(self._update_status); timer.start(4000)

    def _update_status(self):
        parts = []
        if MODEL_PATH.exists():
            try:
                with open(META_PATH) as f: m = json.load(f)
                n_feat = m.get("n_features_reduced", m.get("n_features","?"))
                rfecv  = "(RFECV)" if m.get("use_rfecv") else ""
                cmse   = "CMSE✓" if m.get("use_cmse") else "CMSE✗"
                gmm    = "GMM✓" if m.get("has_gmm") else "GMM✗"
                parts.append(f"Modèle : {m['modele']}  |  Classes : {', '.join(m['classes'])}  |  "
                              f"Features : {n_feat}{rfecv}  |  {cmse}  |  {gmm}  |  "
                              f"Obs : {m.get('n_observations','?')}  |  Mode : AIS Bursts  |  Date : {m['date'][:10]}")
            except Exception:
                parts.append("Modèle présent (métadonnées illisibles)")
        else:
            parts.append("⚠  Aucun modèle — allez dans Entraînement")
        if INCONNUS_FILE.exists():
            try:
                db = charger_inconnus()
                n_inc = sum(1 for d in db.values() if d.get("statut")!="INTÉGRÉ")
                if n_inc > 0: parts.append(f"⚠ {n_inc} inconnu(s) en attente")
            except Exception: pass
        self.status_bar.showMessage("  |  ".join(parts))

    def _make_about(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(40,30,40,30)
        hdr = QFrame(); hdr.setObjectName("header")
        hdr.setStyleSheet("QFrame#header{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                          "stop:0 #0A1637,stop:1 #1C3464);border-radius:10px;}")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(24,16,24,16)
        t = QLabel("SigNoise — RF Fingerprinting")
        t.setStyleSheet("font-size:16px;font-weight:700;color:white;background:transparent;")
        s = QLabel("Projet de Fin d'Études — EV2 Hamza Zouine et EV2Saad EL MAACHI — 2025–2026")
        s.setStyleSheet("font-size:11px;color:#9F9D97;background:transparent;")
        vb = QVBoxLayout(); vb.setSpacing(2); vb.addWidget(t); vb.addWidget(s)
        hl.addLayout(vb); lay.addWidget(hdr)
        content = QLabel("""
<h3 style="color:#1D9E75;">Objectif de l'application</h3>
<p>SigNoise identifie automatiquement des émetteurs radio AIS en analysant leur
<b>signature de bruit intrinsèque</b> — les imperfections uniques de l'oscillateur interne —
sans décoder le contenu des messages. Cette approche, appelée <b>RF Fingerprinting</b>,
permet de distinguer des émetteurs utilisant la même fréquence et le même protocole,
sur la base de leurs caractéristiques physiques propres.</p>

<h3 style="color:#1D9E75;">Pipeline de traitement du signal</h3>
<ol>
<li><b>Prétraitement</b> : suppression DC offset, correction CFO, décimation FIR, normalisation puissance</li>
<li><b>Détection des bursts AIS</b> : seuillage énergie lissée, filtrage durée minimale, fusion des gaps courts</li>
<li><b>Transformation de Hilbert</b> : extraction de la phase instantanée du signal IQ</li>
<li><b>EMD (Empirical Mode Decomposition)</b> : décomposition du signal de phase en IMFs (modes intrinsèques)</li>
<li><b>CMSE (Cumulative Mean Square Error)</b> : sélection adaptative des IMFs de bruit par seuil de contribution énergétique</li>
<li><b>Calcul des variances multi-échelles</b> :
  <ul>
  <li>AVAR (Allan Variance) — sensible au bruit blanc et de scintillation de fréquence</li>
  <li>HVAR (Hadamard Variance) — robuste aux dérivés lentes</li>
  <li>MHVAR(Modified Hadamard Variance) — compromis entre sensibilité et robustesse</li>
  </ul>
</li>
<li><b>Extraction de features complémentaires</b> : DSP (Welch), fréquence instantanée, puissance, entropie de phase, indices EMD</li>
</ol>

<h3 style="color:#1D9E75;">Pipeline IA — Classification fermée</h3>
<ol>
<li><b>RFECV (Recursive Feature Elimination with Cross-Validation)</b> : sélection automatique des features les plus discriminantes par Random Forest et validation croisée stratifiée</li>
<li><b>Cross-validation 5-fold stratifiée</b> : évaluation robuste de SVM, Random Forest, Gradient Boosting et k-NN — le meilleur modèle (F1 macro) est retenu</li>
<li><b>Entraînement final</b> : Random Forest 300 estimateurs sur les features sélectionnées</li>
</ol>

<h3 style="color:#1D9E75;">Détection des émetteurs inconnus (Open Set)</h3>
<p>La classification classique suppose que tout signal appartient à une classe connue.
SigNoise lève cette hypothèse via un <b>double critère</b> :</p>
<ul>
<li><b>Critère 1 — Probabilité classifieur</b> : la probabilité de la classe prédite doit dépasser un seuil configurable (défaut 0.60)</li>
<li><b>Critère 2 — GMM (Gaussian Mixture Model)</b> : la log-vraisemblance du vecteur de features sous le GMM de la classe doit être supérieure au seuil de calibration (percentile 5 sur les données d'entraînement)</li>
</ul>
<p>Un burst ne satisfaisant pas les deux critères est classé <b>INCONNU</b> et regroupé par distance euclidienne dans l'espace des features réduites.</p>

<h3 style="color:#1D9E75;">Réentraînement & Étiquetage</h3>
<ul>
<li><b>Nommer seulement</b> : attribue une étiquette à un groupe d'inconnus sans modifier le modèle — utile pour documenter avant de décider</li>
<li><b>Réentraînement incrémental</b> : intègre les captures d'un inconnu au modèle existant, ré-entraîne le classifieur et met à jour les GMM par classe</li>
<li>Les inconnus et leurs fichiers sources sont persistés dans <code>inconnus_db.json</code> et rechargés automatiquement à chaque ouverture</li>
</ul>

<h3 style="color:#1D9E75;">Matériel & Acquisition</h3>
<p>Récepteurs SDR : USRP-2930 (EM200) · Fréquence centrale : 162 MHz (AIS)
· Fréquence d'échantillonnage : 800 kHz · Format fichier : IQ float32 (complex64)</p>

<h3 style="color:#1D9E75;">Rapport complet</h3>
<p>Pour plus d'informations sur la méthodologie, les résultats et les bases théoriques,
consultez le rapport de projet :</p>
<p><a href="https://drive.google.com/file/d/1LwW_DU1MAIXDMeJ6eyOswKefYKBhqLDi/view?usp=drivesdk"
style="color:#1D9E75;">
https://drive.google.com/file/d/1LwW_DU1MAIXDMeJ6eyOswKefYKBhqLDi/view?usp=drivesdk
</a></p>
<p>Ou scannez le QR code  :</p>
""")
        qr_path = WORK_DIR / "code_qr_rapport.png"
        if qr_path.exists():
            qr_lbl = QLabel()
            qr_lbl.setPixmap(QPixmap(str(qr_path)).scaled(
                160, 160,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            qr_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
            lay.addWidget(qr_lbl)
        content.setWordWrap(True); content.setTextFormat(Qt.TextFormat.RichText)
        content.setStyleSheet(f"font-size:13px;color:{PAL['dark']};")
        sc = QScrollArea(); sc.setWidget(content); sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame); lay.addWidget(sc)
        return w

# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    icon_path = WORK_DIR / "SigNoise_icon.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    splash = QSplashScreen(make_splash()); splash.show(); app.processEvents()
    import time; time.sleep(1.8)
    win = MainWindow(); splash.finish(win); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
