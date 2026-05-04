"""
BioProcess Scale-Up Predictor v2
==================================
AI-driven fermentation scale-up tool with:
  - Rich colour terminal output (ANSI codes)
  - Matplotlib visualisation panel (6 charts)
  - Random Forest ML model
  - Monod kinetics + van't Riet kLa engineering equations
  - Risk scoring system

Run:  python3 bioprocess_scaleup_v2.py
"""

import numpy as np
import pandas as pd
import math
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Literal
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


# ══════════════════════════════════════════════
# ANSI COLOUR PALETTE
# ══════════════════════════════════════════════

class C:
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    RED      = "\033[31m"
    GREEN    = "\033[32m"
    YELLOW   = "\033[33m"
    BLUE     = "\033[34m"
    MAGENTA  = "\033[35m"
    CYAN     = "\033[36m"
    BRED     = "\033[91m"
    BGREEN   = "\033[92m"
    BYELLOW  = "\033[93m"
    BBLUE    = "\033[94m"
    BMAGENTA = "\033[95m"
    BCYAN    = "\033[96m"
    BWHITE   = "\033[97m"
    BG_BLUE  = "\033[44m"
    BG_DARK  = "\033[40m"

    @staticmethod
    def risk_colour(level):
        return {"LOW": C.BGREEN, "MEDIUM": C.BYELLOW, "HIGH": C.BRED}.get(level, C.BWHITE)

    @staticmethod
    def flag_icon(level):
        return {"LOW": "v", "MEDIUM": "!", "HIGH": "X"}.get(level, ".")


def sep(char="─", n=68, colour=None):
    c = colour or (C.DIM + C.CYAN)
    print(f"{c}{char * n}{C.RESET}")


def header(text):
    width = 68
    pad = (width - len(text) - 2) // 2
    print(f"\n{C.BOLD}{C.BG_BLUE}{C.BWHITE}{'=' * width}{C.RESET}")
    print(f"{C.BOLD}{C.BG_BLUE}{C.BWHITE}{' ' * pad}  {text}  {C.RESET}")
    print(f"{C.BOLD}{C.BG_BLUE}{C.BWHITE}{'=' * width}{C.RESET}")


def section(title):
    print(f"\n{C.BOLD}{C.BCYAN}  | {title}{C.RESET}")
    sep()


def kv(label, value, unit="", colour=None, width=28):
    colour = colour or C.BWHITE
    label_s = f"{C.DIM}{label:<{width}}{C.RESET}"
    val_s   = f"{C.BOLD}{colour}{value}{C.RESET}"
    unit_s  = f" {C.DIM}{unit}{C.RESET}" if unit else ""
    print(f"  {label_s}: {val_s}{unit_s}")


# ══════════════════════════════════════════════
# 1. ORGANISM PROFILES
# ══════════════════════════════════════════════

ORGANISM_PROFILES = {
    "ecoli": {
        "name": "E. coli (BL21 / K-12)",
        "opt_temp": 37.0, "opt_pH": 7.0,
        "shear_sensitivity": 0.15, "o2_demand": 0.85,
        "yield_factor": 0.88, "ks": 0.15, "ki": 20.0,
    },
    "bacillus": {
        "name": "Bacillus subtilis",
        "opt_temp": 37.0, "opt_pH": 7.0,
        "shear_sensitivity": 0.20, "o2_demand": 0.90,
        "yield_factor": 0.85, "ks": 0.12, "ki": 18.0,
    },
    "yeast": {
        "name": "Saccharomyces cerevisiae",
        "opt_temp": 30.0, "opt_pH": 5.5,
        "shear_sensitivity": 0.25, "o2_demand": 0.60,
        "yield_factor": 0.92, "ks": 0.20, "ki": 25.0,
    },
    "cho": {
        "name": "CHO cells",
        "opt_temp": 37.0, "opt_pH": 7.2,
        "shear_sensitivity": 0.80, "o2_demand": 0.40,
        "yield_factor": 0.75, "ks": 0.05, "ki": 8.0,
    },
}


# ══════════════════════════════════════════════
# 2. INPUT DATACLASS
# ══════════════════════════════════════════════

@dataclass
class LabScaleInput:
    organism: Literal["ecoli", "bacillus", "yeast", "cho"]
    pH: float
    temperature: float
    agitation_rpm: float
    aeration_vvm: float
    lab_yield_gL: float
    mu_max: float
    do_percent: float
    substrate_conc: float
    target_volume_L: float
    strategy: Literal[
        "constant_kla", "constant_pvv",
        "constant_tip_speed", "constant_Re"
    ] = "constant_kla"
    lab_volume_L: float = 5.0


# ══════════════════════════════════════════════
# 3. BIOPROCESS EQUATIONS
# ══════════════════════════════════════════════

class BioprocessEquations:

    @staticmethod
    def monod_growth_rate(mu_max, substrate, Ks):
        """mu = mu_max * S / (Ks + S)"""
        return mu_max * substrate / (Ks + substrate)

    @staticmethod
    def andrews_inhibition(mu_max, substrate, Ks, Ki):
        """Andrews: mu = mu_max * S / (Ks + S + S^2/Ki)"""
        return mu_max * substrate / (Ks + substrate + substrate**2 / Ki)

    @staticmethod
    def kla_correlation(agitation_rpm, vvm, volume_L=5.0):
        """van't Riet: kLa = 0.026 * N^0.6 * vvm^0.4  (h^-1)"""
        return round(0.026 * (agitation_rpm ** 0.6) * (vvm ** 0.4), 4)

    @staticmethod
    def oxygen_transfer_rate(kla, do_star=7.5, do_actual=2.0):
        """OTR = kLa * (DO* - DO)  mg/L/h"""
        return round(kla * (do_star - do_actual), 3)

    @staticmethod
    def mixing_time(volume_L, agitation_rpm, d=0.15):
        """Ruszkowski: tm = 5.9 * V^0.5 / (N * d^2)  seconds"""
        N = agitation_rpm / 60
        if N == 0:
            return float("inf")
        return round(5.9 * (volume_L / 1000) ** 0.5 / (N * d**2), 1)

    @staticmethod
    def tip_speed(rpm, d=0.15):
        return round(math.pi * d * rpm / 60, 2)

    @staticmethod
    def power_per_volume(rpm, volume_L, Np=5.0, rho=1000, d=0.15):
        N = rpm / 60
        return round(Np * rho * N**3 * d**5 / (volume_L / 1000), 1)

    @staticmethod
    def scale_impeller_diameter(scale_ratio, lab_d=0.05):
        return round(lab_d * scale_ratio ** (1 / 3), 4)

    @staticmethod
    def monod_curve(mu_max, Ks, s_range):
        return mu_max * s_range / (Ks + s_range)


# ══════════════════════════════════════════════
# 4. SYNTHETIC DATASET GENERATOR
# ══════════════════════════════════════════════

class SyntheticDatasetGenerator:
    def __init__(self, n=2000, seed=42):
        self.n = n
        self.seed = seed
        np.random.seed(seed)

    def generate(self):
        eq = BioprocessEquations()
        orgs   = list(ORGANISM_PROFILES.keys())
        strats = ["constant_kla", "constant_pvv", "constant_tip_speed", "constant_Re"]
        vols   = [100, 300, 1000]
        recs   = []

        for _ in range(self.n):
            ok  = np.random.choice(orgs)
            org = ORGANISM_PROFILES[ok]
            pH  = np.random.uniform(5.5, 8.5)
            T   = np.random.uniform(25, 42)
            ag  = np.random.uniform(100, 800)
            vvm = np.random.uniform(0.5, 3.0)
            ly  = np.random.uniform(0.5, 15.0)
            mu  = np.random.uniform(0.05, 1.5)
            do  = np.random.uniform(10, 80)
            sub = np.random.uniform(5, 50)
            tv  = np.random.choice(vols)
            st  = np.random.choice(strats)

            sr  = tv / 5
            ls  = math.log10(sr)
            kla = eq.kla_correlation(ag, vvm)

            if st == "constant_kla":
                ag_s, vvm_s, kla_s = ag*(sr**-0.15), vvm*(sr**0.10), kla*0.95
            elif st == "constant_pvv":
                ag_s, vvm_s, kla_s = ag*(sr**-0.33), vvm*1.1, kla*0.82
            elif st == "constant_tip_speed":
                ag_s, vvm_s, kla_s = ag/(sr**(1/3)), vvm*1.2, kla*0.70
            else:
                ag_s, vvm_s, kla_s = ag*(sr**-0.67), vvm*1.3, kla*0.65

            kd  = max(0, (kla - kla_s) / kla)
            pen = (0.05 * max(0, abs(T - org["opt_temp"]) - 2) / 3
                   + 0.04 * max(0, abs(pH - org["opt_pH"]) - 0.3)
                   + 0.04 * ls
                   + org["shear_sensitivity"] * 0.05 * ls
                   + org["o2_demand"] * kd * 0.30)

            y_sc = max(0.1, ly * org["yield_factor"] * (1 - pen) + np.random.normal(0, 0.03))
            m_sc = max(0.01, mu * (1 - (0.04*ls)*0.5 - (org["o2_demand"]*kd*0.30)*0.4))
            d_sc = max(5, do * (kla_s / max(kla, 0.01)) - 5*ls)
            r_sc = (min(100, org["shear_sensitivity"]*ag/8*ls) * 0.30
                    + min(100, 15*ls + (15 if tv > 500 else 0)) * 0.35
                    + min(100, kd*100 + org["o2_demand"]*20) * 0.35)

            recs.append({
                "organism_enc": orgs.index(ok),
                "pH": pH, "temperature": T, "agitation_rpm": ag, "aeration_vvm": vvm,
                "lab_yield_gL": ly, "mu_max": mu, "do_lab": do, "substrate_gL": sub,
                "scale_ratio": sr, "log_scale": ls, "strategy_enc": strats.index(st),
                "kla_lab": kla, "shear_sensitivity": org["shear_sensitivity"],
                "o2_demand": org["o2_demand"],
                "yield_scale": round(y_sc, 3), "mu_scale": round(m_sc, 4),
                "do_scale": round(d_sc, 1), "risk_score": round(r_sc, 1),
            })
        return pd.DataFrame(recs)


# ══════════════════════════════════════════════
# 5. ML MODEL
# ══════════════════════════════════════════════

FEATURE_COLS = [
    "organism_enc", "pH", "temperature", "agitation_rpm", "aeration_vvm",
    "lab_yield_gL", "mu_max", "do_lab", "substrate_gL", "scale_ratio",
    "log_scale", "strategy_enc", "kla_lab", "shear_sensitivity", "o2_demand",
]
TARGET_COLS = ["yield_scale", "mu_scale", "do_scale", "risk_score"]


class ScaleUpMLModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importances = {}
        self.metrics = {}

    def train(self, df):
        X  = df[FEATURE_COLS]
        Xs = self.scaler.fit_transform(X)
        yt = df[TARGET_COLS]
        Xtr, Xte, ytr, yte = train_test_split(Xs, yt, test_size=0.2, random_state=42)

        section("Training Random Forest Models")
        for t in TARGET_COLS:
            rf = RandomForestRegressor(
                n_estimators=200, max_depth=12,
                min_samples_leaf=3, n_jobs=-1, random_state=42
            )
            rf.fit(Xtr, ytr[t])
            self.models[t] = rf
            self.feature_importances[t] = dict(zip(FEATURE_COLS, rf.feature_importances_))
            ypred = rf.predict(Xte)
            mae = mean_absolute_error(yte[t], ypred)
            r2  = r2_score(yte[t], ypred)
            self.metrics[t] = {"MAE": mae, "R2": r2}
            r2_col = C.BGREEN if r2 > 0.8 else (C.BYELLOW if r2 > 0.5 else C.BRED)
            print(f"  {C.BOLD}{C.BCYAN}{t:<15}{C.RESET}  "
                  f"MAE={C.BYELLOW}{mae:.4f}{C.RESET}   "
                  f"R2={r2_col}{r2:.4f}{C.RESET}")
        self.is_trained = True

    def predict_row(self, feat_dict):
        row = pd.DataFrame([feat_dict])[FEATURE_COLS]
        Xs  = self.scaler.transform(row)
        return {t: self.models[t].predict(Xs)[0] for t in TARGET_COLS}

    def top_features(self, target, n=8):
        return sorted(self.feature_importances[target].items(), key=lambda x: -x[1])[:n]


# ══════════════════════════════════════════════
# 6. PREDICTOR ENGINE
# ══════════════════════════════════════════════

STRAT_ENC = {
    "constant_kla": 0, "constant_pvv": 1,
    "constant_tip_speed": 2, "constant_Re": 3,
}


class BioprocessScaleUpPredictor:
    def __init__(self, ml):
        self.ml = ml
        self.eq = BioprocessEquations()

    def _features(self, inp):
        org = ORGANISM_PROFILES[inp.organism]
        sr  = inp.target_volume_L / inp.lab_volume_L
        kla = self.eq.kla_correlation(inp.agitation_rpm, inp.aeration_vvm)
        return {
            "organism_enc": list(ORGANISM_PROFILES).index(inp.organism),
            "pH": inp.pH, "temperature": inp.temperature,
            "agitation_rpm": inp.agitation_rpm, "aeration_vvm": inp.aeration_vvm,
            "lab_yield_gL": inp.lab_yield_gL, "mu_max": inp.mu_max,
            "do_lab": inp.do_percent, "substrate_gL": inp.substrate_conc,
            "scale_ratio": sr, "log_scale": math.log10(sr),
            "strategy_enc": STRAT_ENC[inp.strategy], "kla_lab": kla,
            "shear_sensitivity": org["shear_sensitivity"],
            "o2_demand": org["o2_demand"],
        }

    def _scale_conditions(self, inp):
        sr   = inp.target_volume_L / inp.lab_volume_L
        cbrt = sr ** (1/3)
        kla  = self.eq.kla_correlation(inp.agitation_rpm, inp.aeration_vvm)
        d_s  = self.eq.scale_impeller_diameter(sr)

        if inp.strategy == "constant_kla":
            ags, vvms, klas = inp.agitation_rpm*(sr**-0.15), inp.aeration_vvm*(sr**0.10), kla*0.95
        elif inp.strategy == "constant_pvv":
            ags, vvms, klas = inp.agitation_rpm*(sr**-0.33), inp.aeration_vvm*1.1, kla*0.82
        elif inp.strategy == "constant_tip_speed":
            ags, vvms, klas = inp.agitation_rpm/cbrt, inp.aeration_vvm*1.2, kla*0.70
        else:
            ags, vvms, klas = inp.agitation_rpm*(sr**-0.67), inp.aeration_vvm*1.3, kla*0.65

        kd = max(0, (kla - klas) / kla)
        return {
            "agit_scale": round(ags, 0), "vvm_scale": round(vvms, 2),
            "kla_lab": round(kla, 4), "kla_scale": round(klas, 4),
            "kla_deficit_pct": round(kd * 100, 1),
            "mixing_time_s": self.eq.mixing_time(inp.target_volume_L, ags, d_s),
            "tip_speed_ms":  self.eq.tip_speed(ags, d_s),
            "pvv":           self.eq.power_per_volume(ags, inp.target_volume_L, d=d_s),
            "d_scale":       d_s,
        }

    def predict(self, inp):
        feat  = self._features(inp)
        ml    = self.ml.predict_row(feat)
        sc    = self._scale_conditions(inp)
        org   = ORGANISM_PROFILES[inp.organism]
        rl    = "LOW" if ml["risk_score"] < 35 else ("MEDIUM" if ml["risk_score"] < 65 else "HIGH")
        ydrop = round((inp.lab_yield_gL - ml["yield_scale"]) / inp.lab_yield_gL * 100, 1)
        otr   = self.eq.oxygen_transfer_rate(sc["kla_scale"])
        mu_m  = self.eq.monod_growth_rate(inp.mu_max, inp.substrate_conc, org["ks"])

        flags = []
        if ml["do_scale"] < 20:
            flags.append(("HIGH", "DO < 20% — severe O2 limitation. Add O2 enrichment."))
        if sc["tip_speed_ms"] > 5.0:
            flags.append(("HIGH", f"Tip speed {sc['tip_speed_ms']} m/s > 5.0 limit — cell damage."))
        if sc["mixing_time_s"] > 30:
            flags.append(("MEDIUM", f"Mixing time {sc['mixing_time_s']}s — substrate gradients."))
        if sc["vvm_scale"] > 2.5:
            flags.append(("HIGH", f"vvm {sc['vvm_scale']} > 2.5 — foaming risk. Antifoam required."))
        if ydrop > 25:
            flags.append(("HIGH", f"Yield drop {ydrop}% — revisit fed-batch strategy."))
        elif ydrop > 15:
            flags.append(("MEDIUM", f"Yield drop {ydrop}% — validate at intermediate scale."))
        if sc["kla_deficit_pct"] > 20:
            flags.append(("MEDIUM", f"kLa deficit {sc['kla_deficit_pct']}% — O2 transfer bottleneck."))
        if not flags:
            flags.append(("LOW", "No critical flags. Proceed with standard monitoring."))

        recs = [
            f"Set agitation to {int(sc['agit_scale'])} rpm via {inp.strategy.replace('_',' ')} strategy.",
            f"Aeration: {sc['vvm_scale']} vvm. Use DO cascade — increase vvm before rpm.",
            "Target mu = 0.15-0.25 h-1 with fed-batch to suppress overflow metabolism.",
            f"pH +/-0.1 | Temp +/-0.5 C. Verify jacket cooling capacity at {inp.target_volume_L}L.",
            f"kLa: {sc['kla_lab']} -> {sc['kla_scale']} h-1 ({sc['kla_deficit_pct']}% drop). OUR every 30 min.",
            f"Mixing time {sc['mixing_time_s']}s. Sample bottom/mid/top every 2 h.",
            f"Tip speed {sc['tip_speed_ms']} m/s — {'safe.' if sc['tip_speed_ms']<5 else 'EXCEEDS 5 m/s limit — reduce rpm.'}",
            f"P/V = {sc['pvv']} W/m3 — {'adequate.' if sc['pvv']>1000 else 'low — check baffle geometry.'}",
        ]
        if org["shear_sensitivity"] > 0.5:
            recs.append(f"{org['name']} is shear-sensitive — use pitched-blade (Np~0.35) not Rushton.")

        return {
            "input": inp, "org": org, "ml": ml, "sc": sc,
            "risk_level": rl, "yield_drop_pct": ydrop,
            "otr": otr, "mu_monod": mu_m,
            "flags": flags, "recs": recs,
        }


# ══════════════════════════════════════════════
# 7. COLOURED REPORT PRINTER
# ══════════════════════════════════════════════

def print_report(res, example_num=1):
    inp = res["input"]
    ml  = res["ml"]
    sc  = res["sc"]
    rl  = res["risk_level"]
    rc  = C.risk_colour(rl)
    org = res["org"]

    header(f"EXAMPLE {example_num}  {org['name'].upper()}  ->  {int(inp.target_volume_L)}L")

    section("Input Conditions (Lab Scale 5L)")
    kv("Organism",      org["name"],                  colour=C.BMAGENTA)
    kv("Target volume", f"{inp.target_volume_L} L",   colour=C.BCYAN)
    kv("Strategy",      inp.strategy.replace("_"," "),colour=C.BCYAN)
    kv("pH",            inp.pH,                       colour=C.BWHITE)
    kv("Temperature",   f"{inp.temperature} C",       colour=C.BYELLOW)
    kv("Agitation",     f"{inp.agitation_rpm} rpm",   colour=C.BWHITE)
    kv("Aeration",      f"{inp.aeration_vvm} vvm",    colour=C.BWHITE)
    kv("Lab yield",     f"{inp.lab_yield_gL} g/L",    colour=C.BGREEN)
    kv("mu_max",        f"{inp.mu_max} h-1",          colour=C.BWHITE)
    kv("DO (lab)",      f"{inp.do_percent} %",         colour=C.BWHITE)
    kv("Substrate",     f"{inp.substrate_conc} g/L",  colour=C.BWHITE)

    section("ML Predictions (Random Forest)")
    yd_col = C.BGREEN if abs(res["yield_drop_pct"]) < 15 else (C.BYELLOW if abs(res["yield_drop_pct"]) < 25 else C.BRED)
    kv("Predicted yield",   f"{ml['yield_scale']:.2f} g/L",  colour=C.BGREEN)
    kv("Yield drop vs lab", f"{res['yield_drop_pct']} %",    colour=yd_col)
    kv("Predicted mu",      f"{ml['mu_scale']:.4f} h-1",     colour=C.BWHITE)
    kv("Predicted DO",      f"{ml['do_scale']:.1f} %",       colour=C.BCYAN)
    kv("Risk score",        f"{ml['risk_score']:.1f} / 100", colour=rc)

    bar_fill  = int(ml['risk_score'] / 100 * 30)
    bar_empty = 30 - bar_fill
    bar_col   = C.BGREEN if rl == "LOW" else (C.BYELLOW if rl == "MEDIUM" else C.BRED)
    print(f"\n  {C.BOLD}Risk Level:{C.RESET}  "
          f"{bar_col}{C.BOLD}{'#'*bar_fill}{'.'*bar_empty}{C.RESET}  "
          f"{bar_col}{C.BOLD}[{rl}]{C.RESET}")

    section("Bioprocess Engineering Calculations")
    kv("kLa (lab)",      f"{sc['kla_lab']} h-1",        colour=C.BBLUE)
    kv("kLa (scale)",    f"{sc['kla_scale']} h-1",      colour=C.BCYAN)
    kd_col = C.BGREEN if sc["kla_deficit_pct"] < 10 else (C.BYELLOW if sc["kla_deficit_pct"] < 25 else C.BRED)
    kv("kLa deficit",    f"{sc['kla_deficit_pct']} %",  colour=kd_col)
    kv("Mixing time",    f"{sc['mixing_time_s']} s",     colour=C.BWHITE)
    kv("Tip speed",      f"{sc['tip_speed_ms']} m/s",
       colour=C.BGREEN if sc["tip_speed_ms"] < 5 else C.BRED)
    kv("Power/Volume",   f"{sc['pvv']} W/m3",            colour=C.BWHITE)
    kv("OTR",            f"{res['otr']} mg/L/h",         colour=C.BWHITE)
    kv("mu (Monod)",     f"{res['mu_monod']:.4f} h-1",  colour=C.BWHITE)

    section(f"Recommended Conditions at {int(inp.target_volume_L)}L")
    kv("Agitation",      f"{int(sc['agit_scale'])} rpm", colour=C.BGREEN)
    kv("Aeration",       f"{sc['vvm_scale']} vvm",        colour=C.BGREEN)
    kv("Impeller diam.", f"{sc['d_scale']} m",            colour=C.BWHITE)

    section("Risk Flags")
    for level, msg in res["flags"]:
        col  = C.risk_colour(level)
        icon = C.flag_icon(level)
        print(f"  {col}{C.BOLD}{icon} [{level}]{C.RESET}  {msg}")

    section("Engineering Recommendations")
    for i, rec in enumerate(res["recs"], 1):
        words = rec.split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 58:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        for j, line in enumerate(lines):
            pfx = f"  {C.BOLD}{C.BCYAN}{i}.{C.RESET} " if j == 0 else "     "
            print(f"{pfx}{line}")

    sep()
    print(f"\n  {C.BOLD}{C.BYELLOW}! MODEL LIMITATIONS{C.RESET}")
    for lim in [
        "Correlations assume Rushton turbine + water-like broth.",
        "No foam, viscosity, or contamination modelling.",
        "Trained on synthetic data — replace with real plant runs.",
        "CHO / mammalian predictions are least reliable.",
        "Validate with a 50L intermediate run before 1000L.",
    ]:
        print(f"  {C.DIM}* {lim}{C.RESET}")
    sep(char="=")


# ══════════════════════════════════════════════
# 8. VISUALISATION — 6-chart dashboard
# ══════════════════════════════════════════════

CHART_BG  = "#0F1117"
PANEL_BG  = "#161B22"
ACCENT1   = "#58A6FF"
ACCENT2   = "#3FB950"
ACCENT3   = "#F78166"
ACCENT4   = "#D2A8FF"
ACCENT5   = "#FFA657"
TEXT_MAIN = "#E6EDF3"
TEXT_DIM  = "#8B949E"
GRID_COL  = "#21262D"


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.7)
    if title:  ax.set_title(title, color=TEXT_MAIN, fontsize=10, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)


def generate_charts(results, df, model):
    fig = plt.figure(figsize=(18, 11), facecolor=CHART_BG)
    fig.suptitle(
        "BioProcess Scale-Up Predictor  —  Analysis Dashboard",
        color=TEXT_MAIN, fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)
    eq = BioprocessEquations()
    palette = [ACCENT1, ACCENT2, ACCENT3]

    labels = [f"Ex{i+1}\n{int(r['input'].target_volume_L)}L" for i, r in enumerate(results)]

    # ── Chart 1: kLa comparison ──
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "kLa: Lab vs Scale-up", "Scenario", "kLa (h-1)")
    kla_lab = [r["sc"]["kla_lab"]   for r in results]
    kla_sc  = [r["sc"]["kla_scale"] for r in results]
    x = np.arange(len(labels)); w = 0.35
    b1 = ax1.bar(x - w/2, kla_lab, w, color=ACCENT1, alpha=0.85, label="Lab (5L)")
    b2 = ax1.bar(x + w/2, kla_sc,  w, color=ACCENT3, alpha=0.85, label="Scale")
    ax1.bar_label(b1, fmt="%.2f", fontsize=7, color=ACCENT1, padding=2)
    ax1.bar_label(b2, fmt="%.2f", fontsize=7, color=ACCENT3, padding=2)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8, color=TEXT_DIM)
    ax1.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_MAIN, framealpha=0.5)

    # ── Chart 2: Yield comparison ──
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Yield: Lab vs Predicted Scale", "Scenario", "g/L")
    lab_y = [r["input"].lab_yield_gL   for r in results]
    sc_y  = [r["ml"]["yield_scale"]    for r in results]
    b3 = ax2.bar(x - w/2, lab_y, w, color=ACCENT2, alpha=0.85, label="Lab yield")
    b4 = ax2.bar(x + w/2, sc_y,  w, color=ACCENT4, alpha=0.85, label="Scale yield")
    ax2.bar_label(b3, fmt="%.1f", fontsize=7, color=ACCENT2, padding=2)
    ax2.bar_label(b4, fmt="%.1f", fontsize=7, color=ACCENT4, padding=2)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8, color=TEXT_DIM)
    ax2.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_MAIN, framealpha=0.5)

    # ── Chart 3: Risk score gauge ──
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, "Risk Score Comparison", "Risk Score (0-100)", "")
    mpl_cols = {"LOW": "#3FB950", "MEDIUM": "#FFA657", "HIGH": "#F78166"}
    bar_cols  = [mpl_cols[r["risk_level"]] for r in results]
    risk_scores = [r["ml"]["risk_score"] for r in results]
    y3 = np.arange(len(labels))
    bars = ax3.barh(y3, risk_scores, color=bar_cols, alpha=0.85, height=0.5)
    ax3.set_yticks(y3); ax3.set_yticklabels(labels, fontsize=8, color=TEXT_DIM)
    ax3.set_xlim(0, 100)
    ax3.axvline(35, color=ACCENT2, linestyle="--", linewidth=0.8, alpha=0.6)
    ax3.axvline(65, color=ACCENT3, linestyle="--", linewidth=0.8, alpha=0.6)
    ax3.text(17, -0.7, "LOW",  color=ACCENT2, fontsize=7, ha="center")
    ax3.text(50, -0.7, "MED",  color=ACCENT5, fontsize=7, ha="center")
    ax3.text(82, -0.7, "HIGH", color=ACCENT3, fontsize=7, ha="center")
    for bar, score in zip(bars, risk_scores):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{score:.0f}", va="center", fontsize=8, color=TEXT_MAIN)

    # ── Chart 4: Monod kinetics curves ──
    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4, "Monod Growth Kinetics", "Substrate (g/L)", "Growth rate h-1")
    s_range = np.linspace(0, 40, 300)
    for i, res in enumerate(results):
        org_p = res["org"]; inp = res["input"]
        mu_curve = eq.monod_curve(inp.mu_max, org_p["ks"], s_range)
        lab_name = f"Ex{i+1} {org_p['name'].split()[0]}"
        ax4.plot(s_range, mu_curve, color=palette[i], linewidth=1.8, label=lab_name, alpha=0.9)
        ax4.axvline(org_p["ks"], color=palette[i], linestyle=":", linewidth=0.8, alpha=0.5)
        ax4.scatter([inp.substrate_conc], [eq.monod_growth_rate(inp.mu_max, inp.substrate_conc, org_p["ks"])],
                    color=palette[i], s=50, zorder=5)
    ax4.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_MAIN, framealpha=0.5)

    # ── Chart 5: Feature importance ──
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5, "Feature Importance (Yield RF)", "Importance Score", "")
    feat_imp = model.top_features("yield_scale", n=8)
    names = [f[0].replace("_", " ") for f in feat_imp]
    imps  = [f[1] for f in feat_imp]
    bar_cs = [ACCENT1 if imp == max(imps) else ACCENT4 for imp in imps]
    y5 = np.arange(len(names))
    hb = ax5.barh(y5, imps, color=bar_cs, alpha=0.85, height=0.6)
    ax5.set_yticks(y5); ax5.set_yticklabels(names, fontsize=7, color=TEXT_DIM)
    ax5.bar_label(hb, fmt="%.3f", fontsize=7, color=TEXT_DIM, padding=3)

    # ── Chart 6: Mixing time vs volume ──
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6, "Mixing Time vs Scale Volume", "Volume (L)", "Mixing Time (s)")
    vols_range = np.linspace(5, 1050, 200)
    d_range    = [eq.scale_impeller_diameter(v/5, 0.05) for v in vols_range]
    strat_curves = [
        (400*(vols_range/5)**-0.15, ACCENT1, "Const. kLa"),
        (400*(vols_range/5)**-0.33, ACCENT2, "Const. P/V"),
        (400/(vols_range/5)**(1/3), ACCENT3, "Const. Tip"),
    ]
    for agit_curve, col, lab in strat_curves:
        tm_curve = [eq.mixing_time(v, a, d) for v, a, d in zip(vols_range, agit_curve, d_range)]
        ax6.plot(vols_range, tm_curve, color=col, linewidth=1.8, label=lab, alpha=0.9)
    ax6.axhline(30, color=ACCENT5, linestyle="--", linewidth=0.9, alpha=0.7)
    ax6.text(900, 32, "30s limit", color=ACCENT5, fontsize=7, va="bottom")
    ax6.set_xlim(0, 1050); ax6.set_ylim(0)
    ax6.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_MAIN, framealpha=0.5)
    for i, res in enumerate(results):
        vl = res["input"].target_volume_L
        tm = res["sc"]["mixing_time_s"]
        ax6.scatter(vl, tm, color=palette[i], s=60, zorder=5,
                    edgecolors=TEXT_MAIN, linewidths=0.5)
        ax6.annotate(f"Ex{i+1}", (vl, tm), textcoords="offset points",
                     xytext=(5, 5), fontsize=7, color=palette[i])

    out = "/home/claude/scaleup_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=CHART_BG)
    plt.close()
    print(f"\n  {C.BGREEN}Charts saved -> {out}{C.RESET}")
    return out


# ══════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════

def main():
    print(f"""
{C.BOLD}{C.BBLUE}
+======================================================================+
|   BioProcess Scale-Up Predictor v2  |  Industrial Biotech AI        |
|   Organisms: E.coli  B.subtilis  S.cerevisiae  CHO                  |
|   Scale: 5L lab  ->  100 / 300 / 1000L production                   |
+======================================================================+
{C.RESET}""")

    print(f"  {C.BCYAN}Generating synthetic training dataset...{C.RESET}")
    gen = SyntheticDatasetGenerator(n=2000)
    df  = gen.generate()
    print(f"  {C.BGREEN}Done{C.RESET} — {C.BOLD}{df.shape[0]} rows x {df.shape[1]} cols{C.RESET}")

    ml = ScaleUpMLModel()
    ml.train(df)

    print(f"\n  {C.DIM}Top features driving yield prediction:{C.RESET}")
    for feat, imp in ml.top_features("yield_scale"):
        bar = int(imp * 200)
        print(f"  {C.DIM}{feat:<22}{C.RESET} {C.BBLUE}{'|'*bar}{C.RESET} {imp:.4f}")

    predictor = BioprocessScaleUpPredictor(ml)

    r1 = predictor.predict(LabScaleInput(
        organism="ecoli", pH=7.0, temperature=37.0,
        agitation_rpm=400, aeration_vvm=1.0,
        lab_yield_gL=4.2, mu_max=0.65, do_percent=30,
        substrate_conc=20.0, target_volume_L=1000, strategy="constant_kla",
    ))
    print_report(r1, 1)

    r2 = predictor.predict(LabScaleInput(
        organism="cho", pH=7.2, temperature=37.0,
        agitation_rpm=120, aeration_vvm=0.5,
        lab_yield_gL=0.8, mu_max=0.04, do_percent=50,
        substrate_conc=5.0, target_volume_L=300, strategy="constant_pvv",
    ))
    print_report(r2, 2)

    r3 = predictor.predict(LabScaleInput(
        organism="yeast", pH=5.5, temperature=30.0,
        agitation_rpm=300, aeration_vvm=1.5,
        lab_yield_gL=8.0, mu_max=0.35, do_percent=40,
        substrate_conc=30.0, target_volume_L=100, strategy="constant_tip_speed",
    ))
    print_report(r3, 3)

    section("Generating 6-Chart Visualisation Dashboard")
    chart_path = generate_charts([r1, r2, r3], df, ml)

    df.to_csv("/home/claude/scaleup_training_data.csv", index=False)
    print(f"  {C.BGREEN}CSV saved{C.RESET} -> scaleup_training_data.csv")

    sep(char="=")
    print(f"\n  {C.BOLD}{C.BGREEN}All done.{C.RESET}  Outputs: .py  .png  .csv\n")


if __name__ == "__main__":
    main()
