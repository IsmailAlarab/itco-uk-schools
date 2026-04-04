"""
Requirements:
    pip install numpy pandas scipy scikit-learn matplotlib seaborn pythermalcomfort

Dataset:
    Place the two Korsavi UK-schools CSV files in the same directory:
        korsavi_part1.csv
        korsavi_part2.csv
    Available (free registration) at: https://ashraeobdatabase.com/
    Export parameters: Country = UK: Coventry | Building = Educational: Classroom
                       Study = Study 1

Usage:
    python thermal_comfort_study.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative
from scipy.optimize import minimize, differential_evolution
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F6AE2D", "#A23B72"]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PART1_PATH = "korsavi_part1.csv"
PART2_PATH = "korsavi_part2.csv"


def compute_pmv_row(row):
    """
    Compute PMV for a single DataFrame row using pythermalcomfort.
    Metabolic rate fixed at 1.2 met (seated, light activity; ASHRAE-55 Table C2).
    Relative air speed corrected per ASHRAE-55 before passing to pmv_ppd.
    """
    try:
        vr = v_relative(v=row["Air_Speed"], met=1.2)
        result = pmv_ppd(
            tdb=row["Temperature_Air"],
            tr=row.get("Temperature_Radiant", row["Temperature_Air"]),
            vr=vr,
            rh=row["Relative_Humidity"],
            met=1.2,
            clo=row["Clothing_Value"],
            standard="ASHRAE",
        )
        return result["pmv"]
    except Exception:
        return np.nan


def load_and_prepare(part1_path=PART1_PATH, part2_path=PART2_PATH):
    p1 = pd.read_csv(part1_path, sep=None, engine="python")
    p2 = pd.read_csv(part2_path, sep=None, engine="python")
    p1.columns = p1.columns.str.strip()
    p2.columns = p2.columns.str.strip()
    print(f"[Data] Part-1 shape: {p1.shape}  Part-2 shape: {p2.shape}")

    def get_window_col(df):
        return next(
            (c for c in df.columns
             if "percentofopenwindow" in c.lower().replace(" ", "")),
            None)

    for df in (p1, p2):
        wc = get_window_col(df)
        df["Window_Open_Frac"] = (
            pd.to_numeric(df[wc], errors="coerce").fillna(0) / 100
            if wc else 0.0)

    for df in (p1, p2):
        if "Temperature_Radiant" not in df.columns:
            df["Temperature_Radiant"] = df.get(
                "Temperature_Operative", df.get("Temperature_Air", np.nan))

    if "Clothing_Value" not in p2.columns:
        p2["Clothing_Value"] = np.nan

    def get_outdoor_col(df):
        return next(
            (c for c in df.columns
             if "outdoor" in c.lower()
             and "temperature" in c.lower()
             and "humidity" not in c.lower()
             and "speed" not in c.lower()),
            None)

    for df in (p1, p2):
        oc = get_outdoor_col(df)
        df["Temperature_Outdoor"] = (
            pd.to_numeric(df[oc], errors="coerce") if oc else np.nan)

    shared = ["Temperature_Air", "Temperature_Radiant", "Air_Speed",
              "Relative_Humidity", "Clothing_Value", "Season", "Mode",
              "Window_Open_Frac", "Temperature_Outdoor"]
    p1["source"] = "p1"
    p2["source"] = "p2"
    p1k = p1[[c for c in shared + ["source"] if c in p1.columns]].copy()
    p2k = p2[[c for c in shared + ["source"] if c in p2.columns]].copy()
    df  = pd.concat([p1k, p2k], ignore_index=True)

    for col in shared:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["Clothing_Value"].isna().any():
        med = df["Clothing_Value"].median()
        print(f"[Data] Imputing {df['Clothing_Value'].isna().sum()} missing "
              f"Clothing_Value entries with median = {med:.3f} clo")
        df["Clothing_Value"] = df["Clothing_Value"].fillna(med)

    df = df.dropna(subset=["Temperature_Air", "Air_Speed",
                            "Relative_Humidity", "Clothing_Value"])
    df = df[df["Temperature_Air"].between(10, 35)]
    df = df[df["Relative_Humidity"].between(5, 100)]
    df = df[df["Air_Speed"].between(0, 2)]
    df["Clothing_Value"] = df["Clothing_Value"].clip(0.1, 1.5)
    df["Temperature_Radiant"] = df["Temperature_Radiant"].fillna(
        df["Temperature_Air"])
    df = df.reset_index(drop=True)

    df["PMV"] = df.apply(compute_pmv_row, axis=1)
    df = df.dropna(subset=["PMV"]).reset_index(drop=True)
    df["Comfort"] = ((df["PMV"] < -0.5) | (df["PMV"] > 0.5)).astype(int)

    # Mode column: 1 = non-heating, 2 = heating (Korsavi data dictionary)
    if "Mode" in df.columns:
        mode_num = pd.to_numeric(df["Mode"], errors="coerce")
        df["Heating"] = (mode_num == 2).astype(int)
        if df["Heating"].sum() == 0 and "Season" in df.columns:
            df["Heating"] = df["Season"].isin([1, 4]).astype(int)
    elif "Season" in df.columns:
        df["Heating"] = df["Season"].isin([1, 4]).astype(int)
    else:
        df["Heating"] = (df["Temperature_Air"] < 19).astype(int)

    df["Window_Open_Frac"] = df.get("Window_Open_Frac",
                                    pd.Series(0.0, index=df.index)).fillna(0.0)
    df["Window_Status"] = (df["Window_Open_Frac"] > 0.05).astype(float)

    print(
        f"[Data] Clean dataset: {len(df)} rows | "
        f"Comfort: {(df['Comfort']==0).sum()} | "
        f"Discomfort: {(df['Comfort']==1).sum()} | "
        f"Heating: {df['Heating'].sum()} | "
        f"Cooling: {(df['Heating']==0).sum()}"
    )
    return df



FEATURES = [
    "Temperature_Air",
    "Air_Speed",
    "Relative_Humidity",
    "Clothing_Value",
    "Window_Status",
]

FEATURE_LABELS = {
    "Temperature_Air":   r"$T_{air}$ [°C]",
    "Air_Speed":         r"$v_r$ [m/s]",
    "Relative_Humidity": "RH [%]",
    "Clothing_Value":    r"$I_{clo}$",
    "Window_Status":     "Window [0/1]",
}

BOUNDS_ORIG = {
    "Temperature_Air": (-5.0, +5.0),
    "Clothing_Value":  (-0.3, +0.3),
    "Window_Status":   (0.0,  1.0),
}

CASES = {
    "Temperature":         ["Temperature_Air"],
    "Temp + Window":       ["Temperature_Air", "Window_Status"],
    "Temp + Window + Clo": ["Temperature_Air", "Window_Status", "Clothing_Value"],
}

METHODS = ["COBYLA", "DE"]



def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_SEED, solver="lbfgs"),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(20,), activation="relu", max_iter=1000,
            learning_rate_init=0.01, random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_SEED),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models  = build_models()
    results = []
    fitted  = {}

    for name, clf in models.items():
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        cv_aucs = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        results.append({
            "Model":       name,
            "Acc [%]":    round(accuracy_score(y_test, y_pred) * 100, 1),
            "F1 [%]":     round(f1_score(y_test, y_pred) * 100, 1),
            "AUC [%]":    round(roc_auc_score(y_test, y_prob) * 100, 1),
            "CV-AUC [%]": f"{cv_aucs.mean()*100:.1f} ± {cv_aucs.std()*100:.1f}",
        })
        fitted[name] = clf
        print(f"  {name:25s}  Acc={results[-1]['Acc [%]']}%  "
              f"AUC={results[-1]['AUC [%]']}%  "
              f"CV-AUC={results[-1]['CV-AUC [%]']}%")

    return fitted, pd.DataFrame(results)


class ThermalComfortOptimiser:
    """
    Unified inverse optimisation class.

    Solvers:
        COBYLA  — gradient-free local solver (baseline)
        DE      — Differential Evolution global solver (proposed)
    """

    def __init__(self, model, scaler, features, heating: bool,
                 opt_features=("Temperature_Air",)):
        self.model     = model
        self.scaler    = scaler
        self.features  = features
        self.heating   = heating
        self.opt_feats = list(opt_features)
        self.opt_idx   = [features.index(f) for f in opt_features]
        self.temp_idx  = features.index("Temperature_Air")

    def _to_norm(self, x_orig):
        return self.scaler.transform(x_orig.reshape(1, -1))[0]

    def _to_orig(self, x_norm):
        return self.scaler.inverse_transform(x_norm.reshape(1, -1))[0]

    def _prob_discomfort(self, x_norm):
        return self.model.predict_proba(x_norm.reshape(1, -1))[0][1]

    def _objective(self, delta, x0_norm):
        """
        Penalised objective: minimise temperature change + comfort penalty.
        λ = 10 ensures comfort restoration dominates the energy cost term.
        """
        x_new = x0_norm.copy()
        for i, fi in enumerate(self.opt_idx):
            x_new[fi] = x0_norm[fi] + delta[i]

        x_new_orig = self._to_orig(x_new)
        x0_orig    = self._to_orig(x0_norm)
        dt         = x_new_orig[self.temp_idx] - x0_orig[self.temp_idx]
        temp_cost  = dt if self.heating else -dt

        return temp_cost + 10.0 * self._prob_discomfort(x_new)

    def _bounds_norm(self):
        bounds = []
        for fi in self.opt_idx:
            fname = self.features[fi]
            lo, hi = BOUNDS_ORIG.get(fname, (-1, 1))
            sigma  = self.scaler.scale_[fi]
            bounds.append((lo / sigma, hi / sigma))
        return bounds

    def _solve_cobyla(self, x0_norm):
        delta0 = np.zeros(len(self.opt_idx))
        constraints = []

        def comfort_con(delta):
            x_new = x0_norm.copy()
            for i, fi in enumerate(self.opt_idx):
                x_new[fi] = x0_norm[fi] + delta[i]
            return 0.55 - self._prob_discomfort(x_new)

        constraints.append({"type": "ineq", "fun": comfort_con})
        for i, (lo, hi) in enumerate(self._bounds_norm()):
            constraints.append({"type": "ineq",
                                 "fun": lambda d, i=i, lo=lo: d[i] - lo})
            constraints.append({"type": "ineq",
                                 "fun": lambda d, i=i, hi=hi: hi - d[i]})

        res = minimize(
            self._objective, delta0,
            args=(x0_norm,),
            method="COBYLA",
            constraints=constraints,
            options={"maxiter": 500, "rhobeg": 0.1},
        )
        return res.x

    def _solve_de(self, x0_norm):
        res = differential_evolution(
            func=self._objective,
            bounds=self._bounds_norm(),
            args=(x0_norm,),
            seed=RANDOM_SEED,
            maxiter=200,
            tol=1e-6,
            popsize=12,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=True,
        )
        return res.x

    def optimise(self, x0_orig, method="DE"):
        x0_norm = self._to_norm(np.asarray(x0_orig, dtype=float))
        y0 = self.model.predict(x0_norm.reshape(1, -1))[0]

        if y0 == 0:
            return np.asarray(x0_orig, dtype=float), 0.0, True

        delta = self._solve_cobyla(x0_norm) if method == "COBYLA" \
            else self._solve_de(x0_norm)

        x_new_norm = x0_norm.copy()
        for i, fi in enumerate(self.opt_idx):
            x_new_norm[fi] = x0_norm[fi] + delta[i]

        x_new_orig = self._to_orig(x_new_norm)
        y_new      = self.model.predict(x_new_norm.reshape(1, -1))[0]
        delta_T    = x_new_orig[self.temp_idx] - float(
            x0_orig[self.features.index("Temperature_Air")])

        return x_new_orig, delta_T, (y_new == 0)



def pmv_feedback_optimise(row, heating):
    """
    Iterative PMV feedback: step temperature until PMV ∈ (−0.5, 0.5).
    Returns (new_temperature [°C], delta_T [°C]).
    """
    tdb = row["Temperature_Air"]
    tr  = row.get("Temperature_Radiant", tdb)
    vr  = v_relative(v=row["Air_Speed"], met=1.2)
    rh  = row["Relative_Humidity"]
    clo = row["Clothing_Value"]

    if -0.5 <= pmv_ppd(tdb, tr, vr, rh, 1.2, clo, standard="ASHRAE")["pmv"] <= 0.5:
        return tdb, 0.0

    step  = 0.1 if heating else -0.1
    t_cur = tdb
    for _ in range(200):
        t_cur += step
        if -0.5 <= pmv_ppd(t_cur, t_cur + (tr - tdb), vr, rh, 1.2, clo,
                            standard="ASHRAE")["pmv"] <= 0.5:
            break
    return t_cur, t_cur - tdb



def run_optimisation(df_cands, fitted_models, scaler):
    records = []
    raw     = {}
    N_opt   = len(df_cands)

    for model_name, clf in fitted_models.items():
        for case_name, opt_feats in CASES.items():
            for method in METHODS:
                print(f"  [{model_name}] [{case_name}] [{method}] ...",
                      flush=True)
                dT_W, dT_S = [], []
                successes  = 0

                for _, row in df_cands.iterrows():
                    x0 = row[FEATURES].values.astype(float)
                    opt = ThermalComfortOptimiser(
                        clf, scaler, FEATURES,
                        heating=bool(row["Heating"]),
                        opt_features=opt_feats)
                    _, dT, ok = opt.optimise(x0, method=method)
                    successes += int(ok)
                    (dT_W if row["Heating"] else dT_S).append(abs(dT))

                sum_W = sum(dT_W)
                sum_S = sum(dT_S)
                N_W   = len(dT_W)
                N_S   = len(dT_S)

                raw[(model_name, case_name, method)] = {"dT_W": dT_W, "dT_S": dT_S}

                records.append({
                    "Model":    model_name,
                    "Case":     case_name,
                    "Solver":   method,
                    "N_W":      N_W,
                    "N_S":      N_S,
                    "ΣΔT_W":   round(sum_W, 1),
                    "ΣΔT_S":   round(sum_S, 1),
                    "MATAW":    round(sum_W / N_W, 2) if N_W else np.nan,
                    "MATAS":    round(sum_S / N_S, 2) if N_S else np.nan,
                    "HAF [%]":  round(sum_W / (sum_W + sum_S) * 100, 1)
                                if (sum_W + sum_S) > 0 else np.nan,
                    "Success":  f"{successes}/{N_opt}",
                    "SuccessN": successes,
                })

    dT_W_pmv, dT_S_pmv = [], []
    for _, row in df_cands.iterrows():
        _, dT = pmv_feedback_optimise(row, heating=bool(row["Heating"]))
        (dT_W_pmv if row["Heating"] else dT_S_pmv).append(abs(dT))

    sum_W = sum(dT_W_pmv)
    sum_S = sum(dT_S_pmv)
    N_W   = len(dT_W_pmv)
    N_S   = len(dT_S_pmv)
    raw[("PMV Feedback", "Temperature", "Iterative")] = {
        "dT_W": dT_W_pmv, "dT_S": dT_S_pmv}

    records.append({
        "Model":    "PMV Feedback",
        "Case":     "Temperature",
        "Solver":   "Iterative",
        "N_W":      N_W,
        "N_S":      N_S,
        "ΣΔT_W":   round(sum_W, 1),
        "ΣΔT_S":   round(sum_S, 1),
        "MATAW":    round(sum_W / N_W, 2) if N_W else np.nan,
        "MATAS":    round(sum_S / N_S, 2) if N_S else np.nan,
        "HAF [%]":  round(sum_W / (sum_W + sum_S) * 100, 1)
                    if (sum_W + sum_S) > 0 else np.nan,
        "Success":  "N/A",
        "SuccessN": np.nan,
    })

    return pd.DataFrame(records), raw


def two_prop_z(s1, n1, s2, n2):
    from scipy.stats import norm
    p1    = s1 / n1
    p2    = s2 / n2
    p_hat = (s1 + s2) / (n1 + n2)
    se    = np.sqrt(p_hat * (1 - p_hat) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return round(z, 2), float(p)


def run_statistical_tests(res_df, N_opt):
    cross_solver = [
        ("Gradient Boosting",   "Temperature",
         "GB: DE vs COBYLA (Temperature)"),
        ("Gradient Boosting",   "Temp + Window + Clo",
         "GB: DE vs COBYLA (Temp+Win+Clo)"),
        ("Logistic Regression", "Temperature",
         "LR: COBYLA vs DE (Temperature)"),
    ]
    feat_comparisons = [
        ("Gradient Boosting",   "DE",     "Temperature",
         "Temp + Window + Clo", "GB+DE: +Clo vs Temp-only"),
        ("Logistic Regression", "DE",     "Temperature",
         "Temp + Window + Clo", "LR+DE: +Clo vs Temp-only"),
        ("MLP",                 "COBYLA", "Temperature",
         "Temp + Window",       "MLP+COBYLA: +Win vs Temp-only"),
    ]

    rows = []

    for model, case, label in cross_solver:
        de_row  = res_df[(res_df["Model"] == model) & (res_df["Case"] == case)
                         & (res_df["Solver"] == "DE")]
        cob_row = res_df[(res_df["Model"] == model) & (res_df["Case"] == case)
                         & (res_df["Solver"] == "COBYLA")]
        if de_row.empty or cob_row.empty:
            continue
        s1 = int(de_row["SuccessN"].iloc[0])
        s2 = int(cob_row["SuccessN"].iloc[0])
        z, p = two_prop_z(s1, N_opt, s2, N_opt)
        rows.append({"Comparison": label,
                     "Success 1": f"{s1}/{N_opt}",
                     "Success 2": f"{s2}/{N_opt}",
                     "z": z, "p": f"{p:.3e}",
                     "Sig. α<0.05": "✓" if p < 0.05 else "✗"})

    for model, solver, case1, case2, label in feat_comparisons:
        r1 = res_df[(res_df["Model"] == model) & (res_df["Solver"] == solver)
                    & (res_df["Case"] == case2)]
        r2 = res_df[(res_df["Model"] == model) & (res_df["Solver"] == solver)
                    & (res_df["Case"] == case1)]
        if r1.empty or r2.empty:
            continue
        s1 = int(r1["SuccessN"].iloc[0])
        s2 = int(r2["SuccessN"].iloc[0])
        z, p = two_prop_z(s1, N_opt, s2, N_opt)
        rows.append({"Comparison": label,
                     "Success 1": f"{s1}/{N_opt}",
                     "Success 2": f"{s2}/{N_opt}",
                     "z": z, "p": f"{p:.3e}",
                     "Sig. α<0.05": "✓" if p < 0.05 else "✗"})

    return pd.DataFrame(rows)


def prescriptive_analysis(df_test, scaler, fitted_models):
    clf_lr  = fitted_models["Logistic Regression"]
    clo_idx = FEATURES.index("Clothing_Value")
    win_idx = FEATURES.index("Window_Status")
    tmp_idx = FEATURES.index("Temperature_Air")

    Xt = scaler.transform(df_test[FEATURES].values)

    discomfort_mask = clf_lr.predict(Xt) == 1
    heating_mask    = df_test["Heating"].values == 1
    cooling_mask    = ~heating_mask

    sigma_clo = scaler.scale_[clo_idx]
    sigma_tmp = scaler.scale_[tmp_idx]

    def restoration_rate(mask, action_fn):
        subset = np.where(mask & discomfort_mask)[0]
        if len(subset) == 0:
            return np.nan
        fixed = sum(
            1 for i in subset
            if clf_lr.predict(action_fn(Xt[i].copy()).reshape(1, -1))[0] == 0
        )
        return round(fixed / len(subset) * 100, 1)

    def add_clo(x):   x[clo_idx] += 0.3 / sigma_clo; return x
    def close_win(x): x[win_idx]  = 0.0;              return x
    def inc_tmp(x):   x[tmp_idx] += 1.0 / sigma_tmp;  return x
    def open_win(x):  x[win_idx]  = 1.0;              return x
    def dec_tmp(x):   x[tmp_idx] -= 1.0 / sigma_tmp;  return x
    def rem_clo(x):   x[clo_idx] -= 0.3 / sigma_clo;  return x

    results = {
        "heating": {
            "+0.3 clo alone":     restoration_rate(heating_mask, add_clo),
            "Close window alone": restoration_rate(heating_mask, close_win),
            "+1°C setpoint":      restoration_rate(heating_mask, inc_tmp),
        },
        "cooling": {
            "Open window alone":  restoration_rate(cooling_mask, open_win),
            "-1°C setpoint":      restoration_rate(cooling_mask, dec_tmp),
            "-0.3 clo alone":     restoration_rate(cooling_mask, rem_clo),
        },
    }

    print("\n── Prescriptive analytics: single-action restoration rates ──")
    print(f"{'Action':<25} {'Heating [%]':>12} {'Cooling [%]':>12}")
    for action in set(list(results["heating"]) + list(results["cooling"])):
        h = results["heating"].get(action, "—")
        c = results["cooling"].get(action, "—")
        print(f"  {action:<23} {str(h):>12} {str(c):>12}")

    return results


def plot_all(df, res_df, fitted_models, scaler, metrics_df, df_test,
             df_cands, raw):
    figs = []

    Xt_test = scaler.transform(df_test[FEATURES])
    yt_test = df_test["Comfort"].values

    # Fig 1: Dataset overview
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.hist(df["PMV"].clip(-3, 3), bins=40,
            color=PALETTE[0], edgecolor="white", alpha=0.9)
    ax.axvspan(-0.5, 0.5, color=PALETTE[2], alpha=0.15,
               label="Comfort zone [−0.5, 0.5]")
    ax.set_xlabel("PMV index"); ax.set_ylabel("Count")
    ax.set_title("PMV Distribution"); ax.legend(fontsize=8)

    ax = axes[1]
    comfort_pct = df.groupby("Heating")["Comfort"].mean() * 100
    bars = ax.bar(["Summer/Spring", "Winter/Autumn"], comfort_pct.values,
                  color=[PALETTE[3], PALETTE[0]], edgecolor="white")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{b.get_height():.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Discomfort Rate [%]")
    ax.set_title("Discomfort by Season")
    ax.set_ylim(0, max(comfort_pct.values) * 1.25)

    ax = axes[2]
    corr_vals = [spearmanr(df[f], df["PMV"])[0] for f in FEATURES]
    colors    = [PALETTE[0] if v > 0 else PALETTE[1] for v in corr_vals]
    ax.barh([FEATURE_LABELS.get(f, f) for f in FEATURES],
            corr_vals, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ with PMV")
    ax.set_title("Feature–PMV Correlation")

    fig.suptitle("Dataset Characterisation — UK Schools",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    figs.append(("fig1_dataset_overview", fig))

    fig, axes = plt.subplots(1, len(fitted_models), figsize=(14, 4))
    for ax, (name, clf) in zip(axes, fitted_models.items()):
        cm   = confusion_matrix(yt_test, clf.predict(Xt_test))
        disp = ConfusionMatrixDisplay(cm,
                                      display_labels=["Comfort", "Discomfort"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontweight="bold")
    fig.suptitle("Confusion Matrices — Test Set", fontweight="bold", y=1.02)
    fig.tight_layout()
    figs.append(("fig2_confusion", fig))

    # Fig 3: HAF by model × solver × case
    plot_df   = res_df[res_df["Model"] != "PMV Feedback"].copy()
    pmv_haf   = res_df[res_df["Model"] == "PMV Feedback"]["HAF [%]"].iloc[0]
    model_ord = list(fitted_models.keys())
    case_ord  = list(CASES.keys())
    x         = np.arange(len(case_ord))
    width     = 0.13
    hatches   = ["", "///"]

    fig, ax = plt.subplots(figsize=(12, 5))
    offsets = np.linspace(-0.32, 0.32, len(model_ord) * 2)
    idx = 0
    for mi, model_name in enumerate(model_ord):
        for si, solver in enumerate(METHODS):
            sub  = plot_df[(plot_df["Model"] == model_name) &
                           (plot_df["Solver"] == solver)]
            vals = [sub[sub["Case"] == c]["HAF [%]"].values[0]
                    if len(sub[sub["Case"] == c]) else np.nan
                    for c in case_ord]
            ax.bar(x + offsets[idx], vals, width,
                   label=f"{model_name[:2]}/{solver}",
                   color=PALETTE[mi], alpha=0.6 + 0.15 * si,
                   hatch=hatches[si], edgecolor="white")
            idx += 1

    ax.axhline(pmv_haf, color="red", linestyle="--", linewidth=1.5,
               label=f"PMV baseline HAF = {pmv_haf:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(case_ord, fontsize=9)
    ax.set_ylabel("Heating Adjustment Fraction [%]")
    ax.set_title("HAF by Model × Solver × Case — lower = more balanced demand",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    figs.append(("fig3_haf", fig))

    clf_best   = fitted_models["MLP"]
    sample_idx = df_cands.index[:min(30, len(df_cands))]
    df_sample  = df_cands.loc[sample_idx]

    fig, axes = plt.subplots(1, len(CASES), figsize=(14, 4), sharey=True)
    for ax, (case_name, opt_feats) in zip(axes, CASES.items()):
        temps_new = []
        for _, row in df_sample.iterrows():
            x0  = row[FEATURES].values.astype(float)
            opt = ThermalComfortOptimiser(
                clf_best, scaler, FEATURES,
                heating=bool(row["Heating"]),
                opt_features=opt_feats)
            x_new, _, _ = opt.optimise(x0, method="DE")
            temps_new.append(x_new[FEATURES.index("Temperature_Air")])

        ax.hist(df_sample["Temperature_Air"], bins=15, alpha=0.5,
                label="Before", color=PALETTE[3], edgecolor="white")
        ax.hist(temps_new, bins=15, alpha=0.7,
                label="After (DE)", color=PALETTE[0], edgecolor="white")
        ax.set_title(case_name, fontweight="bold")
        ax.set_xlabel("Temperature [°C]")
        if ax is axes[0]:
            ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Optimised Temperature Distributions — MLP + DE\n"
        "(30-instance visual sample; all metrics computed on full Nopt)",
        fontweight="bold", y=1.03)
    fig.tight_layout()
    figs.append(("fig4_temp_dist", fig))

    fig, ax = plt.subplots(figsize=(12, 5))
    case_colors = {c: PALETTE[i] for i, c in enumerate(case_ord)}
    x2 = np.arange(len(model_ord))
    w2 = 0.13
    for ci, case in enumerate(case_ord):
        for si, solver in enumerate(METHODS):
            vals = []
            for model in model_ord:
                row = plot_df[(plot_df["Model"] == model) &
                              (plot_df["Case"] == case) &
                              (plot_df["Solver"] == solver)]
                vals.append(
                    float(row["SuccessN"].iloc[0]) / 330 * 100
                    if len(row) else np.nan)
            offset = (ci * 2 + si - 2.5) * w2
            ax.bar(x2 + offset, vals, w2,
                   color=case_colors[case],
                   alpha=0.6 + 0.2 * si,
                   hatch=hatches[si],
                   edgecolor="white",
                   label=f"{case[:4]}/{solver}")

    ax.set_xticks(x2)
    ax.set_xticklabels(model_ord, fontsize=9)
    ax.set_ylabel("Comfort Success Rate [%]")
    ax.set_ylim(0, 110)
    ax.set_title("Comfort Restoration Success Rate", fontweight="bold")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    figs.append(("fig5_success_rate", fig))

    return figs


def main():
    print("=" * 70)
    print("  Inverse Thermal Comfort Optimisation — Reproducible Pipeline")
    print("=" * 70)

    print("\n[0] Checking dataset files ...")
    for path, label in [(PART1_PATH, "Part-1"), (PART2_PATH, "Part-2")]:
        if not os.path.exists(path):
            print(f"\n[!] '{path}' not found.")
            print("    See README for dataset download instructions.")
            return
        print(f"    Found: {path}")

    print("\n[1] Loading and preparing data ...")
    df = load_and_prepare()

    X = df[FEATURES].values
    y = df["Comfort"].values

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(df)),
        test_size=0.30, random_state=RANDOM_SEED, stratify=y)

    df_test  = df.iloc[idx_te].reset_index(drop=True)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)
    print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")

    print("\n[2] Training classifiers (10-fold CV) ...")
    fitted_models, metrics_df = train_and_evaluate(X_tr_sc, X_te_sc, y_tr, y_te)
    print("\n  Classification metrics:")
    print(metrics_df.to_string(index=False))

    gb_clf   = fitted_models["Gradient Boosting"]
    y_pred   = gb_clf.predict(X_te_sc)
    cand_idx = np.where(y_pred == 1)[0]
    df_cands = df_test.iloc[cand_idx].reset_index(drop=True)
    N_opt    = len(df_cands)
    print(f"\n[3] Optimisation candidates (surrogate-predicted discomfort): {N_opt}")

    print("\n[4] Running optimisation (all model × case × solver) ...")
    print("    This may take several minutes.")
    res_df, raw = run_optimisation(df_cands, fitted_models, scaler)

    display_cols = ["Model", "Case", "Solver",
                    "ΣΔT_W", "ΣΔT_S", "MATAW", "MATAS", "HAF [%]", "Success"]
    print("\n  Optimisation results:")
    print(res_df[display_cols].to_string(index=False))

    print("\n[5] Running statistical tests (two-proportion z-test) ...")
    stat_df = run_statistical_tests(res_df, N_opt)
    print(stat_df.to_string(index=False))

    print("\n[6] Prescriptive analytics ...")
    presc = prescriptive_analysis(df_test, scaler, fitted_models)

    print("\n[7] Generating figures ...")
    figs = plot_all(df, res_df, fitted_models, scaler, metrics_df,
                    df_test, df_cands, raw)
    for name, fig in figs:
        path = f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
    plt.close("all")

    metrics_df.to_csv("table_classification_metrics.csv", index=False)
    res_df[display_cols].to_csv("table_optimisation_results.csv", index=False)
    stat_df.to_csv("table_statistical_tests.csv", index=False)
    print("\n[Done] All outputs saved.")
    print("=" * 70)

    return df, fitted_models, scaler, res_df, metrics_df, stat_df, presc


if __name__ == "__main__":
    main()