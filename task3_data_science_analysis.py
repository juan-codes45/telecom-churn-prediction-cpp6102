"""
task3_data_science_analysis.py  --  TASK 3: Data Science Analysis (Matplotlib)
One function per figure; every function saves a PNG into outputs/ and returns
the numbers it plotted so they can be quoted in the report.
"""
import matplotlib
matplotlib.use("Agg")  # saves files; delete this line to pop up windows instead
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from src import config

BLUE, RED = "#1f4e79", "#c0504d"
plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _rate(df, col, order=None):
    s = df.groupby(col)[config.TARGET_FLAG].mean() * 100
    return s.reindex(order) if order else s


def _save(fig, name):
    path = config.OUTPUT_DIR / name
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return path


def fig_contract(df):
    s = _rate(df, "Contract", ["Month-to-month", "One year", "Two year"])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(s.index, s.values, color=[RED, BLUE, BLUE]); ax.bar_label(bars, fmt="%.1f%%")
    ax.axhline(df[config.TARGET_FLAG].mean() * 100, ls="--", c="grey", lw=1)
    ax.set_ylabel("Churn rate (%)"); ax.set_xlabel("Contract type")
    _save(fig, "fig1_contract.png"); return s.round(1).to_dict()


def fig_tenure(df):
    s = _rate(df, "tenure_band", ["0-12", "13-24", "25-48", "49-72"])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(s.index, s.values, color=BLUE); ax.bar_label(bars, fmt="%.1f%%")
    ax.set_ylabel("Churn rate (%)"); ax.set_xlabel("Tenure band (months)")
    _save(fig, "fig2_tenure.png"); return s.round(1).to_dict()


def fig_charges(df):
    kept = df.loc[df.Churn == "No", "MonthlyCharges"]
    lost = df.loc[df.Churn == "Yes", "MonthlyCharges"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bp = ax.boxplot([kept, lost], tick_labels=["Retained", "Churned"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [BLUE, RED]): patch.set_facecolor(c); patch.set_alpha(0.7)
    for m in bp["medians"]: m.set_color("black")
    ax.set_ylabel("Monthly charges (USD)")
    _save(fig, "fig3_charges.png")
    return {"retained_median": float(kept.median()), "churned_median": float(lost.median())}


def fig_service_load(df):
    s = _rate(df, "service_load")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(s.index, s.values, marker="o", c=BLUE)
    ax.set_xlabel("Service load (number of subscribed services)"); ax.set_ylabel("Churn rate (%)")
    _save(fig, "fig4_serviceload.png"); return s.round(1).to_dict()


def fig_roc(fitted, y_te, metrics):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (name, (_, proba)), col in zip(fitted.items(), [BLUE, RED]):
        fpr, tpr, _ = roc_curve(y_te, proba)
        ax.plot(fpr, tpr, c=col, label=f"{name} (AUC = {metrics[name]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(frameon=False)
    _save(fig, "fig5_roc.png")


def fig_features(importance, title_note):
    top = importance.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(top.index, top.values, color=[RED if v > 0 else BLUE for v in top.values])
    ax.set_xlabel(title_note)
    _save(fig, "fig6_features.png")