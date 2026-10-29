"""
task3_data_science_analysis.py  --  TASK 3: Data Science Analysis
Using pandas group-bys, visualise with Matplotlib and communicate to managers.
Every figure function saves a PNG into outputs/ and returns the numbers it
plotted. manager_summary() turns the results into a plain-language one-pager.
"""
import matplotlib
matplotlib.use("Agg")  # saves files; delete this line to pop up windows instead
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
from src import config

BLUE, RED = "#1f4e79", "#c0504d"
plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _rate(df, col, order=None):
    s = df.groupby(col)[config.TARGET_FLAG].mean() * 100
    return s.reindex(order) if order else s


def _title(ax, text):
    if config.FIGURE_TITLES:
        ax.set_title(text, fontsize=10, fontweight="bold", loc="left")


def _save(fig, name):
    path = config.OUTPUT_DIR / name
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return path


def fig_complaints(df):
    """Churn rate by complaint status and by account status."""
    s = pd.concat([_rate(df, "Complains").rename({0: "No complaint", 1: "Complaint"}),
                   _rate(df, "Status").rename({1: "Status 1 (active)", 2: "Status 2 (inactive)"})])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(s.index, s.values, color=[BLUE, RED, BLUE, RED]); ax.bar_label(bars, fmt="%.1f%%")
    ax.axhline(df[config.TARGET_FLAG].mean() * 100, ls="--", c="grey", lw=1)
    ax.set_ylabel("Churn rate (%)"); plt.setp(ax.get_xticklabels(), fontsize=8)
    _title(ax, f"Customers who complain churn at {s['Complaint']:.0f}% vs {s['No complaint']:.0f}%")
    _save(fig, "fig1_complaints.png"); return s.round(1).to_dict()


def fig_tenure(df):
    s = _rate(df, "tenure_band", ["0-24", "25-32", "33-38", "39-47"])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(s.index, s.values, color=BLUE); ax.bar_label(bars, fmt="%.1f%%")
    ax.set_ylabel("Churn rate (%)"); ax.set_xlabel("Subscription length band (months)")
    _title(ax, "Churn by subscription length")
    _save(fig, "fig2_tenure.png"); return s.round(1).to_dict()


def fig_value(df):
    kept = df.loc[df[config.TARGET_FLAG] == 0, "Customer_Value"]
    lost = df.loc[df[config.TARGET_FLAG] == 1, "Customer_Value"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bp = ax.boxplot([kept, lost], tick_labels=["Retained", "Churned"], patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, c in zip(bp["boxes"], [BLUE, RED]): patch.set_facecolor(c); patch.set_alpha(0.7)
    for m in bp["medians"]: m.set_color("black")
    ax.set_ylabel("Customer value")
    _title(ax, f"Churned customers are lower value (median {lost.median():.0f} vs {kept.median():.0f})")
    _save(fig, "fig3_value.png")
    return {"retained_median": round(float(kept.median()), 2), "churned_median": round(float(lost.median()), 2)}


def fig_call_volume(df):
    """Churn rate across call-volume quartiles (Frequency_of_use)."""
    bands = pd.qcut(df[config.CALL_VOLUME_COL], 4, duplicates="drop")
    s = df.groupby(bands, observed=True)[config.TARGET_FLAG].mean() * 100
    labels = [f"{int(i.left)}-{int(i.right)}" for i in s.index]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(labels, s.values, color=BLUE); ax.bar_label(bars, fmt="%.1f%%")
    ax.set_xlabel("Number of calls (quartile ranges)"); ax.set_ylabel("Churn rate (%)")
    _title(ax, "The less customers call, the more likely they leave")
    _save(fig, "fig4_callvolume.png"); return dict(zip(labels, s.round(1)))


def fig_roc(fitted, y_te, metrics):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (name, (_, proba)), col in zip(fitted.items(), [BLUE, RED]):
        fpr, tpr, _ = roc_curve(y_te, proba)
        ax.plot(fpr, tpr, c=col, label=f"{name} (AUC = {metrics[name]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(frameon=False)
    _title(ax, "Both models separate churners from non-churners well")
    _save(fig, "fig5_roc.png")


def fig_features(importance, title_note):
    top = importance.head(12)[::-1]
    signed = bool((importance < 0).any())   # logistic coefficients have signs; forest importances do not
    colours = [RED if v > 0 else BLUE for v in top.values] if signed else BLUE
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(top.index, top.values, color=colours)
    ax.set_xlabel(title_note)
    _title(ax, "What the model relies on most")
    _save(fig, "fig6_features.png")


def manager_summary(R: dict):
    """Plain-language one-pager for managers, built from the computed results."""
    seg = {s["segment"]: s for s in R["segments"]}
    p = R["profile"]
    complaint = seg["Filed a complaint"]
    q_labels = list(R["churn_by_call_volume"])           # call-volume quartile ranges
    quietest = (q_labels[0], R["churn_by_call_volume"][q_labels[0]])
    highcalls = seg["High call volume (top quartile)"]
    tiers = R["risk_tiers"]
    high = tiers.get("High", {"n": 0, "actual_churn_pct": 0.0})
    lift = R["lift"]
    text = f"""# Customer churn: summary for managers

**Headline:** {p['churn_rate_pct']:.1f}% of our {p['rows']:,} customers have churned ({p['churned']:,} people).
A model built from usage, billing and complaint data identifies who is likely to leave
with an accuracy score (AUC) of {R['metrics'][R['best_model']]['auc']:.2f} out of 1.00.

## What we found
1. **Complaints are the loudest warning.** Customers who filed a complaint churn at
   {complaint['churn_rate_pct']:.0f}% ({complaint['n']} customers), compared with {p['churn_rate_pct']:.0f}% overall.
2. **Low usage means high risk.** The quietest quarter of customers ({quietest[0]} calls) churn at {quietest[1]:.0f}%,
   while the most active quarter churn at only {highcalls['churn_rate_pct']:.1f}%.
3. **The model finds the right people.** The riskiest 10% of customers it flagged were
   {lift['top_decile_churn_pct']:.0f}% actual churners, {lift['lift']:.1f} times better than picking at random.

## What we recommend
1. **Resolve complaints fast.** Give every complaint a named owner and a deadline; this is where the most churn is concentrated.
2. **Re-engage quiet customers.** Offer a usage or bundle incentive to customers whose calls fall below the typical level.
3. **Target the retention budget.** Contact the "High" risk tier first: {high['n']} customers in our test group, of whom
   {high['actual_churn_pct']:.0f}% did leave. The full ranked list is in `outputs/predictions.csv`.

## Things to keep in mind
- Results show patterns, not proof of cause. A pilot offer should be tested before a full rollout.
- Complaints and account status may partly be effects of leaving rather than early warnings. Without them the model still scores AUC {R['sensitivity']['auc_reduced']:.2f}.
- The data is one snapshot in time and has no customer IDs; duplicates were removed ({p['duplicates_removed']} rows).
"""
    path = config.OUTPUT_DIR / "manager_summary.md"
    path.write_text(text, encoding="utf-8")
    return path