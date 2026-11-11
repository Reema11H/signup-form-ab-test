"""
Signup form A/B test: does shortening the signup form raise completion rate?
Course: A/B Testing and Causal Inference (SDAIA Academy)

Run:  python main.py
Outputs: data/dataset.csv, outputs/results/*.csv|md, outputs/charts/*.png
All data is SIMULATED with a known true effect (see CONFIG).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHARTS_DIR = ROOT / "outputs" / "charts"
RESULTS_DIR = ROOT / "outputs" / "results"
for d in (DATA_DIR, CHARTS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TEAL, ORANGE, NAVY, GRAY = "#0E7490", "#E58A1F", "#0F2A3F", "#A9B4BC"

# ---------------------------------------------------------------- 3. Design + simulation truth
CONFIG = {
    "project": "Effect of a shorter signup form (8 -> 3 fields) on signup completion",
    # design (known before the experiment)
    "baseline_conversion": 0.40,   # current completion rate among visitors who see the form
    "mde_abs": 0.02,               # smallest effect that changes the decision: 2 points
    "alpha": 0.05,
    "power": 0.80,
    "daily_new_users": 1_500,
    "treatment_share": 0.5,
    # simulation truth (hidden in real life)
    "true_lift_abs": 0.025,
    "activity_strength": 0.8,
    "novelty_boost": 0.0,
    "time_mean": 150, "time_effect": 60,
    "verify_rate": 0.85, "verify_effect": 0.0,
    "active_rate": 0.60, "active_effect": 0.0,
    "fake_rate": 0.03, "fake_effect": 0.0,
    "srm_bug_share": 0.0,
    "seed": 2026,
}


# ---------------------------------------------------------------- helpers
def n_per_group_proportions(p_control, p_treatment, alpha=0.05, power=0.80):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    var = p_control * (1 - p_control) + p_treatment * (1 - p_treatment)
    return int(np.ceil((z_a + z_b) ** 2 * var / (p_treatment - p_control) ** 2))


def mde_proportions(p_control, n_per_group, alpha=0.05, power=0.80):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (z_a + z_b) * np.sqrt(2 * p_control * (1 - p_control) / n_per_group)


def duration_days(n_per_group, daily_new_users, groups=2, round_to_weeks=True):
    days = int(np.ceil(n_per_group * groups / daily_new_users))
    return int(np.ceil(days / 7) * 7) if round_to_weeks else days


def srm_check(n_a, n_b, expected_share_b=0.5, threshold=0.001):
    total = n_a + n_b
    chi2, p = stats.chisquare([n_a, n_b], f_exp=[total * (1 - expected_share_b), total * expected_share_b])
    return chi2, p, p >= threshold


def diff_ci(a, b, alpha=0.05):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"diff": diff, "ci_low": diff - z * se, "ci_high": diff + z * se, "se": se,
            "p": 2 * stats.norm.sf(abs(diff / se))}


def benjamini_hochberg(pvals, alpha=0.05):
    """Manual BH: returns (reject, adjusted p)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj <= alpha, adj


def decide(ci_low, ci_high, mde, guardrails_ok=True):
    if ci_high < 0:
        return "Do not launch: the new form hurt the primary metric."
    if ci_low > 0 and not guardrails_ok:
        return "Do not launch yet: primary metric improved but a guardrail got worse. Fix and re-verify."
    if ci_low > 0:
        return "Launch: the effect is positive with confidence and guardrails are healthy."
    if ci_high < mde:
        return "Do not launch: we ruled out an effect of the size that matters."
    return "Inconclusive: extend the experiment or redesign it with more power."


# ---------------------------------------------------------------- 5. Simulation
def simulate_experiment(cfg, n_days):
    r = np.random.default_rng(cfg["seed"])
    n = n_days * cfg["daily_new_users"]
    is_b = (r.random(n) < cfg["treatment_share"]).astype(int)
    day = r.integers(1, n_days + 1, n)
    intent = r.normal(0, 1, n)                                   # visitor intent (unobserved)
    base = cfg["baseline_conversion"]
    p0 = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + cfg["activity_strength"] * intent)))
    p0 = np.clip(p0 * base / p0.mean(), 0, 1)
    pre_visits_30d = r.poisson(3 * p0 / base)                    # pre-experiment covariate (for CUPED)
    lift = cfg["true_lift_abs"] * (1 + cfg["novelty_boost"] * np.exp(-(day - 1) / 3))
    completed = r.binomial(1, np.clip(p0 + lift * is_b, 0, 1))
    done = completed == 1

    def among_completers(rate, effect):
        return np.where(done, r.binomial(1, np.clip(rate + effect * is_b, 0, 1)), np.nan)

    form_seconds = np.where(done, np.maximum(20, r.normal(cfg["time_mean"] - cfg["time_effect"] * is_b, 40)), np.nan)
    email_verified = among_completers(cfg["verify_rate"], cfg["verify_effect"])
    active_7d = among_completers(cfg["active_rate"], cfg["active_effect"])
    fake_account = among_completers(cfg["fake_rate"], cfg["fake_effect"])
    df = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "variant": np.where(is_b == 1, "B", "A"), "day": day, "pre_visits_30d": pre_visits_30d,
        "completed": completed, "form_seconds": form_seconds, "email_verified": email_verified,
        "active_7d": active_7d, "fake_account": fake_account,
        "active_per_visitor": np.nan_to_num(active_7d)})
    if cfg["srm_bug_share"] > 0:
        lost = (df["variant"] == "B") & (intent < 0) & (r.random(n) < cfg["srm_bug_share"] * 2)
        df = df[~lost]
    return df.reset_index(drop=True)


def main():
    c = CONFIG
    lines = []                                                   # markdown report lines

    def log(text=""):
        print(text)
        lines.append(text)

    # ---- 3. sample size and duration
    n_per_group = n_per_group_proportions(c["baseline_conversion"], c["baseline_conversion"] + c["mde_abs"],
                                          c["alpha"], c["power"])
    days = duration_days(n_per_group, c["daily_new_users"])
    n_planned = days * c["daily_new_users"] // 2
    mde_full = mde_proportions(c["baseline_conversion"], n_planned, c["alpha"], c["power"])
    mde_week = mde_proportions(c["baseline_conversion"], 7 * c["daily_new_users"] // 2, c["alpha"], c["power"])
    log(f"# {c['project']}")
    log("\n## Sample size and duration")
    log(f"- Required sample: {n_per_group:,} per group ({2 * n_per_group:,} total) "
        f"[alpha={c['alpha']}, power={c['power']}, baseline={c['baseline_conversion']:.0%}, MDE={c['mde_abs'] * 100:.1f} pts]")
    log(f"- Duration: {days} days (rounded to full weeks), {n_planned:,} users per group expected")
    log(f"- Smallest detectable effect: {mde_full * 100:.2f} pts over {days} days, {mde_week * 100:.2f} pts over one week")

    # ---- 5. data
    data = simulate_experiment(c, days)
    data.to_csv(DATA_DIR / "dataset.csv", index=False)
    log("\n## Data quality checks")
    log(f"- Simulated users: {len(data):,} (seed={c['seed']}, true lift = {c['true_lift_abs'] * 100:.1f} pts)")
    log(f"- Duplicate user ids: {int(data['user_id'].duplicated().sum())}")
    log(f"- Missing values in primary metric/variant: {int(data[['completed', 'variant']].isna().sum().sum())}")

    # ---- 6. SRM
    counts = data["variant"].value_counts()
    chi2, p_srm, srm_ok = srm_check(counts.get("A", 0), counts.get("B", 0), c["treatment_share"])
    log(f"- SRM check: A={counts.get('A', 0):,}, B={counts.get('B', 0):,}, chi2={chi2:.2f}, p={p_srm:.3g} -> "
        f"{'healthy' if srm_ok else 'SRM DETECTED, do not trust results'}")

    # ---- 7.1 primary + CUPED
    A, B = data[data.variant == "A"], data[data.variant == "B"]
    primary = diff_ci(A["completed"], B["completed"])
    theta = np.cov(data["completed"], data["pre_visits_30d"])[0, 1] / data["pre_visits_30d"].var(ddof=1)
    adjusted = data["completed"] - theta * (data["pre_visits_30d"] - data["pre_visits_30d"].mean())
    primary_cuped = diff_ci(adjusted[data.variant == "A"], adjusted[data.variant == "B"])
    reduction = 1 - (primary_cuped["se"] / primary["se"]) ** 2
    log("\n## Primary metric: signup completion rate")
    log(f"- A: {A['completed'].mean():.2%}   B: {B['completed'].mean():.2%}")
    log(f"- Effect: {primary['diff'] * 100:.2f} pts, 95% CI [{primary['ci_low'] * 100:.2f}, {primary['ci_high'] * 100:.2f}], "
        f"p={primary['p']:.2g}")
    log(f"- With CUPED: {primary_cuped['diff'] * 100:.2f} pts, 95% CI "
        f"[{primary_cuped['ci_low'] * 100:.2f}, {primary_cuped['ci_high'] * 100:.2f}] (variance reduction {reduction:.1%})")

    # ---- 7.2 secondary with BH
    secondary = {
        "Form time in seconds (completers)": diff_ci(A["form_seconds"].dropna(), B["form_seconds"].dropna()),
        "Active accounts per visitor": diff_ci(A["active_per_visitor"], B["active_per_visitor"]),
    }
    sec = pd.DataFrame(secondary).T
    sec["significant_after_BH"], sec["p_adjusted"] = benjamini_hochberg(sec["p"].astype(float))
    sec.round(4).to_csv(RESULTS_DIR / "secondary_metrics.csv")
    log("\n## Secondary metrics (Benjamini-Hochberg corrected)")
    for name, row in sec.iterrows():
        log(f"- {name}: diff={row['diff']:.3f}, CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
            f"p_adj={row['p_adjusted']:.3g}, significant={bool(row['significant_after_BH'])}")

    # ---- 7.3 guardrails
    guardrails = {"email_verified": ("Email verification rate", "down"),
                  "active_7d": ("Active after 7 days rate", "down"),
                  "fake_account": ("Fake/duplicate account rate", "up")}
    rows, guard_status = [], {}
    for col, (label, bad) in guardrails.items():
        res = diff_ci(A[col].dropna(), B[col].dropna())
        harmful = res["ci_high"] < 0 if bad == "down" else res["ci_low"] > 0
        guard_status[label] = "significant deterioration" if harmful else "healthy"
        rows.append({"metric": label, "A": A[col].mean(), "B": B[col].mean(), **res, "status": guard_status[label]})
    guard_table = pd.DataFrame(rows).set_index("metric")
    guard_table.round(4).to_csv(RESULTS_DIR / "guardrail_metrics.csv")
    log("\n## Guardrail metrics")
    for name, row in guard_table.iterrows():
        log(f"- {name}: A={row['A']:.3f}, B={row['B']:.3f}, diff={row['diff']:.4f}, "
            f"CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}] -> {row['status']}")

    # ---- decision
    guardrails_ok = all(s == "healthy" for s in guard_status.values())
    decision = decide(primary["ci_low"], primary["ci_high"], c["mde_abs"], guardrails_ok)
    annual_visitors = c["daily_new_users"] * 365
    extra_active = sec.loc["Active accounts per visitor"]
    log("\n## Recommendation")
    log(f"- Decision: {decision}")
    log(f"- Estimated extra active accounts per year: {extra_active['diff'] * annual_visitors:,.0f} "
        f"(CI {extra_active['ci_low'] * annual_visitors:,.0f} to {extra_active['ci_high'] * annual_visitors:,.0f})")
    if primary["ci_low"] < c["mde_abs"]:
        log(f"- Note: the lower CI bound ({primary['ci_low'] * 100:.2f} pts) is below the {c['mde_abs'] * 100:.0f}-point "
            "threshold, so the true effect may be smaller than what matters commercially.")
    log("- Limits: simulated data; novelty effects and multi-device users are not modeled in the real world.")

    pd.DataFrame([{"estimator": "raw", **primary}, {"estimator": "CUPED", **primary_cuped}]) \
        .round(5).to_csv(RESULTS_DIR / "primary_metric.csv", index=False)
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- charts
    fig, ax = plt.subplots(figsize=(7, 4))
    rates = [A["completed"].mean() * 100, B["completed"].mean() * 100]
    ax.bar(["A (8 fields)", "B (3 fields)"], rates, color=[TEAL, ORANGE])
    for i, v in enumerate(rates):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", color=NAVY)
    ax.set_ylabel("Signup completion rate (%)")
    ax.set_title("Completion rate by variant")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "completion_by_variant.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, (label, res) in enumerate([("Raw", primary), ("CUPED", primary_cuped)]):
        ax.errorbar(res["diff"] * 100, i, xerr=[[(res["diff"] - res["ci_low"]) * 100], [(res["ci_high"] - res["diff"]) * 100]],
                    fmt="o", color=TEAL, capsize=5)
    ax.axvline(0, color=GRAY, linestyle="--")
    ax.axvline(c["mde_abs"] * 100, color=ORANGE, linestyle=":", label="Smallest effect that matters")
    ax.axvline(c["true_lift_abs"] * 100, color=NAVY, linestyle="-", linewidth=0.8, label="True effect (simulation)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Raw", "CUPED"])
    ax.set_xlabel("Effect (percentage points), 95% CI")
    ax.set_title("Estimated effect on completion")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "effect_with_ci.png", dpi=130)
    plt.close(fig)

    daily = data.groupby(["day", "variant"])["completed"].mean().unstack()
    daily_lift = (daily["B"] - daily["A"]) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(daily_lift.index, daily_lift.values, marker="o", color=TEAL, label="Daily effect")
    ax.axhline(primary["diff"] * 100, color=ORANGE, linestyle="--", label="Overall effect")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("Day of entry")
    ax.set_ylabel("Effect (percentage points)")
    ax.set_title("Novelty check: is the effect stable over time?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "novelty_check.png", dpi=130)
    plt.close(fig)

    print(f"\nSaved data to {DATA_DIR}, results to {RESULTS_DIR}, charts to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
