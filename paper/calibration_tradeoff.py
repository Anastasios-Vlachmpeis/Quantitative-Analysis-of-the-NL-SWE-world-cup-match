# Does totals calibration trade off against 1X2 discrimination across models?
#
# The decisive test for a new paper. Two data points (dixon_coles vs its own
# modified variant) cannot distinguish a structural tradeoff from "the 2nd model is just worse".
# Here we fit five INDEPENDENTLY specified scoreline models under identical treatment
# and ask whether better goal-total calibration systematically costs 1X2 skill.
#
# Design note: we deliberately do not reuse the persisted walk-forward predictions.
# I produced those runs weeks apart with refit schedules that the manifests do
# not record, so cross-model differences would be confounded by fit frequency.
# Instead ~ one fit per model on a fixed training window, one shared holdout.

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nlvswe.config import get_config, load_config
from nlvswe.eval.scoring import outcome_codes, vectorized_per_match_scores
from nlvswe.io import save_figure
from nlvswe.logging import get_logger
from nlvswe.models.data import load_model_features
from nlvswe.models.registry import create_model
from nlvswe.plotting.theme import THEME, apply_theme, style_axes, style_figure
from nlvswe.repro import set_seeds

logger = get_logger(__name__)

MODELS = ["poisson", "dixon_coles", "dixon_coles_v2", "bivariate_poisson", "bayesian"]
TOTALS_LINES = (1.5, 2.5, 3.5)
SPLIT = "2022-01-01"
N_BOOT = 300


def prob_over(matrix: np.ndarray, line: float) -> float:
    # P(home + away goals > line) from a scoreline probability matrix
    g = matrix.shape[0]
    i = np.arange(g)[:, None]
    j = np.arange(g)[None, :]
    return float(matrix[(i + j) > line].sum())


def expected_total(matrix: np.ndarray) -> float:
    g = matrix.shape[0]
    i = np.arange(g)[:, None]
    j = np.arange(g)[None, :]
    return float((matrix * (i + j)).sum())


def ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    #Expected calibration error: mean |predicted - observed| weighted by bin count
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = idx == b
        k = int(mask.sum())
        if k == 0:
            continue
        total += (k / n) * abs(float(probs[mask].mean()) - float(outcomes[mask].mean()))
    return total


def fit_and_predict(name: str, train: pd.DataFrame, test: pd.DataFrame, cfg) -> dict:
    #One fit on train, then per-match scoreline + 1X2 probabilities on test
    model = create_model(name, cfg, pd.DataFrame())
    logger.info("[%s] fitting on %d matches", name, len(train))
    model.fit(train)

    probs_1x2 = np.zeros((len(test), 3), dtype=float)
    over = {line: np.zeros(len(test), dtype=float) for line in TOTALS_LINES}
    exp_tot = np.zeros(len(test), dtype=float)

    logger.info("[%s] predicting %d holdout matches", name, len(test))
    for row_i, (_, match) in enumerate(test.iterrows()):
        pred = model.predict(match)
        probs_1x2[row_i] = [pred.probs_1x2["home"], pred.probs_1x2["draw"], pred.probs_1x2["away"]]
        mat = np.asarray(pred.scoreline, dtype=float)
        for line in TOTALS_LINES:
            over[line][row_i] = prob_over(mat, line)
        exp_tot[row_i] = expected_total(mat)

    return {"probs_1x2": probs_1x2, "over": over, "exp_total": exp_tot}


def summarise(name: str, out: dict, test: pd.DataFrame, rng: np.random.Generator) -> dict:
    #Point estimates plus bootstrap CIs for RPS and totals ECE
    codes = outcome_codes(test["outcome_1x2"])
    rps_per_match = vectorized_per_match_scores(out["probs_1x2"], codes, "rps")
    totals = test["total_goals"].to_numpy(dtype=float)

    row: dict = {"model": name, "rps": float(rps_per_match.mean())}
    realised = {line: (totals > line).astype(float) for line in TOTALS_LINES}
    for line in TOTALS_LINES:
        row[f"ece_{line}"] = ece(out["over"][line], realised[line])
    row["mean_pred_total"] = float(out["exp_total"].mean())
    row["mean_actual_total"] = float(totals.mean())

    n = len(test)
    boot_rps, boot_ece = [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        boot_rps.append(float(rps_per_match[pick].mean()))
        boot_ece.append(ece(out["over"][2.5][pick], realised[2.5][pick]))
    row["rps_lo"], row["rps_hi"] = np.quantile(boot_rps, [0.025, 0.975])
    row["ece_2.5_lo"], row["ece_2.5_hi"] = np.quantile(boot_ece, [0.025, 0.975])
    return row


def plot_tradeoff(table: pd.DataFrame) -> None:
    #Totals calibration error against 1X2 skill. A tradeoff slopes downward.
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 6))

    x = table["ece_2.5"].to_numpy()
    y = table["rps"].to_numpy()
    ax.errorbar(
        x,
        y,
        xerr=[x - table["ece_2.5_lo"], table["ece_2.5_hi"] - x],
        yerr=[y - table["rps_lo"], table["rps_hi"] - y],
        fmt="o",
        color=THEME.alpha,
        ecolor=THEME.grid,
        markersize=9,
        capsize=3,
        linestyle="none",
    )
    for _, r in table.iterrows():
        ax.annotate(
            r["model"],
            xy=(r["ece_2.5"], r["rps"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color=THEME.text,
        )

    if len(table) >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, "--", color=THEME.bearish, linewidth=1, alpha=0.7)

    ax.set_xlabel("Totals calibration error (over 2.5 ECE, lower is better)")
    ax.set_ylabel("1X2 Ranked Probability Score (lower is better)")
    ax.set_title("Goal-total calibration vs match-outcome skill")
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, "calibration_tradeoff")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Totals-calibration vs 1X2-skill tradeoff test")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--split", default=SPLIT)
    args = parser.parse_args(argv)

    get_config.cache_clear()
    cfg = load_config()
    set_seeds(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    feats = load_model_features()
    played = feats[feats["result_1x2"].notna()].copy()
    played["outcome_1x2"] = played["result_1x2"]
    played["total_goals"] = played["home_goals"].astype(float) + played["away_goals"].astype(float)

    train = played[played["date_utc"] < args.split]
    test = played[played["date_utc"] >= args.split]
    logger.info("train=%d  holdout=%d  (split %s)", len(train), len(test), args.split)

    rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        out = fit_and_predict(name, train, test, cfg)
        rows.append(summarise(name, out, test, rng))

    table = pd.DataFrame(rows).sort_values("ece_2.5").reset_index(drop=True)
    table.to_csv("paper/calibration_tradeoff.csv", index=False)

    print("\n=== Holdout results (n=%d matches, split %s) ===" % (len(test), args.split))
    cols = ["model", "rps", "rps_lo", "rps_hi", "ece_2.5", "ece_2.5_lo", "ece_2.5_hi",
            "ece_1.5", "ece_3.5", "mean_pred_total", "mean_actual_total"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if len(table) >= 3:
        rho, p = spearmanr(table["ece_2.5"], table["rps"])
        print(f"\nSpearman(ECE_2.5, RPS) = {rho:+.3f}  p = {p:.3f}  (n={len(table)} models)")
        print("Negative rho => better totals calibration comes with worse 1X2 skill (a tradeoff).")

    plot_tradeoff(table)


if __name__ == "__main__":
    main()
