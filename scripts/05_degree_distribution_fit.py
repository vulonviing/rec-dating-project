"""Fit power-law and log-normal models to the profile in-degree distribution
of the bipartite rating network, and produce a CCDF figure with the fit."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DATA_DIR = PROJECT_ROOT / "outputs" / "data"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    profile_metrics = pd.read_csv(DATA_DIR / "profile_metrics_full.csv")
    in_degree = profile_metrics["in_degree"].to_numpy(dtype=np.float64)
    in_degree = in_degree[in_degree > 0]

    # Fit using the powerlaw package (Clauset et al. 2009 method)
    fit = powerlaw.Fit(in_degree, discrete=True, verbose=False)

    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    sigma = fit.power_law.sigma

    # Compare power-law vs log-normal
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential")

    summary = {
        "variable": "profile_in_degree",
        "n_observations": int(in_degree.size),
        "power_law_alpha": round(float(alpha), 4),
        "power_law_xmin": float(xmin),
        "power_law_sigma": round(float(sigma), 4),
        "n_in_tail": int((in_degree >= xmin).sum()),
        "vs_lognormal_R": round(float(R_ln), 4),
        "vs_lognormal_p": round(float(p_ln), 6),
        "vs_exponential_R": round(float(R_exp), 4),
        "vs_exponential_p": round(float(p_exp), 6),
    }

    (DATA_DIR / "degree_distribution_fit_full.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # ── Figure: CCDF with power-law fit ──
    fig, ax = plt.subplots(figsize=(7, 5.2))

    # Empirical CCDF
    sorted_deg = np.sort(in_degree)[::-1]
    ccdf_y = np.arange(1, len(sorted_deg) + 1) / len(sorted_deg)
    ax.plot(sorted_deg, ccdf_y, ".", color="#1d3557", markersize=2.5, alpha=0.5,
            label="Empirical CCDF")

    # Fitted power-law line over the tail
    tail = sorted_deg[sorted_deg >= xmin]
    x_fit = np.logspace(np.log10(xmin), np.log10(tail.max()), 200)
    ccdf_fit = (x_fit / xmin) ** (-(alpha - 1))
    ccdf_fit *= (in_degree >= xmin).sum() / len(in_degree)
    ax.plot(x_fit, ccdf_fit, "-", color="#e63946", linewidth=2.2,
            label=rf"Power-law fit ($\alpha={alpha:.2f}$, $x_{{\min}}={int(xmin)}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Profile Node In-Degree (bipartite network)")
    ax.set_ylabel("CCDF  $P(k \\geq x)$")
    ax.set_title("In-Degree Distribution of Profile Nodes\nin the Bipartite Rating Network")
    ax.legend(frameon=True, loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "degree_distribution_fit_full.png", dpi=220)
    plt.close(fig)

    print("Degree-distribution fit summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
