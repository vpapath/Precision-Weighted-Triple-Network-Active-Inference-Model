#!/usr/bin/env python3
"""
main.py
=======
Entry point for the Precision-Weighted Triple Network simulation.

Usage
-----
    python main.py                          # full run with defaults
    python main.py --n-runs 5 --t-learn 100  # faster run for testing

Outputs
-------
    outputs/fig1_architecture.png
    outputs/fig2_simulations.png
    outputs/fig3_precision.png
    outputs/fig4_statistics.png

References
----------
Papathanasiou, V. (2026). Precision-Weighted Network Dynamics and the
    Lacanian Subject: A Hierarchical Active Inference Model of the Triple Network.
Dall'Aglio, J. (2021a–c). Sex and Prediction Error, Parts 1–3.
    Journal of the American Psychoanalytic Association, 69(4), 693–765.
Adams, R.A., Stephan, K.E., Brown, H.R., Frith, C.D., & Friston, K.J. (2013).
    The computational anatomy of psychosis. Frontiers in Psychiatry, 4, 47.
Feldman, H., & Friston, K.J. (2010). Attention, uncertainty, and free-energy.
    Frontiers in Human Neuroscience, 4, 215.
Friston, K.J., Brown, H.R., Siemerkus, J., & Stephan, K.E. (2016).
    The dysconnection hypothesis (2016). Schizophrenia Research, 176, 83-94.
Heins, C. et al. (2022). pymdp: A Python library for active inference
    in discrete state spaces. JOSS, 7(73), 4098.
Li, L., & Li, C. (2025). Formalizing Lacanian psychoanalysis through the free
    energy principle. Frontiers in Psychology, 16, 1574650.
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triple_network.simulate import run_all, T_LEARN, T_SIM, N_RUNS
from triple_network.figures import generate_all


def parse_args():
    p = argparse.ArgumentParser(
        description="Precision-Weighted Triple Network Active Inference Model")
    p.add_argument("--n-runs",  type=int, default=N_RUNS,
                   help=f"Seeds per condition (default: {N_RUNS})")
    p.add_argument("--t-learn", type=int, default=T_LEARN,
                   help=f"Learning phase steps (default: {T_LEARN})")
    p.add_argument("--t-sim",   type=int, default=T_SIM,
                   help=f"Interaction phase steps (default: {T_SIM})")
    p.add_argument("--seed",    type=int, default=42,
                   help="Global random seed (default: 42)")
    p.add_argument("--output-dir", type=str, default="outputs",
                   help="Output directory for figures (default: outputs)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")
    return p.parse_args()


def print_summary(learning, interaction):
    print("\n" + "─" * 60)
    print("SUMMARY")
    print("─" * 60)
    conditions = ["baseline", "psychosis", "melancholia"]
    for cond in conditions:
        L = learning[cond]
        I = interaction[cond]
        from triple_network.model import N_STATES
        gm_l = L["gamma_trace"].mean(); gv_l = L["gamma_trace"].var()
        gm_i = I["gamma"].mean();       gv_i = I["gamma"].var()
        dm   = (I["dmn"] @ np.arange(N_STATES)).mean()
        cm   = (I["cen"] @ np.arange(N_STATES)).mean()
        dp   = I["dmn_prec"].mean();    cp = I["cen_prec"].mean()
        print(f"\n  {cond.upper()}")
        print(f"    Learning  γ: mean={gm_l:.3f}  var={gv_l:.5f}")
        print(f"    Interactn γ: mean={gm_i:.3f}  var={gv_i:.5f}")
        print(f"    DMN: act={dm:.3f}  prec={dp:.3f}")
        print(f"    CEN: act={cm:.3f}  prec={cp:.3f}")
    print("─" * 60)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Precision-Weighted Triple Network — Dynamic Active Inference Model")
    print(f"  Conditions: baseline / psychosis / melancholia")
    print(f"  Learning phase:     {args.t_learn} steps × {args.n_runs} runs")
    print(f"  Interaction phase:  {args.t_sim} steps × {args.n_runs} runs")
    print(f"  Output directory:   {args.output_dir}/")
    print()

    # ── Run simulations ────────────────────────────────────────────────────────
    print("Phase 1 + 2: Running simulations...")
    learning, interaction = run_all(
        n_runs=args.n_runs,
        T_learn=args.t_learn,
        T_sim=args.t_sim,
        verbose=not args.quiet,
    )

    # ── Print summary ──────────────────────────────────────────────────────────
    if not args.quiet:
        print_summary(learning, interaction)

    # ── Generate figures ───────────────────────────────────────────────────────
    print(f"\nGenerating figures → {args.output_dir}/")
    generate_all(learning, interaction, output_dir=args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
