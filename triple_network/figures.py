"""
triple_network/figures.py
=========================
All figure generation for the triple network paper.

Figure 1: Model architecture
Figure 2: Time-series (activation, γ, A-matrix precision, VFE)
Figure 3: Precision architecture summary
    A. Learning trajectories (γ convergence)
    B. Learned pA diagonal profiles
    C. A-matrix precision curves
Figure 4: Aggregate statistics
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyArrowPatch, Patch

from .model import (
    COLORS, N_STATES, a_diag_from_precision,
    get_dmn_precision, get_cen_precision, expected_activation
)

C = COLORS   # shorthand


# ── FIGURE 1: Architecture ─────────────────────────────────────────────────────

def fig1_architecture(path):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=C["bg"])
    ax.set_facecolor(C["bg"]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)

    nodes = [
        (5.0, 5.5, C["SN"],  "Salience Network\n(Real / Jouissance)"),
        (2.0, 2.0, C["DMN"], "Default Mode Network\n(Imaginary / Ego)"),
        (8.0, 2.0, C["CEN"], "Central Executive\n(Symbolic / Law)"),
        (5.0, 2.0, C["OBJ"], "Anterior Insula\nobjet a  /  γ"),
    ]
    for x, y, col, lbl in nodes:
        ax.add_patch(Circle((x, y), 0.88, color=col, zorder=3, alpha=0.93))
        ax.text(x, y, lbl, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white", zorder=4, linespacing=1.5)

    kw = dict(lw=2.0, zorder=2, mutation_scale=14)
    ax.add_patch(FancyArrowPatch(
        (4.2, 4.8), (2.75, 2.8), arrowstyle="-|>",
        color=C["SN"], connectionstyle="arc3,rad=0.18", **kw))
    ax.add_patch(FancyArrowPatch(
        (5.8, 4.8), (7.25, 2.8), arrowstyle="-|>",
        color=C["SN"], connectionstyle="arc3,rad=-0.18", **kw))
    ax.text(2.85, 3.85, "A(γ) diffuse\n→ DMN suppressed",
            fontsize=6.5, color=C["SN"], rotation=50, style="italic", ha="center")
    ax.text(7.15, 3.85, "A(γ) peaked\n→ CEN activated",
            fontsize=6.5, color=C["SN"], rotation=-50, style="italic", ha="center")
    ax.add_patch(FancyArrowPatch(
        (2.88, 2.0), (4.12, 2.0), arrowstyle="<|-|>",
        color=C["OBJ"], mutation_scale=11, lw=1.8, zorder=2))
    ax.add_patch(FancyArrowPatch(
        (5.88, 2.0), (7.12, 2.0), arrowstyle="<|-|>",
        color=C["OBJ"], mutation_scale=11, lw=1.8, zorder=2))

    ax.text(5.0, 0.7,
            r"$\gamma = \mathrm{clip}(\mathbf{w}\cdot Q_\mathrm{SN},\;0.05,\;0.95)$"
            r"   |   $A_\mathrm{DMN}(\gamma) \propto \Pi^{-1}(\gamma)$"
            r"   |   $A_\mathrm{CEN}(\gamma) \propto \Pi(\gamma)$",
            ha="center", va="center", fontsize=7.5, color="#444", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["grey"], lw=0.8))

    items = [
        (C["SN"],  "SN — Real: interoceptive hub / precision controller"),
        (C["DMN"], "DMN — Imaginary: self-model prior  (precision ∝ 1−γ)"),
        (C["CEN"], "CEN — Symbolic: policy selection  (precision ∝ γ)"),
        (C["OBJ"], "γ (anterior insula) — objet a: the precision parameter itself"),
    ]
    for i, (col, txt) in enumerate(items):
        ax.plot(0.3, 6.6 - i * 0.55, "o", color=col, ms=9, zorder=5)
        ax.text(0.65, 6.6 - i * 0.55, txt, va="center", fontsize=7.5, color="#333")

    ax.set_title(
        "Figure 1.  Precision-Weighted Triple Network Architecture\n"
        "(Feldman & Friston 2010; Adams et al. 2013; Friston et al. 2016)",
        fontsize=9.5, fontweight="bold", pad=8, color="#222")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"✓ {path}")


# ── FIGURE 2: Time-series ──────────────────────────────────────────────────────

def fig2_timeseries(interaction, path):
    conds  = ["baseline", "psychosis", "melancholia"]
    titles = [
        "Baseline (neurotic structure)",
        "Psychosis — aberrant precision (Adams et al. 2013; Friston et al. 2016)",
        "Melancholia — imaginary capture (Hamilton et al. 2015; Dall'Aglio 2021c)",
    ]
    # T is inferred from data
    T = interaction[conds[0]]["gamma"].shape[0]
    t = np.arange(T)

    fig = plt.figure(figsize=(18, 9), facecolor=C["bg"])
    outer = gridspec.GridSpec(3, 1, hspace=0.55, figure=fig)

    for row, (cond, title) in enumerate(zip(conds, titles)):
        r = interaction[cond]
        inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[row], wspace=0.38)

        # ── activation
        ax = fig.add_subplot(inner[0]); ax.set_facecolor(C["bg"])
        ax.plot(t, r["sn"] @ np.arange(N_STATES),  color=C["SN"],  lw=2.2, label="SN / Real")
        ax.plot(t, r["dmn"] @ np.arange(N_STATES), color=C["DMN"], lw=2.2, label="DMN / Imaginary")
        ax.plot(t, r["cen"] @ np.arange(N_STATES), color=C["CEN"], lw=2.2, label="CEN / Symbolic")
        ax.set_ylim(-0.1, 3.4); ax.set_xlabel("t", fontsize=8)
        ax.set_ylabel("E[activation]", fontsize=8)
        ax.set_title("Network activation", fontsize=8.5, fontweight="bold")
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.55)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)

        # ── γ
        ax2 = fig.add_subplot(inner[1]); ax2.set_facecolor(C["bg"])
        ax2.fill_between(t, 0, r["gamma"], color=C["OBJ"], alpha=0.35)
        ax2.plot(t, r["gamma"], color=C["OBJ"], lw=2.5)
        ax2.axhline(0.5, color=C["grey"], lw=1, ls="--", alpha=0.7)
        ax2.set_ylim(0, 1.05); ax2.set_xlabel("t", fontsize=8)
        ax2.set_ylabel("γ  (AI precision / objet a)", fontsize=8)
        ax2.set_title("Anterior insula  γ\n(emergent from learning)", fontsize=8.5, fontweight="bold")
        ax2.tick_params(labelsize=7); ax2.spines[["top", "right"]].set_visible(False)

        # ── A-matrix precision
        ax3 = fig.add_subplot(inner[2]); ax3.set_facecolor(C["bg"])
        ax3.plot(t, r["dmn_prec"], color=C["DMN"], lw=2.2, label="DMN A-diag")
        ax3.plot(t, r["cen_prec"], color=C["CEN"], lw=2.2, label="CEN A-diag")
        ax3.axhline(1.0, color=C["grey"], lw=1, ls="--", alpha=0.7)
        ax3.set_xlabel("t", fontsize=8); ax3.set_ylabel("A-diagonal (gain)", fontsize=8)
        ax3.set_title("Synaptic gain\n(A-matrix precision)", fontsize=8.5, fontweight="bold")
        ax3.legend(fontsize=6.5, framealpha=0.55)
        ax3.tick_params(labelsize=7); ax3.spines[["top", "right"]].set_visible(False)

        # ── VFE
        ax4 = fig.add_subplot(inner[3]); ax4.set_facecolor(C["bg"])
        ax4.plot(t, r["sn_fe"],  color=C["SN"],  lw=2, label="SN")
        ax4.plot(t, r["dmn_fe"], color=C["DMN"], lw=2, label="DMN")
        ax4.plot(t, r["cen_fe"], color=C["CEN"], lw=2, label="CEN")
        ax4.set_xlabel("t", fontsize=8); ax4.set_ylabel("VFE", fontsize=8)
        ax4.set_title("Variational free energy", fontsize=8.5, fontweight="bold")
        ax4.legend(fontsize=6.5, loc="upper right", framealpha=0.55)
        ax4.tick_params(labelsize=7); ax4.spines[["top", "right"]].set_visible(False)

        fig.text(0.005, 1 - (row + 0.5) / 3, title,
                 va="center", rotation=90, fontsize=8, fontweight="bold", color="#333")

    fig.suptitle("Figure 2.  Triple Network Dynamics Under Dynamic A-Matrix Precision Modulation",
                 fontsize=11, fontweight="bold", y=1.01, color="#222")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"✓ {path}")


# ── FIGURE 3: Precision architecture ──────────────────────────────────────────

def fig3_precision(learning, interaction, path):
    conds  = ["baseline", "psychosis", "melancholia"]
    lbls   = ["Baseline", "Psychosis\n(foreclosure)", "Melancholia\n(imag. capture)"]
    colors = [C["DMN"], C["SN"], C["CEN"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=C["bg"])

    # ── Panel A: learning trajectories (γ convergence)
    ax = axes[0]; ax.set_facecolor(C["bg"])
    for cond, col, lbl in zip(conds, colors, lbls):
        gt = learning[cond]["gamma_trace"]
        # Smooth with rolling mean
        window = 10
        smoothed = np.convolve(gt, np.ones(window)/window, mode="valid")
        t_smooth = np.arange(len(smoothed)) + window // 2
        ax.plot(t_smooth, smoothed, color=col, lw=2.2,
                label=f"{lbl.split(chr(10))[0]} (μ={gt.mean():.2f})")
    ax.axhline(0.5, color=C["grey"], lw=1, ls="--", alpha=0.7)
    ax.set_xlabel("Learning step", fontsize=9); ax.set_ylabel("γ (rolling mean)", fontsize=9)
    ax.set_title("A.  γ convergence during learning phase\n(N=20 runs averaged)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.6)
    ax.tick_params(labelsize=7.5); ax.spines[["top", "right"]].set_visible(False)

    # ── Panel B: learned pA diagonal profiles
    ax2 = axes[1]; ax2.set_facecolor(C["bg"])
    for ci, (cond, col, lbl) in enumerate(zip(conds, colors, lbls)):
        pA = learning[cond]["pA"]
        diag = np.diag(pA)
        ax2.bar(np.arange(N_STATES) + ci * 0.25, diag,
                width=0.22, color=col, alpha=0.82, edgecolor="white", label=lbl.split(chr(10))[0])
    ax2.set_xticks(np.arange(N_STATES) + 0.25)
    ax2.set_xticklabels(["State 0\n(suppressed)", "State 1", "State 2", "State 3\n(dominant)"], fontsize=7.5)
    ax2.set_ylabel("Dirichlet concentration (pA diagonal)", fontsize=8.5)
    ax2.set_title("B.  Learned pA diagonal per condition\n(emergent from environmental exposure)",
                  fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7.5, framealpha=0.6)
    ax2.tick_params(labelsize=7.5); ax2.spines[["top", "right"]].set_visible(False)

    # ── Panel C: A-matrix precision curves
    ax3 = axes[2]; ax3.set_facecolor(C["bg"])
    gv = np.linspace(0.05, 0.95, 200)
    dmn_d = np.array([a_diag_from_precision(get_dmn_precision(g)) for g in gv])
    cen_d = np.array([a_diag_from_precision(get_cen_precision(g)) for g in gv])
    ax3.plot(gv, dmn_d, color=C["DMN"], lw=2.5, label="DMN A-diagonal (∝ 1−γ)")
    ax3.plot(gv, cen_d, color=C["CEN"], lw=2.5, label="CEN A-diagonal (∝ γ)")
    gm_vals = [learning[c]["gamma_trace"].mean() for c in conds]
    for gm, col, lbl in zip(gm_vals, colors, lbls):
        ax3.axvline(gm, color=col, lw=1.5, ls=":", alpha=0.85,
                    label=f"{lbl.split(chr(10))[0]} γ={gm:.2f}")
    ax3.set_xlabel("γ  (objet a precision)", fontsize=8.5)
    ax3.set_ylabel("A-matrix diagonal (signal gain)", fontsize=8.5)
    ax3.set_title("C.  A-matrix precision modulation\n(Feldman & Friston 2010)",
                  fontsize=9, fontweight="bold")
    ax3.legend(fontsize=6.5, framealpha=0.6, loc="upper left")
    ax3.tick_params(labelsize=7.5); ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Figure 3.  Dynamic Precision Architecture: Learning Trajectories, Emergent pA, A-Matrix Modulation",
                 fontsize=10.5, fontweight="bold", y=1.02, color="#222")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"✓ {path}")


# ── FIGURE 4: Aggregate statistics ────────────────────────────────────────────

def fig4_statistics(learning, interaction, path):
    conds  = ["baseline", "psychosis", "melancholia"]
    lbls   = ["Baseline", "Psychosis\n(foreclosure)", "Melancholia\n(imag. capture)"]
    colors = [C["DMN"], C["SN"], C["CEN"]]

    metrics = {
        "Mean γ\n(learned)":     lambda c: learning[c]["gamma_trace"].mean(),
        "Var(γ)\n(learned)":     lambda c: learning[c]["gamma_trace"].var(),
        "Mean γ\n(interaction)": lambda c: interaction[c]["gamma"].mean(),
        "Var(γ)\n(interaction)": lambda c: interaction[c]["gamma"].var(),
        "Mean DMN\nactivation":  lambda c: (interaction[c]["dmn"] @ np.arange(N_STATES)).mean(),
        "Mean CEN\nactivation":  lambda c: (interaction[c]["cen"] @ np.arange(N_STATES)).mean(),
        "Mean DMN\nprecision":   lambda c: interaction[c]["dmn_prec"].mean(),
        "Mean CEN\nprecision":   lambda c: interaction[c]["cen_prec"].mean(),
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4), facecolor=C["bg"])
    for ax, (name, fn) in zip(axes, metrics.items()):
        ax.set_facecolor(C["bg"])
        vals = [fn(cond) for cond in conds]
        bars = ax.bar(np.arange(3), vals, color=colors, alpha=0.85, edgecolor="white", lw=0.8)
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(lbls, fontsize=7)
        ax.set_title(name, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + abs(b.get_height()) * 0.02 + 1e-4,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    fig.suptitle("Figure 4.  Aggregate Statistics: Learning Phase + Interaction Phase",
                 fontsize=11, fontweight="bold", y=1.03, color="#222")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"✓ {path}")


def generate_all(learning, interaction, output_dir="outputs"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    fig1_architecture(f"{output_dir}/fig1_architecture.png")
    fig2_timeseries(interaction, f"{output_dir}/fig2_simulations.png")
    fig3_precision(learning, interaction, f"{output_dir}/fig3_precision.png")
    fig4_statistics(learning, interaction, f"{output_dir}/fig4_statistics.png")
