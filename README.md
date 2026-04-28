# Precision-Weighted Triple Network Active Inference Model

Companion code for:

> Papathanasiou, V. (2026). *Precision-Weighted Network Dynamics and the Lacanian Subject: A Hierarchical Active Inference Model of the Triple Network.*

---

## Overview

This repository implements a hierarchical active inference model of the brain's **Default Mode Network (DMN)**, **Salience Network (SN)**, and **Central Executive Network (CEN)**, formalizing the structural homology between Lacan's RSI triad and the triple network model.

The core theoretical contribution is that the **SN acts as a precision controller** — not a co-equal FEP unit — over the DMN-CEN interface. The **anterior insula** is formalized as the precision parameter **γ** (Lacan's *objet petit a*): the cause of attentional allocation that directly modulates the likelihood matrix (A-matrix) precision of the DMN and CEN modules.

This differentiates the model from Li & Li (2025)'s symmetric message-passing formalization and grounds it in:
- **Feldman & Friston (2010)**: precision as gain on prediction error, operationalized as A-matrix peakedness
- **Adams et al. (2013)**: psychosis as aberrant precision encoding
- **Friston et al. (2016)**: dysconnection hypothesis — aberrant synaptic gain modulation
- **Dall'Aglio (2021a–c)**: jouissance as surplus prediction error

---

## Key Feature: Dynamic Learning

Unlike static precision-parameter models, this implementation uses **Dirichlet concentration learning** (`pA` updates via `Agent.update_A()`). 

**All three conditions start with identical agents.** Pathological precision profiles *emerge* from different environmental exposure histories:

| Condition | Environmental statistics | Emergent γ profile |
|-----------|--------------------------|-------------------|
| Baseline | Balanced salience `[0.15, 0.35, 0.35, 0.15]` | Adaptive, moderate mean |
| Psychosis | High/noisy salience `[0.05, 0.10, 0.30, 0.55]` | Chronically elevated, reduced variance |
| Melancholia | Chronically understimulated `[0.55, 0.30, 0.10, 0.05]` | Chronically suppressed, reduced variance |

This is consistent with Adams et al.'s (2013) view that aberrant precision encoding is a *learned* property of the generative model, not an inherent structural defect.

---

## Architecture

```
triple_network/
├── model.py       Core model: agents, γ function, A-matrix precision
├── simulate.py    Two-phase simulation (learning + interaction)
└── figures.py     All figure generation

main.py            Entry point
requirements.txt   Python dependencies
outputs/           Generated figures (created at runtime)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/triple-network-active-inference.git
cd triple-network-active-inference

# Install dependencies
pip install -r requirements.txt
```

**Python version**: 3.9+ recommended.

---

## Usage

### Full run (default settings)
```bash
python main.py
```

### Quick test (fewer runs)
```bash
python main.py --n-runs 3 --t-learn 50 --t-sim 20
```

### All options
```
python main.py --help

  --n-runs     N   Seeds per condition (default: 20)
  --t-learn    T   Learning phase steps (default: 200)
  --t-sim      T   Interaction phase steps (default: 50)
  --seed       S   Global random seed (default: 42)
  --output-dir D   Output directory for figures (default: outputs/)
  --quiet          Suppress progress output
```

---

## Outputs

Four figures saved to `outputs/`:

| File | Description |
|------|-------------|
| `fig1_architecture.png` | Model architecture with precision formulas |
| `fig2_simulations.png` | Time-series: activation, γ, A-matrix precision, VFE |
| `fig3_precision.png` | γ learning trajectories, learned pA profiles, A-matrix curves |
| `fig4_statistics.png` | Aggregate statistics across conditions |

---

## Two-Phase Simulation

**Phase 1 — Learning** (200 steps per run):  
The SN agent updates its Dirichlet concentration prior `pA` from environmental observations via `Agent.update_A()`. Different environmental statistics produce different learned A matrices. No interaction between agents occurs in this phase.

**Phase 2 — Interaction** (50 steps per run):  
Triple-network interaction using the frozen learned SN A matrix. At each step:
1. SN observes environment → infers posterior Q_SN → computes γ
2. γ directly modulates DMN and CEN A matrices:
   - `A_DMN precision = max(0.25, 1.0 − γ + 0.15)` → SN suppresses DMN
   - `A_CEN precision = min(1.10, max(0.20, γ + 0.15))` → SN activates CEN
3. DMN and CEN run active inference with their modulated A matrices
4. All agents sample actions and update states

---

## Theoretical Background

The model formalizes the structural homology proposed in:

> Papathanasiou, V. (2026). *The Lacanian Subject in the Triple Network: A Neuro-Psychoanalytic Mapping of the Real, Symbolic, and Imaginary onto Default Mode, Salience, and Central Executive Networks.*

**RSI–Triple Network mapping:**
- **Imaginary → DMN**: self-referential narrative, ego-as-fiction, autobiographical continuity
- **Symbolic → CEN**: rule-governed, differential, linguistically mediated action
- **Real → SN**: interoceptive insistence, jouissance, unsymbolizable surplus prediction error
- **objet a → anterior insula**: the γ precision parameter, pivot between Imaginary and Symbolic

**Clinical implications** (Dall'Aglio 2021c): Lacanian clinical technique (*scansion*, *punctuation*) works by *provoking* prediction error — "artificial precision modulation" — rather than eliminating it. At the triple-network level, this translates to restoration of γ variance (adaptive precision flexibility) rather than shifts in γ mean.

---

## References

Adams, R.A., Stephan, K.E., Brown, H.R., Frith, C.D., & Friston, K.J. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*, 4, 47.

Dall'Aglio, J. (2021a). Sex and prediction error, Part 1: The metapsychology of jouissance. *JAPA*, 69(4), 693–714.

Dall'Aglio, J. (2021b). Sex and prediction error, Part 2: Jouissance and the free energy principle. *JAPA*, 69(4), 715–741.

Dall'Aglio, J. (2021c). Sex and prediction error, Part 3: Provoking prediction error. *JAPA*, 69(4), 743–765.

Dall'Aglio, J. (2024). *A Lacanian Neuropsychoanalysis: Consciousness Enjoying Uncertainty*. Palgrave Macmillan.

Feldman, H., & Friston, K.J. (2010). Attention, uncertainty, and free-energy. *Frontiers in Human Neuroscience*, 4, 215.

Friston, K.J., Brown, H.R., Siemerkus, J., & Stephan, K.E. (2016). The dysconnection hypothesis. *Schizophrenia Research*, 176, 83–94.

Heins, C. et al. (2022). pymdp: A Python library for active inference in discrete state spaces. *JOSS*, 7(73), 4098.

Li, L., & Li, C. (2025). Formalizing Lacanian psychoanalysis through the free energy principle. *Frontiers in Psychology*, 16, 1574650.

Menon, V. (2011). Large-scale brain networks and psychopathology: A unifying triple network model. *TiCS*, 15(10), 483–506.

Parr, T., & Friston, K.J. (2019). Generalised free energy and active inference. *Biological Cybernetics*, 113, 495–513.

Seth, A.K., & Friston, K.J. (2016). Active interoceptive inference and the emotional brain. *Phil. Trans. R. Soc. B*, 371, 20160007.

---

## Citation

If you use this code, please cite:

```bibtex
@article{papathanasiou2026,
  author  = {Papathanasiou, Vassilis},
  title   = {Precision-Weighted Network Dynamics and the {L}acanian Subject:
             {A} Hierarchical Active Inference Model of the Triple Network},
  journal = {Neuropsychoanalysis},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## License

MIT License. See `LICENSE` for details.
