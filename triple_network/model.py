"""
triple_network/model.py
=======================
Core generative model components for the precision-weighted triple network
active inference model.

Theory
------
Implements the precision-weighted active inference model described in:
  Papathanasiou, V. (2026). Precision-Weighted Network Dynamics and the
  Lacanian Subject: A Hierarchical Active Inference Model of the Triple Network.

Architecture
------------
  SN  (Salience Network)     → Real register       → precision controller γ
  DMN (Default Mode Network) → Imaginary register  → self-model prior
  CEN (Central Executive)    → Symbolic register   → policy selection

The anterior insula is formalized as γ, the precision parameter (objet a):
  γ = clip(w · Q_SN,  0.05,  0.95)

A-matrix precision modulation (Feldman & Friston 2010):
  A_DMN precision ∝ (1 − γ)   [SN suppresses DMN with high salience]
  A_CEN precision ∝  γ         [SN activates CEN with high salience]

The key innovation over Li & Li (2025): the SN acts as precision controller
rather than a co-equal FEP unit, formalizing the Real's asymmetric role in
the Borromean knot.

Pathological dynamics
---------------------
In the DYNAMIC version of this model, all agents start with identical
prior A matrices. Pathological precision profiles EMERGE from different
learning histories (Dirichlet updating via pA), rather than being imposed
as fixed parameter differences. This is consistent with the computational
psychiatry view (Adams et al. 2013; Friston et al. 2016) that psychiatric
disorders involve learned predictive models that perpetuate themselves.
"""

import numpy as np
from pymdp import utils
from pymdp.agent import Agent

# ── Constants ─────────────────────────────────────────────────────────────────
N_STATES = 4      # hidden states: 0=suppressed → 3=dominant
N_OBS    = 4      # observations: same cardinality
GAMMA_W  = np.array([0.0, 0.2, 0.6, 1.0])  # state → γ weight vector
GAMMA_CLIP = (0.05, 0.95)

# ── Prior A matrix (near-identity) ────────────────────────────────────────────
A_PRIOR_DIAG    = 10.0   # Dirichlet concentration on diagonal
A_PRIOR_OFFDIAG =  0.5   # Dirichlet concentration off-diagonal

# ── Colour palette (for figures) ──────────────────────────────────────────────
COLORS = dict(SN="#D95F3B", DMN="#3A7EC2", CEN="#2A9E63",
              OBJ="#8E44AD", grey="#AAAAAA", bg="#FAFAFA")


# ── Matrix constructors ────────────────────────────────────────────────────────

def make_pA():
    """
    Dirichlet concentration prior for A (near-identity start).
    pA[obs, state]: high diagonal concentration, low off-diagonal.
    Normalised A = pA / sum(pA, axis=0) produces a peaked likelihood.
    """
    pA = utils.obj_array(1)
    pa = np.full((N_STATES, N_STATES), A_PRIOR_OFFDIAG)
    np.fill_diagonal(pa, A_PRIOR_DIAG)
    pA[0] = pa
    return pA


def pA_to_A(pA_arr):
    """Convert Dirichlet concentration matrix to normalised likelihood matrix."""
    A = utils.obj_array(1)
    A[0] = pA_arr / pA_arr.sum(axis=0, keepdims=True)
    return A


def make_A_from_precision(precision=1.0):
    """
    Construct A directly from a precision scalar.
    precision=1.0  → baseline near-identity (noise ≈ 0.10)
    precision>1.0  → sharper (lower noise)
    precision<1.0  → flatter (higher noise, signal suppressed)
    """
    noise = float(np.clip(0.10 / precision, 1e-3, 0.45))
    a = np.eye(N_STATES) * (1.0 - noise) + noise / N_STATES
    A = utils.obj_array(1)
    A[0] = a / a.sum(axis=0, keepdims=True)
    return A


def make_B():
    """
    Transition matrix B[next_state, current_state, action].
    Actions: 0=maintain, 1=increase, 2=decrease activation.
    Self-transition probability = 0.75.
    """
    b = np.zeros((N_STATES, N_STATES, 3))
    for s in range(N_STATES):
        for ai, ds in enumerate([0, 1, -1]):
            ns = int(np.clip(s + ds, 0, N_STATES - 1))
            b[ns, s, ai] = 0.75
            for ns2 in range(N_STATES):
                if ns2 != ns:
                    b[ns2, s, ai] += 0.25 / (N_STATES - 1)
    B = utils.obj_array(1)
    B[0] = b / b.sum(axis=0, keepdims=True)
    return B


def make_C(preferred_state=2, strength=2.5):
    """Log-preference vector over observations."""
    C = utils.obj_array(1)
    c = np.zeros(N_STATES)
    c[preferred_state] = strength
    C[0] = c
    return C


def make_D(init_state=2):
    """Prior over initial hidden states."""
    D = utils.obj_array(1)
    D[0] = utils.onehot(init_state, N_STATES)
    return D


def make_sn_agent(lr_pA=0.3, pref_state=2, pref_strength=2.5):
    """
    Create an SN agent with Dirichlet learning enabled.
    All conditions start with the SAME agent — pathology emerges from
    different environmental exposure histories via update_A().
    """
    pA = make_pA()
    return Agent(
        A=pA_to_A(pA[0]),
        B=make_B(),
        C=make_C(pref_state, pref_strength),
        D=make_D(2),
        pA=pA,
        modalities_to_learn=[0],
        lr_pA=lr_pA,
    )


def make_dmn_agent(init_state=2, pref_state=2, pref_strength=2.5):
    """DMN agent — A matrix is precision-modulated externally via γ."""
    return Agent(
        A=make_A_from_precision(1.0),
        B=make_B(),
        C=make_C(pref_state, pref_strength),
        D=make_D(init_state),
    )


def make_cen_agent(init_state=1):
    """CEN agent — A matrix is precision-modulated externally via γ."""
    return Agent(
        A=make_A_from_precision(1.0),
        B=make_B(),
        C=make_C(1, 2.5),
        D=make_D(init_state),
    )


# ── γ: the objet a ────────────────────────────────────────────────────────────

def gamma_from_qs(qs_sn):
    """
    Compute precision parameter γ ∈ [0.05, 0.95] from SN posterior.
    γ is the formal objet a: the cause of attentional allocation,
    mediating between Imaginary self-modeling and Symbolic action
    without belonging to either.
    """
    g = float(np.dot(qs_sn, GAMMA_W))
    return float(np.clip(g, *GAMMA_CLIP))


def get_dmn_precision(gamma):
    """
    A_DMN precision as function of γ.
    High γ → lower precision (SN suppresses DMN signal).
    """
    return float(np.clip(1.0 - gamma + 0.15, 0.25, 1.10))


def get_cen_precision(gamma):
    """
    A_CEN precision as function of γ.
    High γ → higher precision (SN activates CEN signal).
    """
    return float(np.clip(gamma + 0.15, 0.20, 1.10))


def a_diag_from_precision(precision):
    """Return A-matrix diagonal value (signal gain) for a given precision."""
    noise = float(np.clip(0.10 / precision, 1e-3, 0.45))
    return 1.0 - noise


# ── VFE approximation ─────────────────────────────────────────────────────────

def kl_div(q, p):
    """KL(q || p) divergence."""
    q = np.clip(q, 1e-10, 1.0)
    p = np.clip(p, 1e-10, 1.0)
    return float(np.sum(q * np.log(q / p)))


def approx_vfe(qs, agent, obs):
    """
    Variational free energy approximation:
    F ≈ KL(Q(s) || D) − log P(o | Q(s))
    """
    kl = kl_div(qs, agent.D[0])
    ll = float(-np.log(agent.A[0][obs, :] @ qs + 1e-8))
    return kl + ll


def state_step(state, action):
    """Deterministic state transition from sampled action."""
    if action == 1:
        return min(state + 1, N_STATES - 1)
    if action == 2:
        return max(state - 1, 0)
    return state


def expected_activation(beliefs):
    """E[s] under categorical posterior beliefs."""
    return float(beliefs @ np.arange(N_STATES))
