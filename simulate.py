"""
triple_network/simulate.py
==========================
Dynamic simulation: pathological precision profiles EMERGE from
different learning histories, not from manually set parameters.

Two-phase structure
-------------------
Phase 1 — LEARNING (T_learn steps):
    SN agent updates Dirichlet concentration pA from environmental exposure.
    Different environmental statistics → different learned A matrices.

    Condition environments:
        baseline    : balanced salience  [0.15, 0.35, 0.35, 0.15]
        psychosis   : high, noisy salience  [0.05, 0.10, 0.30, 0.55]
                      (aberrant precision encoding; Adams et al. 2013;
                       Kapur 2003 dopaminergic hyperactivation)
        melancholia : chronically understimulated  [0.55, 0.30, 0.10, 0.05]
                      (reduced synaptic gain; Friston et al. 2016)

Phase 2 — INTERACTION (T_sim steps):
    Triple-network interaction using learned SN A matrix.
    γ is fully emergent from learned precision profile.
    DMN and CEN A matrices modulated in real time by γ.
"""

import numpy as np
from copy import deepcopy
from pymdp import utils

from .model import (
    make_sn_agent, make_dmn_agent, make_cen_agent,
    make_A_from_precision, pA_to_A,
    gamma_from_qs, get_dmn_precision, get_cen_precision,
    approx_vfe, state_step, expected_activation,
    a_diag_from_precision, N_STATES
)

# ── Environmental salience distributions ──────────────────────────────────────
ENVIRONMENTS = {
    "baseline":    np.array([0.15, 0.35, 0.35, 0.15]),
    "psychosis":   np.array([0.05, 0.10, 0.30, 0.55]),
    "melancholia": np.array([0.55, 0.30, 0.10, 0.05]),
}

# ── Simulation hyperparameters ────────────────────────────────────────────────
T_LEARN = 200   # learning phase: steps per run
T_SIM   = 50    # interaction phase: steps per run
N_RUNS  = 20    # seeds for averaging


# ── Phase 1: SN learning ──────────────────────────────────────────────────────

def learn_sn(condition, seed, T=T_LEARN, lr_pA=0.35):
    """
    Run the SN learning phase under a specific environmental distribution.
    Returns the learned pA (Dirichlet concentration) and derived A matrix.
    """
    np.random.seed(seed)
    env = ENVIRONMENTS[condition]

    # DMN preference biased toward high activation in melancholia
    if condition == "melancholia":
        agent = make_sn_agent(lr_pA=lr_pA, pref_state=0, pref_strength=3.0)
    else:
        agent = make_sn_agent(lr_pA=lr_pA)

    sn_state = 2
    gamma_trace = []

    for _ in range(T):
        obs = int(np.random.choice(N_STATES, p=env))
        qs = agent.infer_states([obs])[0]
        agent.infer_policies()
        action = int(agent.sample_action()[0])

        # Dirichlet update: learned pA concentrates around what is observed
        agent.update_A([obs])

        gamma_trace.append(gamma_from_qs(qs))
        sn_state = state_step(sn_state, action)

    return {
        "pA": deepcopy(agent.pA[0]),
        "A":  deepcopy(agent.A[0]),
        "gamma_trace": np.array(gamma_trace),
    }


def learn_condition(condition, n_runs=N_RUNS, T=T_LEARN):
    """Average SN learning results across multiple seeds."""
    runs = [learn_sn(condition, seed=s, T=T) for s in range(n_runs)]
    avg_pA = np.mean([r["pA"] for r in runs], axis=0)
    avg_A  = avg_pA / avg_pA.sum(axis=0, keepdims=True)
    avg_gamma_trace = np.mean([r["gamma_trace"] for r in runs], axis=0)
    return {
        "pA":          avg_pA,
        "A":           avg_A,
        "gamma_trace": avg_gamma_trace,
        "runs":        runs,
    }


# ── Phase 2: Triple-network interaction ────────────────────────────────────────

def run_interaction(learned_A, condition, seed, T=T_SIM):
    """
    Run triple-network interaction phase using the learned SN A matrix.
    γ is fully emergent from the learned precision profile.

    Parameters
    ----------
    learned_A : np.ndarray, shape (N_STATES, N_STATES)
        The SN's learned likelihood matrix from Phase 1.
    condition : str
        Used to set DMN initial state / preference for melancholia.
    seed : int
    T : int

    Returns
    -------
    dict with time-series arrays of length T.
    """
    np.random.seed(seed)
    env = ENVIRONMENTS[condition]

    # SN agent uses learned A (no further learning in interaction phase)
    sn_A = utils.obj_array(1); sn_A[0] = learned_A
    sn_agent = make_sn_agent(lr_pA=0.0)  # learning rate 0 = frozen
    sn_agent.A = sn_A

    # DMN and CEN agents — A modulated by γ each step
    if condition == "melancholia":
        dmn_agent = make_dmn_agent(init_state=3, pref_state=3, pref_strength=4.5)
    else:
        dmn_agent = make_dmn_agent()
    cen_agent = make_cen_agent()

    # Storage
    sb = np.zeros((T, N_STATES)); db = np.zeros((T, N_STATES)); cb = np.zeros((T, N_STATES))
    gamma_ts = np.zeros(T)
    dmn_prec = np.zeros(T); cen_prec = np.zeros(T)
    sn_fe = np.zeros(T); dmn_fe = np.zeros(T); cen_fe = np.zeros(T)

    sn_s = 2; dmn_s = 2; cen_s = 1

    for t in range(T):
        obs = int(np.random.choice(N_STATES, p=env))

        # ── SN inference (frozen A)
        qs_sn = sn_agent.infer_states([obs])[0]
        sn_agent.infer_policies()
        a_sn = int(sn_agent.sample_action()[0])
        sb[t] = qs_sn

        # ── γ: objet a / anterior insula precision
        g = gamma_from_qs(qs_sn)
        gamma_ts[t] = g

        # ── A-matrix precision modulation (Feldman & Friston 2010)
        dp = get_dmn_precision(g); cp = get_cen_precision(g)
        dmn_prec[t] = a_diag_from_precision(dp)
        cen_prec[t] = a_diag_from_precision(cp)

        dmn_agent.A = make_A_from_precision(dp)
        qs_dm = dmn_agent.infer_states([int(dmn_s)])[0]
        dmn_agent.infer_policies()
        a_dm = int(dmn_agent.sample_action()[0])
        db[t] = qs_dm

        cen_agent.A = make_A_from_precision(cp)
        qs_cn = cen_agent.infer_states([int(cen_s)])[0]
        cen_agent.infer_policies()
        a_cn = int(cen_agent.sample_action()[0])
        cb[t] = qs_cn

        # ── VFE
        sn_fe[t]  = approx_vfe(qs_sn, sn_agent,  obs)
        dmn_fe[t] = approx_vfe(qs_dm, dmn_agent, int(dmn_s))
        cen_fe[t] = approx_vfe(qs_cn, cen_agent, int(cen_s))

        sn_s  = state_step(sn_s,  a_sn)
        dmn_s = state_step(dmn_s, a_dm)
        cen_s = state_step(cen_s, a_cn)

    return dict(
        sn=sb, dmn=db, cen=cb,
        gamma=gamma_ts,
        dmn_prec=dmn_prec, cen_prec=cen_prec,
        sn_fe=sn_fe, dmn_fe=dmn_fe, cen_fe=cen_fe,
    )


def run_condition(condition, learned_A, n_runs=N_RUNS, T=T_SIM):
    """
    Average triple-network interaction over multiple seeds.

    Parameters
    ----------
    condition : str
    learned_A : np.ndarray
        Learned SN likelihood from Phase 1 (averaged across seeds).
    """
    runs = [run_interaction(learned_A, condition, seed=s, T=T) for s in range(n_runs)]
    avg = {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0]}
    avg["runs"] = runs
    return avg


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_all(conditions=("baseline", "psychosis", "melancholia"),
            n_runs=N_RUNS, T_learn=T_LEARN, T_sim=T_SIM, verbose=True):
    """
    Run both phases for all conditions.

    Returns
    -------
    learning : dict[condition → learning results]
    interaction : dict[condition → interaction results]
    """
    learning = {}
    interaction = {}

    for cond in conditions:
        if verbose:
            print(f"  [{cond}] learning phase ({T_learn} steps × {n_runs} runs)...")
        learning[cond] = learn_condition(cond, n_runs=n_runs, T=T_learn)

        if verbose:
            gm = learning[cond]["gamma_trace"].mean()
            print(f"           learned γ mean = {gm:.3f}")
            print(f"  [{cond}] interaction phase ({T_sim} steps × {n_runs} runs)...")

        interaction[cond] = run_condition(
            cond, learning[cond]["A"], n_runs=n_runs, T=T_sim
        )

        if verbose:
            r = interaction[cond]
            gm2  = r["gamma"].mean()
            dm_m = np.mean([expected_activation(r["dmn"][t]) for t in range(T_sim)])
            cm_m = np.mean([expected_activation(r["cen"][t]) for t in range(T_sim)])
            print(f"           interaction γ = {gm2:.3f}  |  DMN = {dm_m:.3f}  CEN = {cm_m:.3f}")

    return learning, interaction
