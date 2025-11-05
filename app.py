# app.py
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Expect these to already exist in the environment / repo ----------
# base_state, base_params, base_precomputed, base_schedules, flu, T, timesteps_per_day
# -----------------------------------------------------------------------------


# =============== Utilities (tensor-safe helpers) ===============

def simulate_total_admits(state, params, precomputed, schedules, T, tpd):
    """Return daily aggregate admissions as a 1-D numpy array of length T."""
    with torch.no_grad():
        admits = flu.torch_simulate_hospital_admits(state, params, precomputed, schedules, T, tpd)
        # admits likely [T, L, A, R] daily; aggregate to [T]
        return torch.sum(admits, dim=(1, 2, 3)).cpu().numpy()

def as_like(val, like_tensor):
    """Create a tensor with the same dtype/device as like_tensor."""
    return torch.as_tensor(val, dtype=like_tensor.dtype, device=like_tensor.device)

def apply_rate_multiplier(p, field_name, mult):
    """Multiply a rate tensor in-place (tensor-safe)."""
    cur = getattr(p, field_name)
    setattr(p, field_name, cur * as_like(mult, cur))

def set_beta_by_location(p, beta_vec):
    """Set beta_baseline from a length-L vector; broadcast to [L, A, R] (tensor-safe)."""
    L, A, R = p.beta_baseline.shape
    beta_like = p.beta_baseline
    vec = np.asarray(beta_vec, dtype=float)
    assert len(vec) == L, f"beta_by_location must have length L={L}"
    new_beta = torch.as_tensor(vec, dtype=beta_like.dtype, device=beta_like.device).view(L, 1, 1).expand(L, A, R)
    p.beta_baseline = new_beta

# =============== Streamlit UI ===============

st.set_page_config(page_title="Metapop Admissions Explorer", layout="wide")

st.title("Metapop Admissions Explorer (β & Rate Sliders)")
st.caption("Adjust location betas and epidemiological rate multipliers; see aggregate admissions over time.")

# Pull some defaults from your model to show
L, A, R = base_params.beta_baseline.shape
default_rates = {
    "E_to_I_rate":    float(base_params.E_to_I_rate),
    "IP_to_IS_rate":  float(base_params.IP_to_IS_rate),
    "IS_to_R_rate":   float(base_params.IS_to_R_rate),
    "IA_to_R_rate":   float(base_params.IA_to_R_rate),
}
default_immunity = {
    "R_to_S_rate":                      float(base_params.R_to_S_rate),
    "inf_induced_immune_wane":          float(torch.as_tensor(base_params.inf_induced_immune_wane).mean().item()),
    "inf_induced_inf_risk_reduce":      float(torch.as_tensor(base_params.inf_induced_inf_risk_reduce).mean().item()),
}

with st.expander("Show model defaults", expanded=False):
    st.write("**Rate parameters (per day):**")
    st.json(default_rates)
    st.write("**Immunity / reinfection parameters (per day or unitless multipliers as applicable):**")
    st.json(default_immunity)

st.markdown("### β by Location")
col_b0, col_b1, col_b2 = st.columns(3)

# β sliders — defaults 0.0005, range [0.0001 .. 0.2], step 0.0001
beta0 = col_b0.slider("β L0", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta1 = col_b1.slider("β L1", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta2 = col_b2.slider("β L2", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta_vec = [beta0, beta1, beta2]

st.markdown("---")
st.markdown("### Rate Multipliers (relative to defaults)")
col_r1, col_r2, col_r3, col_r4 = st.columns(4)

# Reasonable range for multipliers: 0.25x .. 2.0x (step 0.05)
m_EI = col_r1.slider("E→I multiplier",  min_value=0.25, max_value=2.0, value=1.0, step=0.05)
m_IP = col_r2.slider("IP→IS multiplier", min_value=0.25, max_value=2.0, value=1.0, step=0.05)
m_ISR = col_r3.slider("IS→R multiplier", min_value=0.25, max_value=2.0, value=1.0, step=0.05)
m_IAR = col_r4.slider("IA→R multiplier", min_value=0.25, max_value=2.0, value=1.0, step=0.05)

st.markdown("---")
st.markdown("### Reinfection / Immunity Multipliers")
col_i1, col_i2, col_i3 = st.columns(3)

# R→S multiplier lets you do the 0.25x trick from earlier (default 1.0)
m_RS  = col_i1.slider("R→S (reinfection) multiplier", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
# Waning multiplier (bigger = faster waning)
m_wane = col_i2.slider("inf_induced_immune_wane multiplier", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
# Residual protection multiplier (bigger = stronger protection)
m_prot = col_i3.slider("inf_induced_inf_risk_reduce multiplier", min_value=0.5, max_value=1.5, value=1.0, step=0.05)

st.markdown("---")
go = st.button("Run Simulation")

# =============== Run & Plot ===============
if go:
    # Build working params from base
    p = copy.deepcopy(base_params)

    # 1) Set β by location
    set_beta_by_location(p, beta_vec)

    # 2) Apply rate multipliers (tensor-safe)
    apply_rate_multiplier(p, "E_to_I_rate",   m_EI)
    apply_rate_multiplier(p, "IP_to_IS_rate", m_IP)
    apply_rate_multiplier(p, "IS_to_R_rate",  m_ISR)
    apply_rate_multiplier(p, "IA_to_R_rate",  m_IAR)

    # 3) Apply immunity/reinfection multipliers (tensor-safe)
    apply_rate_multiplier(p, "R_to_S_rate", m_RS)

    # inf_induced_immune_wane and inf_induced_inf_risk_reduce may be scalar-tensors or arrays.
    # We multiply the existing tensors to preserve shapes.
    cur_wane = p.inf_induced_immune_wane
    p.inf_induced_immune_wane = cur_wane * as_like(m_wane, torch.as_tensor(cur_wane))

    cur_prot = p.inf_induced_inf_risk_reduce
    p.inf_induced_inf_risk_reduce = cur_prot * as_like(m_prot, torch.as_tensor(cur_prot))

    # 4) Simulate and plot
    y = simulate_total_admits(base_state, p, base_precomputed, base_schedules, T, timesteps_per_day)
    xs = np.arange(len(y))  # daily x-axis

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, y, linewidth=2)
    ax.set_title("Aggregate Hospital Admissions (sum over L, A, R)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Total Daily Hospital Admissions")
    ax.grid(True, linestyle='--', alpha=0.5)

    # annotate current settings under the plot
    txt = (
        f"β = [{beta_vec[0]:.4f}, {beta_vec[1]:.4f}, {beta_vec[2]:.4f}]  |  "
        f"E→I×{m_EI:.2f}, IP→IS×{m_IP:.2f}, IS→R×{m_ISR:.2f}, IA→R×{m_IAR:.2f}  |  "
        f"R→S×{m_RS:.2f}, wane×{m_wane:.2f}, protection×{m_prot:.2f}"
    )
    ax.text(0.01, -0.18, txt, transform=ax.transAxes, fontsize=10, va='top', ha='left', wrap=True)

    st.pyplot(fig)
else:
    st.info("Adjust sliders and click **Run Simulation** to generate the plot.")
