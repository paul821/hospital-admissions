# app.py
from pathlib import Path
import sys
import subprocess
import copy
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# Page / Layout
# =========================
st.set_page_config(page_title="Metapop Admissions Explorer", layout="wide")

# =========================
# Torch guard
# =========================
try:
    import torch  # noqa
except Exception as e:
    st.error(
        "PyTorch failed to import. Ensure your repo has `requirements.txt` with `torch==2.9.0`, "
        "then Manage app → Clear cache and reinstall.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# =========================
# Ensure model repo present
# =========================
APP_DIR = Path(__file__).parent.resolve()
CLT_DIR = APP_DIR / "CLT_BaseModel"

def ensure_clt_repo():
    if CLT_DIR.exists():
        return True
    try:
        st.info("CLT_BaseModel not found. Cloning repository…")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/LP-relaxation/CLT_BaseModel.git", str(CLT_DIR)],
            check=True, capture_output=True, text=True
        )
        st.success("CLT_BaseModel cloned successfully.")
        return True
    except Exception as e:
        st.error(
            "Failed to clone CLT_BaseModel automatically.\n"
            "Please vendor the folder into your app repo at `CLT_BaseModel/` and redeploy.\n\n"
            f"Details: {e}"
        )
        return False

if not ensure_clt_repo():
    st.stop()

if str(CLT_DIR) not in sys.path:
    sys.path.insert(0, str(CLT_DIR))

# =========================
# Import model packages
# =========================
try:
    import clt_toolkit as clt
    import flu_core as flu
except Exception as e:
    st.error(
        "Failed to import `clt_toolkit` / `flu_core` from CLT_BaseModel.\n"
        "Verify those modules exist in CLT_BaseModel/ and are importable.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# =========================
# Helpers
# =========================
def as_like(val, like_tensor):
    return torch.as_tensor(val, dtype=like_tensor.dtype, device=like_tensor.device)

def set_beta_by_location(p, beta_vec):
    """beta_vec length must equal L; broadcast to [L, A, R]."""
    L, A, R = p.beta_baseline.shape
    vec = np.asarray(beta_vec, dtype=float)
    if len(vec) != L:
        raise ValueError(f"beta_by_location length must be {L}, got {len(vec)}")
    new_beta = torch.as_tensor(vec, dtype=p.beta_baseline.dtype, device=p.beta_baseline.device)\
                    .view(L, 1, 1).expand(L, A, R)
    p.beta_baseline = new_beta

def apply_rate_multiplier(p, field_name, mult):
    cur = getattr(p, field_name)
    setattr(p, field_name, cur * as_like(mult, cur))

def simulate_total_admits(state, params, precomputed, schedules, T, tpd):
    with torch.no_grad():
        admits = flu.torch_simulate_hospital_admits(state, params, precomputed, schedules, T, tpd)
        return torch.sum(admits, dim=(1, 2, 3)).cpu().numpy()  # [T]

# =========================
# Load model inputs
# =========================
@st.cache_resource(show_spinner=True)
def load_model_inputs():
    import pandas as pd

    T = 180
    timesteps_per_day = 4

    texas_files_path = CLT_DIR / "flu_instances" / "texas_input_files"
    calibration_files_path = CLT_DIR / "flu_instances" / "calibration_research_input_files"

    subpopA_init_vals_fp = calibration_files_path / "subpopA_init_vals.json"
    subpopB_init_vals_fp = calibration_files_path / "subpopB_init_vals.json"
    subpopC_init_vals_fp = calibration_files_path / "subpopC_init_vals.json"
    common_subpop_params_fp = texas_files_path / "common_subpop_params.json"
    mixing_params_fp = calibration_files_path / "ABC_mixing_params.json"
    simulation_settings_fp = texas_files_path / "simulation_settings.json"

    calendar_df = pd.read_csv(texas_files_path / "school_work_calendar.csv", index_col=0)
    humidity_df = pd.read_csv(texas_files_path / "absolute_humidity_austin_2023_2024.csv", index_col=0)
    vaccines_df = pd.read_csv(texas_files_path / "daily_vaccines_constant.csv", index_col=0)

    schedules_info = flu.FluSubpopSchedules(
        absolute_humidity=humidity_df,
        flu_contact_matrix=calendar_df,
        daily_vaccines=vaccines_df,
    )

    subpopA_init_vals = clt.make_dataclass_from_json(subpopA_init_vals_fp, flu.FluSubpopState)
    subpopB_init_vals = clt.make_dataclass_from_json(subpopB_init_vals_fp, flu.FluSubpopState)
    subpopC_init_vals = clt.make_dataclass_from_json(subpopC_init_vals_fp, flu.FluSubpopState)

    common_subpop_params = clt.make_dataclass_from_json(common_subpop_params_fp, flu.FluSubpopParams)
    mixing_params = clt.make_dataclass_from_json(mixing_params_fp, flu.FluMixingParams)
    simulation_settings = clt.make_dataclass_from_json(simulation_settings_fp, flu.SimulationSettings)

    simulation_settings = clt.updated_dataclass(simulation_settings, {"timesteps_per_day": timesteps_per_day})

    # These are just seed baseline betas (you override via sliders)
    subpopA_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 1.5})
    subpopB_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 2.5})
    subpopC_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 2.2})

    subpopA = flu.FluSubpopModel(subpopA_init_vals, subpopA_params, simulation_settings,
                                 np.random.Generator(np.random.MT19937(111)), schedules_info, name="subpopA")
    subpopB = flu.FluSubpopModel(subpopB_init_vals, subpopB_params, simulation_settings,
                                 np.random.Generator(np.random.MT19937(222)), schedules_info, name="subpopB")
    subpopC = flu.FluSubpopModel(subpopC_init_vals, subpopC_params, simulation_settings,
                                 np.random.Generator(np.random.MT19937(333)), schedules_info, name="subpopC")

    flu_demo_model = flu.FluMetapopModel([subpopA, subpopB, subpopC], mixing_params)
    d = flu_demo_model.get_flu_torch_inputs()

    return dict(
        base_state=d["state_tensors"],
        base_params=d["params_tensors"],
        base_schedules=d["schedule_tensors"],
        base_precomputed=d["precomputed"],
        T=T,
        timesteps_per_day=timesteps_per_day,
    )

ctx = load_model_inputs()
base_state       = ctx["base_state"]
base_params      = ctx["base_params"]
base_schedules   = ctx["base_schedules"]
base_precomputed = ctx["base_precomputed"]
T                = ctx["T"]
timesteps_per_day= ctx["timesteps_per_day"]

# =========================
# Sidebar (scrollable) — all controls
# =========================
st.sidebar.title("Controls")

# Show defaults for transparency
L, A, R = base_params.beta_baseline.shape
default_rates = {
    "E_to_I_rate":    float(base_params.E_to_I_rate),
    "IP_to_IS_rate":  float(base_params.IP_to_IS_rate),
    "IS_to_R_rate":   float(base_params.IS_to_R_rate),
    "IA_to_R_rate":   float(base_params.IA_to_R_rate),
}
default_immunity = {
    "R_to_S_rate":                 float(base_params.R_to_S_rate),
    "inf_induced_immune_wane":     float(torch.as_tensor(base_params.inf_induced_immune_wane).mean().item()),
    "inf_induced_inf_risk_reduce": float(torch.as_tensor(base_params.inf_induced_inf_risk_reduce).mean().item()),
}

with st.sidebar.expander("Model defaults (from loaded tensors)", expanded=False):
    st.write("**Rates (per day):**"); st.json(default_rates)
    st.write("**Immunity / reinfection:**"); st.json(default_immunity)

st.sidebar.markdown("### β by Location")
beta0 = st.sidebar.slider("β L0", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta1 = st.sidebar.slider("β L1", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta2 = st.sidebar.slider("β L2", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta_vec = [beta0, beta1, beta2]

st.sidebar.markdown("---")
st.sidebar.markdown("### Rate Multipliers (relative to defaults)")
m_EI  = st.sidebar.slider("E→I ×",   0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_IP  = st.sidebar.slider("IP→IS ×", 0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_ISR = st.sidebar.slider("IS→R ×",  0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_IAR = st.sidebar.slider("IA→R ×",  0.0001, 5.0, 1.0, 0.01, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("### Reinfection / Immunity Multipliers")
m_RS   = st.sidebar.slider("R→S × (reinfection)",           0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_wane = st.sidebar.slider("inf_induced_immune_wane ×",     0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_prot = st.sidebar.slider("inf_induced_inf_risk_reduce ×", 0.0001, 5.0, 1.0, 0.01, format="%.4f")

# =========================
# Build params and simulate (reactive)
# =========================
p = copy.deepcopy(base_params)

# 1) Set β by location
set_beta_by_location(p, beta_vec)

# 2) Rate multipliers
apply_rate_multiplier(p, "E_to_I_rate",   m_EI)
apply_rate_multiplier(p, "IP_to_IS_rate", m_IP)
apply_rate_multiplier(p, "IS_to_R_rate",  m_ISR)
apply_rate_multiplier(p, "IA_to_R_rate",  m_IAR)

# 3) Reinfection / immunity
apply_rate_multiplier(p, "R_to_S_rate", m_RS)
p.inf_induced_immune_wane     = p.inf_induced_immune_wane * as_like(m_wane,  torch.as_tensor(p.inf_induced_immune_wane))
p.inf_induced_inf_risk_reduce = p.inf_induced_inf_risk_reduce * as_like(m_prot, torch.as_tensor(p.inf_induced_inf_risk_reduce))

with st.spinner("Simulating…"):
    y = simulate_total_admits(base_state, p, base_precomputed, base_schedules, T, timesteps_per_day)

# =========================
# Main pane — wide plot
# =========================
st.title("Aggregate Hospital Admissions")
xs = np.arange(len(y))  # daily

fig, ax = plt.subplots(figsize=(12, 5))  # width here is less critical since we use container width
ax.plot(xs, y, linewidth=2)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Total Daily Hospital Admissions")
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_title("Sum over locations (L), ages (A), and risks (R)")

caption = (
    f"β = [{beta_vec[0]:.4f}, {beta_vec[1]:.4f}, {beta_vec[2]:.4f}]  |  "
    f"E→I×{m_EI:.4f}, IP→IS×{m_IP:.4f}, IS→R×{m_ISR:.4f}, IA→R×{m_IAR:.4f}  |  "
    f"R→S×{m_RS:.4f}, wane×{m_wane:.4f}, protection×{m_prot:.4f}"
)
st.caption(caption)

# Use the full available width for the plot container
st.pyplot(fig, use_container_width=True)
