# app.py
from pathlib import Path
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- Dependency guard: PyTorch ----------------
try:
    import torch
except Exception as e:
    st.error(
        "PyTorch failed to import. Ensure your requirements.txt at repo root contains "
        "`torch==2.9.0`, then Manage app → Clear cache and reinstall.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# ---------------- Locate vendored model repo ----------------
APP_DIR = Path(__file__).parent.resolve()
CLT_DIR = APP_DIR / "CLT_BaseModel"   # <- put the model code+data here

if not CLT_DIR.exists():
    st.error(
        "Missing model code/data: `CLT_BaseModel/` not found in the app repository.\n\n"
        "Please vendor the CLT_BaseModel repo (copy the folder or add as a git submodule) so this app "
        "can load `flu_core`, `clt_toolkit`, and the input files in `flu_instances/`."
    )
    st.stop()

if str(CLT_DIR) not in sys.path:
    sys.path.insert(0, str(CLT_DIR))

# ---------------- Import model packages ----------------
try:
    import clt_toolkit as clt
    import flu_core as flu
except Exception as e:
    st.error(
        "Failed to import `clt_toolkit` / `flu_core` from CLT_BaseModel.\n"
        "Verify those modules are present and importable in `CLT_BaseModel/`.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# ---------------- Utility helpers ----------------
def as_like(val, like_tensor):
    return torch.as_tensor(val, dtype=like_tensor.dtype, device=like_tensor.device)

def set_beta_by_location(p, beta_vec):
    """beta_vec length must equal L; broadcast to [L, A, R]."""
    L, A, R = p.beta_baseline.shape
    beta_like = p.beta_baseline
    vec = np.asarray(beta_vec, dtype=float)
    if len(vec) != L:
        raise ValueError(f"beta_by_location length must be {L}, got {len(vec)}")
    new_beta = torch.as_tensor(vec, dtype=beta_like.dtype, device=beta_like.device).view(L, 1, 1).expand(L, A, R)
    p.beta_baseline = new_beta

def apply_rate_multiplier(p, field_name, mult):
    cur = getattr(p, field_name)
    setattr(p, field_name, cur * as_like(mult, cur))

def simulate_total_admits(state, params, precomputed, schedules, T, tpd):
    with torch.no_grad():
        admits = flu.torch_simulate_hospital_admits(state, params, precomputed, schedules, T, tpd)
        # aggregate to daily total series [T]
        return torch.sum(admits, dim=(1, 2, 3)).cpu().numpy()

@st.cache_resource(show_spinner=True)
def load_model_inputs():
    """Load model data & build base tensors. Everything is created here."""
    import pandas as pd
    # ---- timing ----
    T = 180
    timesteps_per_day = 4

    # ---- resolve paths (must exist in vendored repo) ----
    texas_files_path = CLT_DIR / "flu_instances" / "texas_input_files"
    calibration_files_path = CLT_DIR / "flu_instances" / "calibration_research_input_files"

    # JSONs
    subpopA_init_vals_fp = calibration_files_path / "subpopA_init_vals.json"
    subpopB_init_vals_fp = calibration_files_path / "subpopB_init_vals.json"
    subpopC_init_vals_fp = calibration_files_path / "subpopC_init_vals.json"
    common_subpop_params_fp = texas_files_path / "common_subpop_params.json"
    mixing_params_fp = calibration_files_path / "ABC_mixing_params.json"
    simulation_settings_fp = texas_files_path / "simulation_settings.json"

    # CSV schedules
    calendar_df = pd.read_csv(texas_files_path / "school_work_calendar.csv", index_col=0)
    humidity_df = pd.read_csv(texas_files_path / "absolute_humidity_austin_2023_2024.csv", index_col=0)
    vaccines_df = pd.read_csv(texas_files_path / "daily_vaccines_constant.csv", index_col=0)

    schedules_info = flu.FluSubpopSchedules(
        absolute_humidity=humidity_df,
        flu_contact_matrix=calendar_df,
        daily_vaccines=vaccines_df,
    )

    # Build dataclasses
    subpopA_init_vals = clt.make_dataclass_from_json(subpopA_init_vals_fp, flu.FluSubpopState)
    subpopB_init_vals = clt.make_dataclass_from_json(subpopB_init_vals_fp, flu.FluSubpopState)
    subpopC_init_vals = clt.make_dataclass_from_json(subpopC_init_vals_fp, flu.FluSubpopState)

    common_subpop_params = clt.make_dataclass_from_json(common_subpop_params_fp, flu.FluSubpopParams)
    mixing_params = clt.make_dataclass_from_json(mixing_params_fp, flu.FluMixingParams)
    simulation_settings = clt.make_dataclass_from_json(simulation_settings_fp, flu.SimulationSettings)

    # Set tpd (you used 4)
    simulation_settings = clt.updated_dataclass(simulation_settings, {"timesteps_per_day": timesteps_per_day})

    # Provide per-location beta baselines (will be overridden by sliders)
    subpopA_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 1.5})
    subpopB_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 2.5})
    subpopC_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 2.2})

    # Build subpops (seeded RNGs)
    subpopA = flu.FluSubpopModel(subpopA_init_vals, subpopA_params, simulation_settings,
                                 np.random.Generator(np.random.MT19937(111)), schedules_info, name="subpopA")
    subpopB = flu.FluSubpopModel(subpopB_init_vals, subpopB_params, simulation_settings,
                                 np.random.Generator(np.random.MT19937(222)), schedules_info, name="subpopB")
    subpopC = flu.FluSubpopModel(subpopC_init_vals, subpopC_params, simulation_settings,
                                 np.random.Generator(np.random.MT19937(333)), schedules_info, name="subpopC")

    # Metapop wrapper
    flu_demo_model = flu.FluMetapopModel([subpopA, subpopB, subpopC], mixing_params)

    # Torch inputs
    d = flu_demo_model.get_flu_torch_inputs()
    base_state = d["state_tensors"]
    base_params = d["params_tensors"]
    base_schedules = d["schedule_tensors"]
    base_precomputed = d["precomputed"]

    return dict(
        base_state=base_state,
        base_params=base_params,
        base_schedules=base_schedules,
        base_precomputed=base_precomputed,
        T=T,
        timesteps_per_day=timesteps_per_day,
    )

# ===================== UI =====================
st.set_page_config(page_title="Metapop Admissions Explorer", layout="wide")
st.title("Metapop Admissions Explorer (β & Rate Controls)")
st.caption("Self-contained app: loads model, exposes sliders, and plots aggregate hospital admissions.")

ctx = load_model_inputs()
base_state       = ctx["base_state"]
base_params      = ctx["base_params"]
base_schedules   = ctx["base_schedules"]
base_precomputed = ctx["base_precomputed"]
T                = ctx["T"]
timesteps_per_day= ctx["timesteps_per_day"]

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

with st.expander("Model defaults", expanded=False):
    st.write("**Rates (per day):**"); st.json(default_rates)
    st.write("**Immunity / reinfection:**"); st.json(default_immunity)

# β by location (defaults 0.0005, range [0.0001..0.2])
st.markdown("### β by Location")
c0, c1, c2 = st.columns(3)
beta0 = c0.slider("β L0", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta1 = c1.slider("β L1", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta2 = c2.slider("β L2", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta_vec = [beta0, beta1, beta2]

st.markdown("---")
st.markdown("### Rate Multipliers (relative to defaults)")
r1, r2, r3, r4 = st.columns(4)
m_EI  = r1.slider("E→I ×",   0.25, 2.0, 1.0, 0.05)
m_IP  = r2.slider("IP→IS ×", 0.25, 2.0, 1.0, 0.05)
m_ISR = r3.slider("IS→R ×",  0.25, 2.0, 1.0, 0.05)
m_IAR = r4.slider("IA→R ×",  0.25, 2.0, 1.0, 0.05)

st.markdown("---")
st.markdown("### Reinfection / Immunity Multipliers")
i1, i2, i3 = st.columns(3)
m_RS   = i1.slider("R→S × (reinfection)",                0.1, 2.0, 1.0, 0.05)
m_wane = i2.slider("inf_induced_immune_wane ×",          0.1, 2.0, 1.0, 0.05)
m_prot = i3.slider("inf_induced_inf_risk_reduce ×",      0.5, 1.5, 1.0, 0.05)

run = st.button("Run Simulation")

if run:
    # Build a working params tensor pack
    p = copy.deepcopy(base_params)

    # 1) Set β
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

    # 4) Simulate and plot
    y = simulate_total_admits(base_state, p, base_precomputed, base_schedules, T, timesteps_per_day)
    xs = np.arange(len(y))  # daily

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, y, linewidth=2)
    ax.set_title("Aggregate Hospital Admissions (sum over L, A, R)")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Total Daily Hospital Admissions")
    ax.grid(True, linestyle='--', alpha=0.5)

    txt = (
        f"β = [{beta_vec[0]:.4f}, {beta_vec[1]:.4f}, {beta_vec[2]:.4f}]  |  "
        f"E→I×{m_EI:.2f}, IP→IS×{m_IP:.2f}, IS→R×{m_ISR:.2f}, IA→R×{m_IAR:.2f}  |  "
        f"R→S×{m_RS:.2f}, wane×{m_wane:.2f}, protection×{m_prot:.2f}"
    )
    ax.text(0.01, -0.18, txt, transform=ax.transAxes, fontsize=10, va='top', ha='left', wrap=True)

    st.pyplot(fig)
else:
    st.info("Adjust sliders and click **Run Simulation**.")
