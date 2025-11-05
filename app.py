# app.py
from pathlib import Path
import sys
import subprocess
import copy
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Metapop Admissions Explorer", layout="wide")

# ----- Torch guard -----
try:
    import torch
except Exception as e:
    st.error(
        "PyTorch failed to import. Ensure `requirements.txt` pins `torch==2.9.0`, "
        "then Manage app → Clear cache & reboot.\n\n"
        f"Import error: {e}"
    )
    st.stop()

APP_DIR = Path(__file__).parent.resolve()
CLT_DIR = APP_DIR / "CLT_BaseModel"

def ensure_clt_repo():
    if CLT_DIR.exists():
        return True
    try:
        st.info("CLT_BaseModel not found. Cloning…")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/LP-relaxation/CLT_BaseModel.git", str(CLT_DIR)],
            check=True, capture_output=True, text=True
        )
        st.success("CLT_BaseModel cloned.")
        return True
    except Exception as e:
        st.error(
            "Failed to clone CLT_BaseModel. Vendor the folder at `CLT_BaseModel/` and redeploy.\n\n"
            f"Details: {e}"
        )
        return False

if not ensure_clt_repo():
    st.stop()

if str(CLT_DIR) not in sys.path:
    sys.path.insert(0, str(CLT_DIR))

try:
    import clt_toolkit as clt
    import flu_core as flu
except Exception as e:
    st.error(
        "Could not import `clt_toolkit`/`flu_core` from CLT_BaseModel. "
        "Verify the repo structure and try again.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# ===== Helpers =====
def as_like(val, like_tensor):
    return torch.as_tensor(val, dtype=like_tensor.dtype, device=like_tensor.device)

def set_beta_by_location(p, beta_vec):
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

def apply_prob_multiplier_clip01(p, field_name, mult):
    """For fields that represent probabilities/reductions in [0,1].
       Works for scalars or A×R arrays; preserves shape and clips to [0,1]."""
    cur = getattr(p, field_name)
    t = torch.as_tensor(cur, dtype=torch.as_tensor(cur).dtype)
    new_val = (t * float(mult)).clamp(0.0, 1.0)
    # Keep device/dtype of original param tensor in p if it is a torch tensor
    like = torch.as_tensor(cur, dtype=new_val.dtype)
    new_val = new_val.to(like.dtype)
    setattr(p, field_name, new_val)

def simulate_total_admits(state, params, precomputed, schedules, T, tpd):
    with torch.no_grad():
        admits = flu.torch_simulate_hospital_admits(state, params, precomputed, schedules, T, tpd)
        return torch.sum(admits, dim=(1, 2, 3)).cpu().numpy()

# ===== Load model inputs =====
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

    # seed betas (will be overridden by sliders)
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

# ===== Sidebar controls (scrollable) =====
st.sidebar.title("Controls")

L, A, R = base_params.beta_baseline.shape

# Show defaults
default_rates = {
    "E_to_I_rate":    float(base_params.E_to_I_rate),
    "IP_to_IS_rate":  float(base_params.IP_to_IS_rate),
    "IS_to_R_rate":   float(base_params.IS_to_R_rate),
    "IA_to_R_rate":   float(base_params.IA_to_R_rate),
    "IS_to_H_rate":   float(base_params.IS_to_H_rate),
    "H_to_R_rate":    float(base_params.H_to_R_rate),
    "H_to_D_rate":    float(base_params.H_to_D_rate),
}
default_split_inf = {
    "E_to_IA_prop":   float(torch.as_tensor(base_params.E_to_IA_prop).mean().item()),
    "IP_relative_inf": float(base_params.IP_relative_inf),
    "IA_relative_inf": float(base_params.IA_relative_inf),
}
default_immunity = {
    "R_to_S_rate":                     float(base_params.R_to_S_rate),
    "inf_induced_immune_wane":         float(torch.as_tensor(base_params.inf_induced_immune_wane)),
    "vax_induced_immune_wane":         float(torch.as_tensor(base_params.vax_induced_immune_wane)),
    "inf_induced_inf_risk_reduce":     float(torch.as_tensor(base_params.inf_induced_inf_risk_reduce).mean().item()),
    "vax_induced_inf_risk_reduce":     float(torch.as_tensor(base_params.vax_induced_inf_risk_reduce).mean().item()),
    "inf_induced_hosp_risk_reduce":    float(torch.as_tensor(base_params.inf_induced_hosp_risk_reduce).mean().item()),
    "inf_induced_death_risk_reduce":   float(torch.as_tensor(base_params.inf_induced_death_risk_reduce).mean().item()),
    "vax_induced_hosp_risk_reduce":    float(torch.as_tensor(base_params.vax_induced_hosp_risk_reduce).mean().item()),
    "vax_induced_death_risk_reduce":   float(torch.as_tensor(base_params.vax_induced_death_risk_reduce).mean().item()),
}

with st.sidebar.expander("Model defaults (from loaded tensors)", expanded=False):
    st.write("**Rates (per day):**"); st.json(default_rates)
    st.write("**Split/Relative Inf:**"); st.json(default_split_inf)
    st.write("**Immunity / Risk Reductions:**"); st.json(default_immunity)

st.sidebar.markdown("### β by Location")
beta0 = st.sidebar.slider("β L0", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta1 = st.sidebar.slider("β L1", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta2 = st.sidebar.slider("β L2", min_value=0.0001, max_value=0.2, value=0.0005, step=0.0001, format="%.4f")
beta_vec = [beta0, beta1, beta2]

st.sidebar.markdown("---")
st.sidebar.markdown("### Infectious Flow Rate Multipliers (0.0001×–5×)")
# Existing
m_EI   = st.sidebar.slider("E→I ×",    0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_IP   = st.sidebar.slider("IP→IS ×",  0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_ISR  = st.sidebar.slider("IS→R ×",   0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_IAR  = st.sidebar.slider("IA→R ×",   0.0001, 5.0, 1.0, 0.01, format="%.4f")
# New: hospital flow
m_ISH  = st.sidebar.slider("IS→H ×",   0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_HR   = st.sidebar.slider("H→R ×",    0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_HD   = st.sidebar.slider("H→D ×",    0.0001, 5.0, 1.0, 0.01, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("### Split / Infectiousness Multipliers")
# E_to_IA_prop is a proportion; scale then clip [0,1]
m_EIAp = st.sidebar.slider("E→IA proportion × (clip 0–1)", 0.0001, 5.0, 1.0, 0.01, format="%.4f")
# Relative infectiousness (>0)
m_IPinf = st.sidebar.slider("IP relative infectiousness ×", 0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_IAinf = st.sidebar.slider("IA relative infectiousness ×", 0.0001, 5.0, 1.0, 0.01, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("### Reinfection / Immunity Multipliers")
m_RS   = st.sidebar.slider("R→S × (reinfection)",             0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_wane_inf = st.sidebar.slider("inf_induced_immune_wane ×",   0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_wane_vax = st.sidebar.slider("vax_induced_immune_wane ×",   0.0001, 5.0, 1.0, 0.01, format="%.4f")

st.sidebar.markdown("#### Risk Reduction Multipliers (clip 0–1)")
m_inf_inf   = st.sidebar.slider("inf_induced_inf_risk_reduce ×",    0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_vax_inf   = st.sidebar.slider("vax_induced_inf_risk_reduce ×",    0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_inf_hosp  = st.sidebar.slider("inf_induced_hosp_risk_reduce ×",   0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_inf_death = st.sidebar.slider("inf_induced_death_risk_reduce ×",  0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_vax_hosp  = st.sidebar.slider("vax_induced_hosp_risk_reduce ×",   0.0001, 5.0, 1.0, 0.01, format="%.4f")
m_vax_death = st.sidebar.slider("vax_induced_death_risk_reduce ×",  0.0001, 5.0, 1.0, 0.01, format="%.4f")

# ===== Build params from sliders & simulate (reactive) =====
p = copy.deepcopy(base_params)

# 1) β by location
set_beta_by_location(p, beta_vec)

# 2) Rate multipliers (infectious + hospital flows)
apply_rate_multiplier(p, "E_to_I_rate",   m_EI)
apply_rate_multiplier(p, "IP_to_IS_rate", m_IP)
apply_rate_multiplier(p, "IS_to_R_rate",  m_ISR)
apply_rate_multiplier(p, "IA_to_R_rate",  m_IAR)
apply_rate_multiplier(p, "IS_to_H_rate",  m_ISH)
apply_rate_multiplier(p, "H_to_R_rate",   m_HR)
apply_rate_multiplier(p, "H_to_D_rate",   m_HD)

# 3) Split / infectiousness
apply_prob_multiplier_clip01(p, "E_to_IA_prop", m_EIAp)
p.IP_relative_inf = p.IP_relative_inf * as_like(m_IPinf,  torch.as_tensor(p.IP_relative_inf))
p.IA_relative_inf = p.IA_relative_inf * as_like(m_IAinf,  torch.as_tensor(p.IA_relative_inf))

# 4) Reinfection / immunity
apply_rate_multiplier(p, "R_to_S_rate", m_RS)
p.inf_induced_immune_wane = p.inf_induced_immune_wane * as_like(m_wane_inf, torch.as_tensor(p.inf_induced_immune_wane))
p.vax_induced_immune_wane = p.vax_induced_immune_wane * as_like(m_wane_vax, torch.as_tensor(p.vax_induced_immune_wane))

# 5) Risk reductions (clip to [0,1])
apply_prob_multiplier_clip01(p, "inf_induced_inf_risk_reduce",   m_inf_inf)
apply_prob_multiplier_clip01(p, "vax_induced_inf_risk_reduce",   m_vax_inf)
apply_prob_multiplier_clip01(p, "inf_induced_hosp_risk_reduce",  m_inf_hosp)
apply_prob_multiplier_clip01(p, "inf_induced_death_risk_reduce", m_inf_death)
apply_prob_multiplier_clip01(p, "vax_induced_hosp_risk_reduce",  m_vax_hosp)
apply_prob_multiplier_clip01(p, "vax_induced_death_risk_reduce", m_vax_death)

with st.spinner("Simulating…"):
    y = simulate_total_admits(base_state, p, base_precomputed, base_schedules, T, timesteps_per_day)

# ===== Main pane — big plot =====
st.title("Aggregate Hospital Admissions (Sum over L, A, R)")
xs = np.arange(len(y))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(xs, y, linewidth=2)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Total Daily Hospital Admissions")
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_title("Reactive plot — adjusts instantly with sliders")

caption = (
    f"β = [{beta_vec[0]:.4f}, {beta_vec[1]:.4f}, {beta_vec[2]:.4f}]  |  "
    f"E→I×{m_EI:.3f}, IP→IS×{m_IP:.3f}, IS→R×{m_ISR:.3f}, IA→R×{m_IAR:.3f}, "
    f"IS→H×{m_ISH:.3f}, H→R×{m_HR:.3f}, H→D×{m_HD:.3f}  |  "
    f"E→IA prop×{m_EIAp:.3f}, IP_inf×{m_IPinf:.3f}, IA_inf×{m_IAinf:.3f}  |  "
    f"R→S×{m_RS:.3f}, inf_wane×{m_wane_inf:.3f}, vax_wane×{m_wane_vax:.3f}  |  "
    f"risk↓ (inf:{m_inf_inf:.3f}, vax:{m_vax_inf:.3f}, hosp_inf:{m_inf_hosp:.3f}, "
    f"death_inf:{m_inf_death:.3f}, hosp_vax:{m_vax_hosp:.3f}, death_vax:{m_vax_death:.3f})"
)
st.caption(caption)
st.pyplot(fig, use_container_width=True)
