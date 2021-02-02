# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:13:36 2021

@author: cwgeo
"""

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.utils import surface2ind_topo
from SimPEG import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    utils,
)
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static import induced_polarization as ip
from SimPEG.electromagnetics.static.utils.static_utils import plot_pseudoSection

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

# path to the directory containing our data
data_path = Path('data')

# files to work with
topo_filename = Path(data_path / "rv_dipole_dipole_topo.txt")
dc_data_filename = Path(data_path / "rv_dipole_dipole_dc.txt")
ip_data_filename = Path(data_path / "rv_dipole_dipole_ip.txt")

# Load data
topo_xz = np.loadtxt(topo_filename)
dobs_dc = np.loadtxt(dc_data_filename)
dobs_ip = np.loadtxt(ip_data_filename)

# sort topo data by first column
topo_xz = topo_xz[list(np.argsort(topo_xz[:, 0])), :]

# Extract source and receiver electrode locations and the observed data
#  we are assuming the dc and ip tx-rx pairs are the same. In this case they are.
# get electrode elevations
A_electrodes = np.c_[dobs_dc[:, 0], np.interp(dobs_dc[:, 0], topo_xz[:, 0], 
                                              topo_xz[:, 1])]
B_electrodes = np.c_[dobs_dc[:, 1], np.interp(dobs_dc[:, 1], topo_xz[:, 0], 
                                              topo_xz[:, 1])]
M_electrodes = np.c_[dobs_dc[:, 2], np.interp(dobs_dc[:, 2], topo_xz[:, 0], 
                                              topo_xz[:, 1])]
N_electrodes = np.c_[dobs_dc[:, 3], np.interp(dobs_dc[:, 3], topo_xz[:, 0], 
                                              topo_xz[:, 1])]
dobs_dc = dobs_dc[:, -1]
dobs_ip = dobs_ip[:, -1]

# Define survey
unique_tx, k = np.unique(np.c_[A_electrodes, B_electrodes], axis=0, return_index=True)
n_sources = len(k)
k = np.r_[k, len(A_electrodes) + 1]

source_list = []
for ii in range(0, n_sources):

    # MN electrode locations for receivers. Each is an (N, 3) numpy array
    M_locations = M_electrodes[k[ii] : k[ii + 1], :]
    N_locations = N_electrodes[k[ii] : k[ii + 1], :]
    receiver_list = [dc.receivers.Dipole(M_locations, N_locations, data_type="volt")]

    # AB electrode locations for source. Each is a (1, 3) numpy array
    A_location = A_electrodes[k[ii], :]
    B_location = B_electrodes[k[ii], :]
    source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))

# Define survey
dc_survey = dc.survey.Survey_ky(source_list)
ip_survey = ip.from_dc_to_ip_survey(dc_survey, dim="2.5D")

# Define the a data object. Uncertainties are added later
dc_data = data.Data(dc_survey, dobs=dobs_dc)
ip_data = data.Data(ip_survey, dobs=dobs_ip)

# Plot apparent resistivity using pseudo-section
mpl.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(11, 9))

ax1 = fig.add_axes([0.05, 0.55, 0.8, 0.45])
plot_pseudoSection(
    dc_data,
    ax=ax1,
    survey_type="pole-dipole",
    data_type="appResistivity",
    space_type="half-space",
    scale="log",
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Apparent Resistivity [Ohm m]")

ax2 = fig.add_axes([0.05, 0.05, 0.8, 0.45])
plot_pseudoSection(
    ip_data,
    ax=ax2,
    survey_type="pole-dipole",
    data_type="appChargeability",
    space_type="half-space",
    scale="linear",
    pcolorOpts={"cmap": "plasma"},
)
ax2.set_title("Apparent Chargeability (mV/V)")

plt.show()

# Compute standard deviations
std_dc = 0.05 * np.abs(dobs_dc)
std_ip = 0.01 * np.abs(dobs_dc)

# Add standard deviations to data object
dc_data.standard_deviation = std_dc
ip_data.standard_deviation = std_ip

# Mesh

core_x_min = 0
core_x_max = 1900
core_x_width = core_x_max - core_x_min
delta_z = 0
core_z_min = -750
core_z_max = delta_z
core_z_width = core_z_max - core_z_min

dh = 12.5  # base cell width 25m / 4
x_min = -1500.  # domain width x
x_max = 3400.
dom_width_x = x_max - x_min
z_max = 0.
z_min = -1800.
dom_width_z = z_max - z_min
# num. base cells x and z
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))

actual_x_width = nbcx*dh
actual_z_width = nbcz*dh
x_origin = core_x_min - (actual_x_width - core_x_width)/2
z_origin = -1*actual_z_width + delta_z

# Define the base mesh
hx = [(dh, nbcx)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hz], x0=[x_origin, z_origin])
# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, topo_xz, octree_levels=[1], method="surface", finalize=False
)

# Mesh refinement near transmitters and receivers
electrode_locations = np.r_[
    dc_survey.locations_a,
    dc_survey.locations_b,
    dc_survey.locations_m,
    dc_survey.locations_n,
]

unique_locations = np.unique(electrode_locations, axis=0)

mesh = refine_tree_xyz(
    mesh, unique_locations, octree_levels=[2, 4], method="radial",
    finalize=False)

# Refine core mesh region
xp, zp = np.meshgrid([core_x_min, core_x_max], [core_z_min, core_z_max])
xyz = np.c_[mkvc(xp), mkvc(zp)]
mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 2], method="box",
                       finalize=False)

mesh.finalize()

# Find cells that lie below surface topography
ind_active = surface2ind_topo(mesh, topo_xz)

# Shift electrodes to the surface of discretized topography
dc_survey.drape_electrodes_on_topography(mesh, ind_active, option="top")
ip_survey.drape_electrodes_on_topography(mesh, ind_active, option="top")

# Define conductivity model in S/m (or resistivity model in Ohm m)
air_conductivity = np.log(1e-8)
background_conductivity = np.log(1e-2)

active_map = maps.InjectActiveCells(mesh, ind_active, np.exp(air_conductivity))
nC = int(ind_active.sum())

conductivity_map = active_map * maps.ExpMap()

# Define model
starting_conductivity_model = background_conductivity * np.ones(nC)

plotting_map = maps.ActiveCells(mesh, ind_active, np.nan)

# Plot Starting Model
fig = plt.figure(figsize=(9, 4))

# Make conductivities in log10
starting_conductivity_model_log10 = np.log10(
    np.exp(starting_conductivity_model))

ax1 = fig.add_axes([0.1, 0.12, 0.72, 0.8])
model_plot = mesh.plot_image(
    plotting_map * starting_conductivity_model_log10,
    ax=ax1,
    grid=False,
    clim=(-4, 4),
    pcolorOpts={"cmap": "viridis"}
)
ax1.set_title("Starting Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([core_x_min, core_x_max])
ax1.set_ylim([core_z_min, core_z_max])
plt.show()



# Define the problem. Define the cells below topography and the mapping
dc_simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=dc_survey, sigmaMap=conductivity_map, Solver=Solver
)

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# Within the data misfit, the residual between predicted and observed data are
# normalized by the data's standard deviation.
dc_data_misfit = data_misfit.L2DataMisfit(data=dc_data, simulation=dc_simulation)

# Define the regularization (model objective function)
dc_regularization = regularization.Simple(
    mesh,
    indActive=ind_active,
    mref=starting_conductivity_model,
    alpha_s=0.01,
    alpha_x=1,
    alpha_y=1,
)

# Define how the optimization problem is solved. Here we will use a projected
# Gauss-Newton approach that employs the conjugate gradient solver.
dc_optimization = optimization.ProjectedGNCG(
    maxIter=20, lower=-10.0, upper=10.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
dc_inverse_problem = inverse_problem.BaseInvProblem(
    dc_data_misfit, dc_regularization, dc_optimization
)

# Apply and update sensitivity weighting as the model updates
update_sensitivity_weighting = directives.UpdateSensitivityWeights()

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)

# Set the rate of reduction in trade-off parameter (beta) each time the
# the inverse problem is solved. And set the number of Gauss-Newton iterations
# for each trade-off paramter value.
beta_schedule = directives.BetaSchedule(coolingFactor=2, coolingRate=1)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

directives_list = [
    update_sensitivity_weighting,
    starting_beta,
    beta_schedule,
    save_iteration,
    target_misfit,
]

# Here we combine the inverse problem and the set of directives
dc_inversion = inversion.BaseInversion(
    dc_inverse_problem, directiveList=directives_list
)

# Run inversion
recovered_conductivity_model = dc_inversion.run(starting_conductivity_model)

# Plot Recovered Model
fig = plt.figure(figsize=(9, 4))

# Make conductivities in log10
recovered_conductivity_model_log10 = np.log10(
    np.exp(recovered_conductivity_model))

plotting_map = maps.ActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.12, 0.72, 0.8])
mesh.plot_image(
    plotting_map * recovered_conductivity_model_log10,
    ax=ax1,
    grid=False,
    clim=(np.min(recovered_conductivity_model_log10),
          np.max(recovered_conductivity_model_log10)),
    pcolorOpts={"cmap": "viridis"}
)
ax1.set_title("Recovered Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([core_x_min, core_x_max])
ax1.set_ylim([core_z_min, core_z_max])
plt.show()

ax2 = fig.add_axes([0.83, 0.12, 0.05, 0.8])
norm = mpl.colors.Normalize(
    vmin=np.min(recovered_conductivity_model_log10),
    vmax=np.max(recovered_conductivity_model_log10),
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="10^%.1f"
)
cbar.set_label("$S/m$", rotation=270, labelpad=15, size=12)

plt.show()