# -*- coding: utf-8 -*-
"""
Scripts for Rendez-vous 2021 discussion about noise estimates

@author: Sean Walker
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import pickle

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

try:
    from pymatsolver.direct import Pardiso as Solver
    print('Using Paradiso Solver')
except ImportError:
    print('Using LU Solver')
    from SimPEG import SolverLU as Solver


def setup_and_run_std_inv(mesh, dc_survey, dc_data, std_dc, conductivity_map,
                          ind_active, starting_conductivity_model):
    """Code to setup and run a standard inversion.

    Parameters
    ----------
    mesh : TYPE
        DESCRIPTION.
    dc_survey : TYPE
        DESCRIPTION.
    dc_data : TYPE
        DESCRIPTION.
    std_dc : TYPE
        DESCRIPTION.
    conductivity_map : TYPE
        DESCRIPTION.
    ind_active : TYPE
        DESCRIPTION.
    starting_conductivity_model : TYPE
        DESCRIPTION.

    Returns
    -------
    save_iteration : TYPE
        DESCRIPTION.
    save_dict_iteration : TYPE
        DESCRIPTION.
    """
    # Add standard deviations to data object
    dc_data.standard_deviation = std_dc

    # Define the simulation (physics of the problem)
    dc_simulation = dc.simulation_2d.Simulation2DNodal(
        mesh, survey=dc_survey, sigmaMap=conductivity_map, Solver=Solver
    )

    # Define the data misfit.
    dc_data_misfit = data_misfit.L2DataMisfit(data=dc_data,
                                              simulation=dc_simulation)

    # Define the regularization (model objective function)
    dc_regularization = regularization.Simple(
        mesh,
        indActive=ind_active,
        mref=starting_conductivity_model,
        alpha_s=0.01,
        alpha_x=1,
        alpha_y=1
    )

    # Define how the optimization problem is solved. Here we will use a
    # projected. Gauss-Newton approach that employs the conjugate gradient
    # solver.
    dc_optimization = optimization.ProjectedGNCG(
        maxIter=15, lower=-np.inf, upper=np.inf, maxIterLS=20, maxIterCG=10,
        tolCG=1e-3
    )

    # Here we define the inverse problem that is to be solved
    dc_inverse_problem = inverse_problem.BaseInvProblem(
        dc_data_misfit, dc_regularization, dc_optimization
    )

    # Define inversion directives

    # Apply and update sensitivity weighting as the model updates
    update_sensitivity_weighting = directives.UpdateSensitivityWeights()

    # Defining a starting value for the trade-off parameter (beta) between the
    # data misfit and the regularization.
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e2)

    # Set the rate of reduction in trade-off parameter (beta) each time the
    # the inverse problem is solved. And set the number of Gauss-Newton
    # iterations for each trade-off paramter value.
    beta_schedule = directives.BetaSchedule(coolingFactor=10, coolingRate=1)

    # Options for outputting recovered models and predicted data for each beta.
    save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

    # save results from each iteration in a dict
    save_dict_iteration = directives.SaveOutputDictEveryIteration(
        saveOnDisk=False)

    directives_list = [
        update_sensitivity_weighting,
        starting_beta,
        beta_schedule,
        save_iteration,
        save_dict_iteration,
    ]

    # Here we combine the inverse problem and the set of directives
    dc_inversion = inversion.BaseInversion(
        dc_inverse_problem, directiveList=directives_list
    )

    # Run inversion
    _ = dc_inversion.run(starting_conductivity_model)

    return save_iteration, save_dict_iteration


def plot_inv_result(it, mesh, mrec, plotting_map, dpred, dobs, std_err,
                    core_defn=None, locations=None, clim=None, grid=False,
                    ):
    phid = (dobs - dpred)/std_err
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[1, :-1])
    plot_model(mesh,
               mrec,
               plotting_map,
               core_defn=core_defn,
               locations=locations,
               clim=clim,
               fig_ax=(fig, ax1),
               grid=grid,
               title=f"Recovered Model. Iteration:{it}",
               cbar_title="log10 Conductivity (S/m)")
    d_phid = ax2.scatter(mid_x, mid_z, c=phid_it)
    c_phid = fig.colorbar(d_phid, ax=ax2, orientation='vertical')
    ax2.set_xlim(core_x_min, core_x_max)
    ax2.set_ylim(core_z_min, core_z_max)
    ax2.axis('equal')
    ax2.set_title(f"Data misfit {it}")
    ax2.set_xlabel('X')
    ax2.set_ylabel('a (50m) x n')
    ax3 = fig.add_subplot(gs[1,-1])
    h_dobs = ax3.hist(phid_it, bins=25)
    ax3.set_title('$\phi_d$')
    plt.show()


    return


def plot_model(mesh, model, plotting_map, core_defn=None, locations=None,
               clim=None, fig_ax=None, grid=False, title='Model',
               cbar_title='Units'):

    if fig_ax is None:
        fig = plt.figure(figsize=(16, 4))
        gs = GridSpec(1, 4, figure=fig)
        ax = fig.add_subplot(gs[:, :-1])
    else:
        fig, ax = fig_ax
    if clim is None:
        clim = (model.min(), model.max())
    mod = mesh.plot_image(
        plotting_map * model,
        grid=grid,
        clim=clim,
        ax=ax,
        pcolorOpts={"cmap": "viridis"},
    )
    if locations is not None:
        ax.plot(locations[:, 0], locations[:, 1], 'ko')
    if core_defn is not None:
        ax.set_xlim(core_defn[0], core_defn[1])
        ax.set_ylim(core_defn[2], core_defn[3])
    ax.set_title(title)
    fig.colorbar(mod[0], ax=ax, orientation='vertical', label=cbar_title)
    if fig_ax is None:
        plt.show()
    return fig, ax


def plot_data(xy_data, data_lim=None, fig_ax=None, data_title='Data',
              x_label='X', y_label='Z', cbar_title='Units', bins=25,
              hist_title='Histogram'):

    if fig_ax is None:
        fig = plt.figure(figsize=(16, 4))
        gs = GridSpec(1, 4, figure=fig)
        ax1 = fig.add_subplot(gs[:, : -1])
        ax2 = fig.add_subplot(gs[:, -1])
    else:
        fig, (ax1, ax2) = fig_ax
    if data_lim is None:
        vmin, vmax = np.min(xy_data[:, 2]), np.max(xy_data[:, 2])
    else:
        vmin, vmax = data_lim[0], data_lim[1]
    d_res = ax1.scatter(xy_data[:, 0], xy_data[:, 1], c=xy_data[:, 2],
                        vmin=vmin, vmax=vmax)
    fig.colorbar(d_res, ax=ax1, orientation='vertical', label=cbar_title)
    ax1.axis('equal')
    ax1.set_title(data_title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    ax2.hist(xy_data[:, 2], bins=bins, range=(vmin, vmax))
    ax2.set_title(hist_title)
    if fig_ax is None:
        plt.show()
    return fig, (ax1, ax2)

    return


def plot_hist():
    pass

    return


def plot_conv_curve():
    # num_it = len(out_5_per.phi_d)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.semilogy(np.linspace(1, num_it, num_it), out_5_per.phi_d,'ko-')
    # ax2.plot(np.linspace(1, num_it, num_it), out_5_per.phi_m,'ko-')
    pass
    return


def save_results_dict(results_dict, pickle_filename):
    """
    Pickle dict full of results.

    Parameters
    ----------
    results_dict : dict
        DESCRIPTION. Dict holding results we want to save for later
    pickle_filename : str
        DESCRIPTION. String with the filename we are storing results in

    Returns
    -------
    None.

    """
    with open(pickle_filename, 'wb') as f:
        # Pickle the results
        pickle.dump(results_dict, f, pickle.DEFAULT_PROTOCOL)
    return


def load_results_dict(pickle_filename):
    """
    Load results in a dict that have been pickled.

    Parameters
    ----------
    pickle_filename : str
        DESCRIPTION. String with the filename we are loading results from

    Returns
    -------
    results_dict : dict
        DESCRIPTION. Dict holding results we want to analyze.

    """
    with open(pickle_filename, 'rb') as f:
        results_dict = pickle.load(f)
    return results_dict
