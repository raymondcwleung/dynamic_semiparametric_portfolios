import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.optimize
from jax import grad, jit, vmap
from jax import random
import jax
import jax.test_util
import jaxtyping as jpt

import typing as tp
import pickle
import os
import dataclasses
import chex

import numpy as np
from pandas._libs.tslibs.timedeltas import truediv_object_array

import utils
import innovations
import dcc

from innovations import SimulatedInnovations
from dcc import ModelDcc



num_sample = 100
dim = 2


# Init
lst_params_mean = []
lst_params_uvar_vol = []
lst_params_mvar_cor = []
lst_params_mvar_corQbar = []

# List files
dir_data_simulations = utils.get_simulations_data_dir(num_sample=num_sample, dim=dim)
dir_data_simulations_simreturns = dir_data_simulations.joinpath("./simreturns")
dir_data_simulations_estimations = dir_data_simulations.joinpath("./estimations")

# Load in "simreturns_{num_sample}_{dim}.pkl" file
with open(dir_data_simulations_simreturns.joinpath(f"simreturns_{num_sample}_{dim}.pkl"), "rb") as f:
    simreturns: dcc.SimulatedReturns = pickle.load(f)

# Load in the estimation files
lst_estimation_files = list(dir_data_simulations_estimations.glob("*.pkl"))

for fn in lst_estimation_files:
    with open(fn, "rb") as f:
        estimation_res = pickle.load(f)

    # Extract the true parameters
    params_mean_true : dcc.ParamsMean = simreturns.model_dcc_true.mean
    params_uvar_vol_true: dcc.ParamsUVarVol = simreturns.model_dcc_true.uvar_vol
    params_mvar_cor_true: dcc.ParamsMVarCor = simreturns.model_dcc_true.mvar_cor
    params_mvar_corQbar_true: dcc.ParamsMVarCorQbar = simreturns.model_dcc_true.mvar_corQbar

    # Extract the estimated parameters
    params_mean_est : dcc.ParamsMean = estimation_res.dcc_model.mean
    params_uvar_vol_est: dcc.ParamsUVarVol = estimation_res.dcc_model.uvar_vol
    params_mvar_cor_est: dcc.ParamsMVarCor = estimation_res.dcc_model.mvar_cor
    params_mvar_corQbar_est: dcc.ParamsMVarCorQbar = estimation_res.dcc_model.mvar_corQbar


    # Compute the square difference between the true parameters and the estimated parameters
    res_params_mean = utils.calc_param_squared_difference(
        params_mean_true, params_mean_est, dcc.ParamsMean
    )
    res_params_uvar_vol = utils.calc_param_squared_difference(
        params_uvar_vol_true, params_uvar_vol_est, dcc.ParamsUVarVol
    )
    res_params_mvar_cor = utils.calc_param_squared_difference(
        params_mvar_cor_true, params_mvar_cor_est, dcc.ParamsMVarCor
    )

    # Update
    lst_params_mean.append(res_params_mean)
    lst_params_uvar_vol.append(res_params_uvar_vol)
    lst_params_mvar_cor.append(res_params_mvar_cor)




dict_results_params_mean = utils.calc_param_analytics_summary(lst_params_mean, dcc.ParamsMean)
dict_results_params_uvar_vol = utils.calc_param_analytics_summary(lst_params_uvar_vol, dcc.ParamsUVarVol)
dict_results_params_mvar_cor = utils.calc_param_analytics_summary(lst_params_mvar_cor, dcc.ParamsMVarCor)


