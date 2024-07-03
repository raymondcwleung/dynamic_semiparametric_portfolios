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



num_sample = 500
dim = 2




# Init
lst_params_mean = []
lst_params_uvar_vol = []
lst_params_mvar_cor = []

# List files
dir = utils.get_simulations_data_dir(num_sample=num_sample, dim=dim)
lst_files = dir.glob("*.pkl")

# Need to chop out the "simreturns_{num_sample}_{dim}.pkl" file
simreturns_file = dir.glob("simreturns_*.pkl")
lst_files = list(set(lst_files) - set(simreturns_file))

for fn in lst_files:
    with open(fn, "rb") as f:
        _ = pickle.load(f)

    simreturns = _["simreturns"]
    estimation_res = _["estimation_res"]


    # FIX: Need to put in mean parameters here

    # Extract the true parameters
    params_mean_true : dcc.ParamsMean = simreturns.model_dcc_true.mean
    params_uvar_vol_true: dcc.ParamsUVarVol = simreturns.model_dcc_true.uvar_vol
    params_mvar_cor_true: dcc.ParamsMVarCor = simreturns.model_dcc_true.mvar_cor

    # Extract the estimated parameters
    params_mean_est : dcc.ParamsMean = estimation_res.dcc_model.mean
    params_uvar_vol_est: dcc.ParamsUVarVol = estimation_res.dcc_model.uvar_vol
    params_mvar_cor_est: dcc.ParamsUVarVol = estimation_res.dcc_model.mvar_cor

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


