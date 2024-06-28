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
import sgt
import dcc



num_sample = 500
dim = 2




# Init
lst_params_z_sgt = []
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
    params_z_sgt_true: sgt.ParamsZSgt = simreturns.siminnov.params_z_sgt
    params_uvar_vol_true: dcc.ParamsUVarVol = simreturns.params_dcc_true.uvar_vol
    params_mvar_cor_true: dcc.ParamsMVarCor = simreturns.params_dcc_true.mvar_cor

    # Extract the estimated parameters
    params_z_sgt_est: sgt.ParamsZSgt = estimation_res.params_dcc_sgt_garch.sgt
    params_uvar_vol_est: dcc.ParamsUVarVol = (
        estimation_res.params_dcc_sgt_garch.dcc.uvar_vol
    )
    params_mvar_cor_est: dcc.ParamsUVarVol = (
        estimation_res.params_dcc_sgt_garch.dcc.mvar_cor
    )

    # Compute the square difference between the true parameters and the estimated parameters
    res_params_z_sgt = utils.calc_param_squared_difference(
        params_z_sgt_true, params_z_sgt_est, sgt.ParamsZSgt
    )
    res_params_uvar_vol = utils.calc_param_squared_difference(
        params_uvar_vol_true, params_uvar_vol_est, dcc.ParamsUVarVol
    )
    res_params_mvar_cor = utils.calc_param_squared_difference(
        params_mvar_cor_true, params_mvar_cor_est, dcc.ParamsMVarCor
    )

    # Update
    lst_params_z_sgt.append(res_params_z_sgt)
    lst_params_uvar_vol.append(res_params_uvar_vol)
    lst_params_mvar_cor.append(res_params_mvar_cor)



dict_results_params_z_sgt = utils.calc_param_analytics_summary(lst_params_z_sgt, sgt.ParamsZSgt)
dict_results_params_uvar_vol = utils.calc_param_analytics_summary(lst_params_uvar_vol, dcc.ParamsUVarVol)
dict_results_params_mvar_cor = utils.calc_param_analytics_summary(lst_params_mvar_cor, dcc.ParamsMVarCor)


