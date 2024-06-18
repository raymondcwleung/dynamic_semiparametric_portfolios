import jax
import jax.numpy as jnp
import jax.scipy.optimize
from jax import random
import jax.test_util

import numpy as np

import logging
import pathlib


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logs/simulate_dcc_sgt_garch.log",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    filemode="w",
)

import dcc
import sgt

jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod

seed = 1234567
key = random.key(seed)
rng = np.random.default_rng(seed)
num_sample = int(1e2)
dim = 5
num_cores = 8

#################################################################
## Parameters for time-varying SGT
#################################################################
vec_lbda_true = rng.uniform(-0.25, 0.25, dim)
vec_p0_true = rng.uniform(2, 10, dim)
vec_q0_true = rng.uniform(2, 10, dim)

mat_lbda_tvparams_true = rng.uniform(-0.25, 0.25, (dcc.NUM_LBDA_TVPARAMS, dim))
mat_p0_tvparams_true = rng.uniform(-1, 1, (dcc.NUM_P0_TVPARAMS, dim))
mat_q0_tvparams_true = rng.uniform(-1, 1, (dcc.NUM_Q0_TVPARAMS, dim))
# mat_lbda_tvparams_true[0, :] = np.abs(mat_lbda_tvparams_true[0, :])
mat_p0_tvparams_true[0, :] = np.abs(mat_p0_tvparams_true[0, :])
mat_q0_tvparams_true[0, :] = np.abs(mat_q0_tvparams_true[0, :])

vec_z_init_t0 = 2 * jax.random.uniform(key, shape=(dim,)) - 1
vec_z_init_t0 = jnp.repeat(0.0, dim)
vec_lbda_init_t0 = rng.uniform(-0.25, 0.25, dim)
vec_p0_init_t0 = rng.uniform(2, 4, dim)
vec_q0_init_t0 = rng.uniform(2, 4, dim)

params_z_sgt_true = sgt.ParamsZSGT(
    mat_lbda_tvparams=mat_lbda_tvparams_true,
    mat_p0_tvparams=mat_p0_tvparams_true,
    mat_q0_tvparams=mat_q0_tvparams_true,
)

#################################################################
## Parameters for DCC-GARCH
#################################################################
# Parameters for the mean returns vector
params_mean_true = dcc.ParamsMean(vec_mu=rng.uniform(0, 1, dim) / 50)
# dict_params_mean_true = {dcc.VEC_MU: rng.uniform(0, 1, dim) / 50}

# Params for DCC -- univariate vols
# dict_params_dcc_uvar_vol_true = {
#     dcc.VEC_OMEGA: rng.uniform(0, 1, dim) / 2,
#     dcc.VEC_BETA: rng.uniform(0, 1, dim) / 3,
#     dcc.VEC_ALPHA: rng.uniform(0, 1, dim) / 10,
#     dcc.VEC_PSI: rng.uniform(0, 1, dim) / 5,
# }

# Params for DCC -- univariate vols
params_uvar_vol_true = dcc.ParamsUVarVol(
    vec_omega=rng.uniform(0, 1, dim) / 2,
    vec_beta=rng.uniform(0, 1, dim) / 3,
    vec_alpha=rng.uniform(0, 1, dim) / 10,
    vec_psi=rng.uniform(0, 1, dim) / 5,
)


# Params for DCC -- multivariate correlations
# dict_params_dcc_mvar_cor_true = {
#     # Ensure \delta_1, \delta_2 \in [0,1] and \delta_1 + \delta_2 \le 1
#     dcc.VEC_DELTA: np.array([0.007, 0.930]),
#     dcc.MAT_QBAR: dcc.generate_random_cov_mat(key=key, dim=dim) / 5,
# }


# Params for DCC -- multivariate correlations
params_mvar_cor_true = dcc.ParamsMVarCor(
    vec_delta=np.array([0.007, 0.930]),
    mat_Qbar=dcc.generate_random_cov_mat(key=key, dim=dim) / 5,
)

params_dcc_true = dcc.ParamsDCC(
    mean=params_mean_true, uvar_vol=params_uvar_vol_true, mvar_cor=params_mvar_cor_true
)

breakpoint()


dict_params_true = {
    dcc.DICT_PARAMS_MEAN: dict_params_mean_true,
    dcc.DICT_PARAMS_Z: dict_params_z_true,
    dcc.DICT_PARAMS_DCC_UVAR_VOL: dict_params_dcc_uvar_vol_true,
    dcc.DICT_PARAMS_DCC_MVAR_COR: dict_params_dcc_mvar_cor_true,
}

key, _ = random.split(key)
mat_Sigma_init_t0 = dcc.generate_random_cov_mat(key=key, dim=dim)
key, _ = random.split(key)
mat_Q_init_t0 = dcc.generate_random_cov_mat(key=key, dim=dim)

#################################################################
## Simulate DCC-SGT-GARCH
#################################################################
# data_siminnov_savepath = pathlib.Path().resolve() / "data_timevarying_sgt.pkl"
data_siminnov_savepath = None
data_simreturns_savepath = pathlib.Path().resolve() / "data_simreturns.pkl"
dcc.simulate_dcc_sgt_garch(
    key=key,
    dim=dim,
    num_sample=num_sample,
    # SGT parameters
    mat_lbda_tvparams_true=mat_lbda_tvparams_true,
    mat_p0_tvparams_true=mat_p0_tvparams_true,
    mat_q0_tvparams_true=mat_q0_tvparams_true,
    vec_lbda_init_t0=vec_lbda_init_t0,
    vec_p0_init_t0=vec_p0_init_t0,
    vec_q0_init_t0=vec_q0_init_t0,
    vec_z_init_t0=vec_z_init_t0,
    # DCC-GARCH parameters
    dict_params_mean_true=dict_params_mean_true,
    dict_params_dcc_uvar_vol_true=dict_params_dcc_uvar_vol_true,
    dict_params_dcc_mvar_cor_true=dict_params_dcc_mvar_cor_true,
    mat_Sigma_init_t0=mat_Sigma_init_t0,
    mat_Q_init_t0=mat_Q_init_t0,
    # Saving paths
    data_simreturns_savepath=data_simreturns_savepath,
    data_siminnov_savepath=data_siminnov_savepath,
)
