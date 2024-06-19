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
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/simulate_dcc_sgt_garch.log", mode="w"),
        logging.StreamHandler(),
    ],
)

import dcc
import sgt

jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod

seed = 1234567
key = random.key(seed)
rng = np.random.default_rng(seed)
num_sample = int(3e2)
dim = 5
num_cores = 8

#################################################################
## Parameters for time-varying SGT
#################################################################
# Set the true parameters of the SGT z_t process
mat_lbda_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_LBDA_TVPARAMS, dim))
mat_lbda_tvparams[0, :] = np.abs(mat_lbda_tvparams[0, :])
mat_p0_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_P0_TVPARAMS, dim))
mat_p0_tvparams[0, :] = np.abs(mat_p0_tvparams[0, :])
mat_q0_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_Q0_TVPARAMS, dim))
mat_q0_tvparams[0, :] = np.abs(mat_q0_tvparams[0, :])
params_z_sgt_true = sgt.ParamsZSgt(
    mat_lbda_tvparams=jnp.array(mat_lbda_tvparams),
    mat_p0_tvparams=jnp.array(mat_p0_tvparams),
    mat_q0_tvparams=jnp.array(mat_q0_tvparams),
)

# Set the initial t = 0 conditions for the various processes
# in constructing time-varying parameters
inittimecond_z_sgt = sgt.InitTimeConditionZSgt(
    vec_z_init_t0=jnp.repeat(0.0, dim),
    vec_lbda_init_t0=jnp.array(rng.uniform(-0.25, 0.25, dim)),
    vec_p0_init_t0=jnp.array(rng.uniform(2, 4, dim)),
    vec_q0_init_t0=jnp.array(rng.uniform(2, 4, dim)),
)

#################################################################
## Parameters for DCC-GARCH
#################################################################
# Parameters for the mean returns vector
params_mean_true = dcc.ParamsMean(vec_mu=jnp.array(rng.uniform(0, 1, dim) / 50))

# Params for DCC -- univariate vols
params_uvar_vol_true = dcc.ParamsUVarVol(
    vec_omega=jnp.array(rng.uniform(0, 1, dim) / 2),
    vec_beta=jnp.array(rng.uniform(0, 1, dim) / 3),
    vec_alpha=jnp.array(rng.uniform(0, 1, dim) / 10),
    vec_psi=jnp.array(rng.uniform(0, 1, dim) / 5),
)
# Params for DCC -- multivariate Q
params_mvar_cor_true = dcc.ParamsMVarCor(
    vec_delta=jnp.array([0.007, 0.930]),
    mat_Qbar=dcc.generate_random_cov_mat(key=key, dim=dim) / 5,
)

# Package all the DCC params together
params_dcc_true = dcc.ParamsDcc(
    mean=params_mean_true, uvar_vol=params_uvar_vol_true, mvar_cor=params_mvar_cor_true
)


# Initial t = 0 conditions for the DCC Q_t process
subkeys = random.split(key, 2)
mat_Sigma_init_t0 = dcc.generate_random_cov_mat(key=subkeys[0], dim=dim)
mat_Q_init_t0 = dcc.generate_random_cov_mat(key=subkeys[1], dim=dim)
inittimecond_dcc = dcc.InitTimeConditionDcc(
    mat_Sigma_init_t0=mat_Sigma_init_t0, mat_Q_init_t0=mat_Q_init_t0
)

#################################################################
## Simulate DCC-SGT-GARCH
#################################################################
data_simreturns_savepath = (
    pathlib.Path().resolve() / "simulated_data/data_simreturns_timevarying_sgt.pkl"
)
dcc.simulate_dcc_sgt_garch(
    key=key,
    dim=dim,
    num_sample=num_sample,
    # SGT parameters
    params_z_sgt_true=params_z_sgt_true,
    inittimecond_z_sgt=inittimecond_z_sgt,
    # DCC-GARCH parameters
    params_dcc_true=params_dcc_true,
    inittimecond_dcc=inittimecond_dcc,
    # Saving paths
    data_simreturns_savepath=data_simreturns_savepath,
)
