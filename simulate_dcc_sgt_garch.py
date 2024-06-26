import jax
import jax.numpy as jnp
import jax.scipy.optimize
from jax import random
import jax.test_util

import numpy as np

import logging
import pathlib
import uuid

import argparse

import utils
import dcc
import sgt

jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod

parser = argparse.ArgumentParser()
parser.add_argument("--num_sample", type=int, help="Number of samples", required=True)
parser.add_argument("--dim", type=int, help="Dimension size (i.e. number of assets)", required=True)
args = parser.parse_args()

#################################################################
## Setup
#################################################################
# Random initial seed based on current time
SEED = utils.gen_seed_number()
HASHID = uuid.uuid4().hex
NUM_SAMPLE = args.num_sample
DIM = args.dim
STR_ID = utils.gen_str_id(num_sample=NUM_SAMPLE, dim=DIM, hashid=HASHID)

key = random.key(SEED)
rng = np.random.default_rng(SEED)

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/simulate_dcc_sgt_garch_{STR_ID}.log", mode="w"),
        logging.StreamHandler(),
    ],
)


#################################################################
## Parameters for time-varying SGT
#################################################################
# Set the true parameters of the SGT z_t process
mat_lbda_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_LBDA_TVPARAMS, DIM))
mat_lbda_tvparams[0, :] = np.abs(mat_lbda_tvparams[0, :])
mat_p0_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_P0_TVPARAMS, DIM))
mat_p0_tvparams[0, :] = np.abs(mat_p0_tvparams[0, :])
mat_q0_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_Q0_TVPARAMS, DIM))
mat_q0_tvparams[0, :] = np.abs(mat_q0_tvparams[0, :])
params_z_sgt_true = sgt.ParamsZSgt(
    mat_lbda_tvparams=jnp.array(mat_lbda_tvparams),
    mat_p0_tvparams=jnp.array(mat_p0_tvparams),
    mat_q0_tvparams=jnp.array(mat_q0_tvparams),
)

# Set the initial t = 0 conditions for the various processes
# in constructing time-varying parameters
inittimecond_z_sgt_true = sgt.InitTimeConditionZSgt(
    vec_z_init_t0=jnp.repeat(0.0, DIM),
    vec_lbda_init_t0=jnp.array(rng.uniform(-0.25, 0.25, DIM)),
    vec_p0_init_t0=jnp.array(rng.uniform(2, 4, DIM)),
    vec_q0_init_t0=jnp.array(rng.uniform(2, 4, DIM)),
)

#################################################################
## Parameters for DCC-GARCH
#################################################################
# Parameters for the mean returns vector
params_mean_true = dcc.ParamsMean(vec_mu=jnp.array(rng.uniform(0, 1, DIM) / 50))

# Params for DCC -- univariate vols
params_uvar_vol_true = dcc.ParamsUVarVol(
    vec_omega=jnp.array(rng.uniform(0, 1, DIM) / 2),
    vec_beta=jnp.array(rng.uniform(0, 1, DIM) / 3),
    vec_alpha=jnp.array(rng.uniform(0, 1, DIM) / 10),
    vec_psi=jnp.array(rng.uniform(0, 1, DIM) / 5),
)
# Params for DCC -- multivariate Q
params_mvar_cor_true = dcc.ParamsMVarCor(
    vec_delta=jnp.array([0.007, 0.930]),
    mat_Qbar=dcc.generate_random_cov_mat(key=key, dim=DIM) / 5,
)

# Package all the DCC params together
params_dcc_true = dcc.ParamsDcc(
    uvar_vol=params_uvar_vol_true, mvar_cor=params_mvar_cor_true
)


# Initial t = 0 conditions for the DCC Q_t process
subkeys = random.split(key, 2)
mat_Sigma_init_t0 = dcc.generate_random_cov_mat(key=subkeys[0], dim=DIM)
mat_Q_init_t0 = dcc.generate_random_cov_mat(key=subkeys[1], dim=DIM)
inittimecond_dcc_true = dcc.InitTimeConditionDcc(
    mat_Sigma_init_t0=mat_Sigma_init_t0, mat_Q_init_t0=mat_Q_init_t0
)

# Put everything together
params_dcc_sgt_garch_true = dcc.ParamsDccSgtGarch(
    mean=params_mean_true, dcc=params_dcc_true, sgt=params_z_sgt_true
)
inittimecond_dcc_sgt_garch_true = dcc.InitTimeConditionDccSgtGarch(
    sgt=inittimecond_z_sgt_true, dcc=inittimecond_dcc_true
)

#################################################################
## Simulate DCC-SGT-GARCH
#################################################################
fn = f"simulated_data/data_simreturns_timevarying_sgt_{STR_ID}.pkl"
data_simreturns_savepath = pathlib.Path().resolve() / fn
dcc.simulate_dcc_sgt_garch(
    seed=SEED,
    hashid = HASHID,
    key=key,
    dim=DIM,
    num_sample=NUM_SAMPLE,
    params_dcc_sgt_garch=params_dcc_sgt_garch_true,
    inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch_true,
    data_simreturns_savepath=data_simreturns_savepath,
)
