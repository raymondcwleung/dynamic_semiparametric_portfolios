import jax
import jax.numpy as jnp
from jax import random

import logging
import pathlib
import pickle
import argparse

import numpy as np

import utils
import dcc
import sgt


#################################################################
## Setup
#################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--path_simreturns", type=str, help="Path to the saved simulated returns file", required=True)
args = parser.parse_args()

#################################################################
## Read in simulated results
#################################################################
path_simreturns = args.path_simreturns
with open(pathlib.Path(path_simreturns) , "rb") as f:
    simreturns = pickle.load(f)

#################################################################
## Setup
#################################################################
HASHID = simreturns.hashid
NUM_SAMPLE = simreturns.num_sample
DIM = simreturns.dim
STR_ID = utils.gen_str_id(num_sample=NUM_SAMPLE, dim=DIM, hashid=HASHID)

logger = logging.getLogger(__name__)
logging.basicConfig(
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/estimate_dcc_sgt_garch.log", mode="w"),
        logging.StreamHandler(),
    ],
)

seed = utils.gen_seed_number()
#HACK:
seed = 12345
key = random.key(seed)
rng = np.random.default_rng(seed)

#################################################################
## Initial parameter guesses
#################################################################
# Initial guess of the parameters of the time-varying SGT process
# mat_lbda_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_LBDA_TVPARAMS, DIM))
# mat_lbda_tvparams[0, :] = np.abs(mat_lbda_tvparams[0, :])
# mat_p0_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_P0_TVPARAMS, DIM))
# mat_p0_tvparams[0, :] = np.abs(mat_p0_tvparams[0, :])
# mat_q0_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_Q0_TVPARAMS, DIM))
# mat_q0_tvparams[0, :] = np.abs(mat_q0_tvparams[0, :])
mat_lbda_tvparams = rng.uniform(-0.35, 0.35, (sgt.NUM_LBDA_TVPARAMS, DIM))
mat_lbda_tvparams[0, :] = np.abs(mat_lbda_tvparams[0, :])
mat_p0_tvparams = rng.uniform(-0.35, 0.35, (sgt.NUM_P0_TVPARAMS, DIM))
mat_p0_tvparams[0, :] = np.abs(mat_p0_tvparams[0, :])
mat_q0_tvparams = rng.uniform(-0.45, 0.45, (sgt.NUM_Q0_TVPARAMS, DIM))
mat_q0_tvparams[0, :] = np.abs(mat_q0_tvparams[0, :])
params_z_sgt_init_guess = sgt.ParamsZSgt(
    mat_lbda_tvparams=jnp.array(mat_lbda_tvparams),
    mat_p0_tvparams=jnp.array(mat_p0_tvparams),
    mat_q0_tvparams=jnp.array(mat_q0_tvparams),
)

# Initial guess for parameters for the mean returns vector
# params_mean_init_guess = dcc.ParamsMean(vec_mu=jnp.array(rng.uniform(0, 1, DIM) / 50))
params_mean_init_guess = dcc.ParamsMean(vec_mu=jnp.array(rng.uniform(0, 1, DIM)))

# Initial guess for params for DCC -- univariate vols
# params_uvar_vol_init_guess = dcc.ParamsUVarVol(
#     vec_omega=jnp.array(rng.uniform(0, 1, DIM) / 2),
#     vec_beta=jnp.array(rng.uniform(0, 1, DIM) / 3),
#     vec_alpha=jnp.array(rng.uniform(0, 1, DIM) / 10),
#     vec_psi=jnp.array(rng.uniform(0, 1, DIM) / 5),
# )
params_uvar_vol_init_guess = dcc.ParamsUVarVol(
    vec_omega=jnp.array(rng.uniform(0, 1, DIM)),
    vec_beta=jnp.array(rng.uniform(0, 1, DIM)),
    vec_alpha=jnp.array(rng.uniform(0, 1, DIM)),
    vec_psi=jnp.array(rng.uniform(0, 1, DIM)),
)
# Initial guess for params for DCC -- multivariate Q
# FIX: Need to randomize this
params_mvar_cor_init_guess = dcc.ParamsMVarCor(
    vec_delta=jnp.array([0.154, 0.530]),
    # mat_Qbar=dcc.generate_random_cov_mat(key=key, dim=DIM) / 5,
    mat_Qbar=dcc.generate_random_cov_mat(key=key, dim=DIM),
)

# Package all the initial guess DCC params together
params_dcc_init_guess = dcc.ParamsDcc(
    uvar_vol=params_uvar_vol_init_guess,
    mvar_cor=params_mvar_cor_init_guess,
)

guess_params_dcc_sgt_garch = dcc.ParamsDccSgtGarch(
    sgt=params_z_sgt_init_guess,
    mean=params_mean_init_guess,
    dcc=params_dcc_init_guess,
)

# Initial t = 0 conditions for the DCC Q_t process
subkeys = random.split(key, 2)
mat_Sigma_init_t0_guess = dcc.generate_random_cov_mat(key=subkeys[0], dim=DIM)
mat_Q_init_t0_guess = dcc.generate_random_cov_mat(key=subkeys[1], dim=DIM)
inittimecond_dcc_guess = dcc.InitTimeConditionDcc(
    mat_Sigma_init_t0=mat_Sigma_init_t0_guess, mat_Q_init_t0=mat_Q_init_t0_guess
)

# Initital t = 0 conditions for the SGT time-varying
# parameters process
inittimecond_z_sgt_guess = sgt.InitTimeConditionZSgt(
    vec_z_init_t0=jnp.repeat(0.0, DIM),
    vec_lbda_init_t0=jnp.array(rng.uniform(-0.25, 0.25, DIM)),
    vec_p0_init_t0=jnp.array(rng.uniform(3, 4, DIM)),
    vec_q0_init_t0=jnp.array(rng.uniform(3, 4, DIM)),
)

inittimecond_dcc_sgt_garch_guess = dcc.InitTimeConditionDccSgtGarch(
    sgt=inittimecond_z_sgt_guess, dcc=inittimecond_dcc_guess
)

#################################################################
## Maximum likelihood estimation
#################################################################
data_simreturns_estimate_savepath = (
    pathlib.Path().resolve() / f"simulated_data/data_simreturns_timevarying_sgt_estimate_{STR_ID}.pkl"
)

valid_optimization = False
max_loop_iter = 10

valid_optimization, neg_loglik_val, params_dcc_sgt_garch = dcc.dcc_sgt_garch_mle(
    mat_returns=simreturns.data_mat_returns,
    guess_params_dcc_sgt_garch=guess_params_dcc_sgt_garch,
    inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch_guess,
)

