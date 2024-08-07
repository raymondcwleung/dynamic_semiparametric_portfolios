import numpy as np

import time
import uuid
import pathlib
import logging
from datetime import datetime
import pickle
import argparse
import os

import innovations
import dcc
import utils


# FIX: Remove default values
parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_simulations",
    type=int,
    help="Number of simulations",
    required=True,
    const=25,
    nargs="?",
)
parser.add_argument(
    "--num_sample",
    type=int,
    help="Number of time observations",
    required=True,
    const=100,
    nargs="?",
)
parser.add_argument(
    "--dim",
    type=int,
    help="Dimension size (i.e. number of assets)",
    required=True,
    const=2,
    nargs="?",
)
args = parser.parse_args()

num_simulations = args.num_simulations
num_sample = args.num_sample
dim = args.dim

# Number of additional samples to simulate, so that
# the total number of samples generated is
# BUFFER_SAMPLE + num_sample. However, we will discard
# the initial BUFFER_SAMPLE amount so as to remove the
# effects of initial conditions in autoregressive-type
# models.
BUFFER_SAMPLE = 1000

# Directories setup
dir_data_simulations = utils.get_simulations_data_dir(num_sample=num_sample, dim=dim)
dir_log_simulations = dir_data_simulations.joinpath("./_logs")
dir_data_simulations_simreturns = dir_data_simulations.joinpath("./simreturns")
dir_data_simulations_estimations = dir_data_simulations.joinpath("./estimations")
dir_log_simulations.mkdir(parents=True, exist_ok=True)
dir_data_simulations.mkdir(parents=True, exist_ok=True)
dir_data_simulations_simreturns.mkdir(parents=True, exist_ok=True)
dir_data_simulations_estimations.mkdir(parents=True, exist_ok=True)

# Logging
current_time = datetime.now().strftime("%Y%m%d%H%M%S")
logger = logging.getLogger(__name__)
logging.basicConfig(
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(
            dir_log_simulations.joinpath(
                f"{current_time}_{num_simulations}_{num_sample}_{dim}.log"
            ),
            mode="w",
        ),
        logging.StreamHandler(),
    ],
)


#########################################################
## Step 1: Simulate returns
#########################################################
simreturns_fn = dir_data_simulations_simreturns.joinpath(f"./simreturns_{num_sample}_{dim}.pkl")
if simreturns_fn.is_file() is False:
    # Generate simulations if it's not available yet
    logger.info(f"No existing simulation. Begin simulation")
    simreturns = dcc.gen_simulation_dcc_gaussian_garch(
        num_sample=BUFFER_SAMPLE + num_sample, dim=dim
    )
    with open(simreturns_fn, "wb") as f:
        pickle.dump(simreturns, f)

    logger.info(f"End simulation")
else:
    # Load in already computed simulated results
    with open(simreturns_fn, "rb") as f:
        simreturns = pickle.load(f)
    logger.info(f"Finished loading simulation {simreturns_fn}")


# Truncate the sample so that we are not affected
# by the initial conditions of the simulation. Thus the
# data shape that enters into estimation is of
# (num_sample, dim).
mat_returns = simreturns.data_mat_returns[BUFFER_SAMPLE:]


#########################################################
## Step 2: Estimation
#########################################################
# Clean all existing (if any) estimations results
logger.info("Clean all existing estimations")
filelist = [f for f in os.listdir(dir_data_simulations_estimations)]
for f in filelist: 
    os.remove(os.path.join(dir_data_simulations_estimations, f))


sim = 0
while sim < num_simulations:
    # Setup
    hashid = uuid.uuid4().hex
    str_id = utils.gen_str_id(num_sample=num_sample, dim=dim, hashid=hashid)
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    fn = f"{current_time}_{str_id}"

    logger.info(
        f"--- Scenario iter {sim}/{num_simulations} num_sample = {num_sample} and dim = {dim} ---"
    )
    beg_time = time.perf_counter()

    # Estimate parameters
    logger.info(f"Begin estimation")
    estimation_res = dcc.calc_estimation_dcc_gaussian_garch(mat_returns=mat_returns)

    # Save only results where optimization was valid
    if estimation_res.valid_optimization:
        neg_loglik_val = np.array(estimation_res.neg_loglik_val)
        with open(dir_data_simulations_estimations.joinpath(f"{fn}.pkl"), "wb") as f:
            pickle.dump(estimation_res, f)

        logger.info(f"End estimation with at value {neg_loglik_val}")
        sim += 1
    else:
        logger.info(f"Invalid estimation at simulation {sim}")
        # Cancel this simulation and re-do
        sim -= 1

    end_time = time.perf_counter()
    total_time = end_time - beg_time

    logger.info(
        f"--- Done DCC-Gaussian-GARCH simulations and estimation results in {total_time:.3f}s {fn} ---"
    )
