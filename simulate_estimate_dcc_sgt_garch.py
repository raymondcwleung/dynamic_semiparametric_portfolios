import numpy as np

import time
import uuid
import yaml
import pathlib
import logging
from datetime import datetime
import pickle
import argparse

import dcc
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--num_simulations", type=int, help="Number of simulations", required=True)
parser.add_argument("--num_sample", type=int, help="Number of time observations", required=True)
parser.add_argument("--dim", type=int, help="Dimension size (i.e. number of assets)", required=True)
args = parser.parse_args()

num_simulations = args.num_simulations
num_sample = args.num_sample
dim = args.dim

# Directories setup
dir_log_simulations = pathlib.Path().resolve().joinpath("./logs/simulations/")
dir_data_simulations = pathlib.Path().resolve().joinpath("./data/simulations/")
dir_log_simulations.mkdir(parents=True, exist_ok=True)
dir_data_simulations.mkdir(parents=True, exist_ok=True)



for sim in range(num_simulations):
    # Setup
    hashid = uuid.uuid4().hex
    str_id = utils.gen_str_id(num_sample=num_sample, dim=dim, hashid=hashid)
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    fn = f"{current_time}_{str_id}"


    # Logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        datefmt="%Y-%m-%d %I:%M:%S %p",
        level=logging.INFO,
        format="%(levelname)s | %(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(dir_log_simulations.joinpath(f"{fn}.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger.info(f"--- Scenario iter {sim}/{num_simulations} num_sample = {num_sample} and dim = {dim} ---")


    beg_time = time.perf_counter()

    # Simulate returns
    logger.info(f"Begin simulation")
    simreturns = dcc.gen_simulation_dcc_sgt_garch(num_sample = num_sample, dim = dim)
    logger.info(f"End simulation")

    # Estimate parameters
    logger.info(f"Begin estimation")
    estimation_res = dcc.calc_estimation_dcc_sgt_garch(num_sample=num_sample, dim = dim, simreturns=simreturns)

    # Save only results where optimization was valid
    if estimation_res.valid_optimization:
        neg_loglik_val = np.array(estimation_res.neg_loglik_val)
        dict_res = {"simreturns" : simreturns, "estimation_res" : estimation_res}
        with open(dir_data_simulations.joinpath(f"{fn}.pkl"), "wb") as f:
            pickle.dump(dict_res, f)

        logger.info(f"End estimation with at value {neg_loglik_val}")
    else:
        logger.info(f"Invalid estimation at simulation {sim}")


    end_time = time.perf_counter()
    total_time = end_time - beg_time

    logger.info(f"--- Done DCC-SGT-GARCH simulations and estimation results in {total_time:.3f}s {fn} ---")


