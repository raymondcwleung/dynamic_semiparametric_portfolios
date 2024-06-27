from jax._src.source_info_util import is_user_filename
import jax.numpy as jnp
import jaxtyping as jpt

import dataclasses
import typing as tp
import chex

import pathlib

import datetime


@chex.dataclass
class ParamAnalytics:
    """
    Convenient placeholder for storing the true, estimated and the difference
    squared parameters per-simulation.
    """

    true: tp.Any
    est: tp.Any
    diff: tp.Any


def gen_seed_number() -> int:
    """
    Generate a random seed number based on current time
    """
    _tim = datetime.datetime.now()
    seed = _tim.hour * 10000 + _tim.minute * 100 + _tim.second
    return seed


def gen_str_id(num_sample: int, dim: int, hashid: str) -> str:
    """
    Generate a unique string identifier.
    """
    str_id = f"{num_sample}_{dim}_{hashid}"
    return str_id


def positive_part(x: jpt.Array) -> jpt.Array:
    """
    Positive part of a scalar x^{+} := \\max\\{ x, 0 \\}
    """
    return jnp.maximum(x, 0)


def negative_part(x: jpt.Array) -> jpt.Array:
    """
    Negative part of a scalar x^{-} :=
    \\max\\{ -x, 0 \\} = -min\\{ x, 0 \\}
    """
    return -1 * jnp.minimum(x, 0)


def indicator(x: jpt.Array):
    """
    Indicator function x \\mapsto \\ind(x \\le 0)
    """
    return x <= 0


def get_simulations_data_dir(
    num_sample: int,
    dim: int,
    make_dir: bool = True,
    subdir="./data/simulations/",
    basedir: pathlib.Path = pathlib.Path().resolve(),
) -> pathlib.Path:
    dir = basedir.joinpath(subdir)
    dir = dir.joinpath(f"numsample{num_sample}_dim{dim}")

    if make_dir:
        dir.mkdir(parents=True, exist_ok=True)

    return dir


def calc_param_squared_difference(
    dataclass_true_param, dataclass_est_param, dataclass_out
):
    """
    Calculate the squared difference between the true parameters and the
    estimated parameters
    """
    dict_diffsquared = {}

    for key in dataclasses.asdict(dataclass_true_param).keys():
        true_param = dataclasses.asdict(dataclass_true_param)[key]
        est_param = dataclasses.asdict(dataclass_est_param)[key]

        diff_squared = (true_param - est_param) ** 2
        dict_diffsquared[key] = diff_squared

    return ParamAnalytics(
        true=dataclass_true_param,
        est=dataclass_est_param,
        diff=dataclass_out(**dict_diffsquared),
    )


def calc_param_analytics_summary(
    lst_params: tp.List[ParamAnalytics], param_class
) -> tp.Dict[str, ParamAnalytics]:
    """
    Compute the mean and standard error of simulated-estimated data.
    """
    dict_results_summary = {}

    # Fix a parameter type
    for param_type in param_class.__annotations__.keys():
        lst_true = []
        lst_est = []
        lst_diff = []

        # Loop through each simulated-estimated result
        # and extract the true parameters, estimated paramters
        # and their squared differenced quantities
        for res_param in lst_params:
            lst_true.append(getattr(getattr(res_param, "true"), param_type))
            lst_est.append(getattr(getattr(res_param, "est"), param_type))
            lst_diff.append(getattr(getattr(res_param, "diff"), param_type))

        # Stack
        tns_true = jnp.array(lst_true)
        tns_est = jnp.array(lst_est)
        tns_diff = jnp.array(lst_diff)

        # Compute means
        mat_true_param_mean = tns_true.mean(axis=0)
        mat_est_param_mean = tns_est.mean(axis=0)
        # Take also the square root here of the
        # squared differences
        mat_diff_param_sqmean = jnp.sqrt(tns_diff.mean(axis=0))

        # Put things together
        dict_results_summary[param_type] = ParamAnalytics(
            true=mat_true_param_mean, est=mat_est_param_mean, diff=mat_diff_param_sqmean
        )

    return dict_results_summary
