import jax
import jax.numpy as jnp
import jaxtyping as jpt

import datetime

def gen_seed_number() -> int:
    """
    Generate a random seed number based on current time
    """
    _tim = datetime.datetime.now()
    seed = _tim.hour * 10000 + _tim.minute * 100 + _tim.second
    return seed

def gen_str_id(num_sample : int, dim : int, hashid : str) -> str:
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
