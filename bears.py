import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.optimize
from jax import grad, jit, vmap
from jax import random
import jax
import jax.test_util

import typing as tp

from jax import Array
from jax.typing import ArrayLike, DTypeLike
from jax._src.random import KeyArrayLike

import optax
import jaxopt

import numpy as np
import scipy
import matplotlib.pyplot as plt


import time

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
# jax.config.update("jax_enable_x64", True) # Should use x64 in full prod
jax.config.update("jax_debug_nans", True)  # Should disable in full prod
