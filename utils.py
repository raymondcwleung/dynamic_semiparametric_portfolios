import jax
import jax.numpy as jnp
import jaxtyping as jpt

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