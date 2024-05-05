import jax.numpy as jnp
import jax.scipy as jscipy

from jax import grad, jit, vmap
from jax import random
import jax

import numpy as np
import matplotlib.pyplot as plt


key = random.key(0)

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

xs = np.random.normal(size=(100,))
noise = np.random.normal(scale=0.1, size=(100,))
ys = xs * 3 - 1 + noise


def apply_matrix(v):
    return jnp.dot(mat, v)


def model(theta, x):
    """Computes wx + b on a batch of input x."""
    w, b = theta
    return w * x + b


def loss_fn(theta, x, y):
    prediction = model(theta, x)
    return jnp.mean((prediction - y) ** 2)


@jit
def update(theta, x, y, lr=0.1):
    return theta - lr * jax.grad(loss_fn)(theta, x, y)


# theta = jnp.array([1.0, 1.0])
#
# for _ in range(10000):
#     theta = update(theta, xs, ys)
#
#
# plt.scatter(xs, ys)
# plt.plot(xs, model(theta, xs))
#
# w, b = theta
# print(f"w: {w:<.2f}, b: {b:<.2f}")
#

# nu = jnp.array([3, 4])
# xi = jnp.array([0.1, 0.1])
# z = jnp.array([0.05, 0.25])


def mu_t(rtn_t, mu_bar=0):
    return mu_bar


def sgt_density(z, mu, sigma, lbda, p0, q0):
    v = jnp.power(q0, -1 / p0) / jnp.sqrt(
        (1 + 3 * lbda**2)
        * jscipy.special.beta(3 / p0, q0 - 2 / p0)
        / jscipy.special.beta(1 / p0, q0)
        - 4
        * lbda**2
        * jscipy.special.beta(2 / p0, q0 - 1 / p0) ** 2
        / jscipy.special.beta(1 / p0, q0) ** 2
    )

    m = (
        lbda
        * v
        * sigma
        * 2
        * jnp.power(q0, 1 / p0)
        * jscipy.special.beta(2 / p0, q0 - 1 / p0)
        / jscipy.special.beta(1 / p0, q0)
    )

    density = p0 / (
        2
        * v
        * sigma
        * jnp.power(q0, 1 / p0)
        * jscipy.special.beta(1 / p0, q0)
        * jnp.power(
            1
            + jnp.power(jnp.abs(z - mu + m), p0)
            / (
                q0
                * jnp.power(v * sigma, p0)
                * jnp.power(1 + lbda * jnp.sign(z - mu + m), p0)
            ),
            1 / p0 + q0,
        )
    )
    return density


@jit
def mvar_sgt_density(vec_z, vec_mu, vec_sigma, vec_lbda, vec_p0, vec_q0):
    n = len(vec_z)

    density = 1
    for ii in jnp.arange(n):
        d_ = sgt_density(
            z=vec_z[ii],
            mu=vec_mu[ii],
            sigma=vec_sigma[ii],
            lbda=vec_lbda[ii],
            p0=vec_p0[ii],
            q0=vec_q0[ii],
        )
        density *= d_

    return density


def _indicator(x):
    """
    Indicator function x \mapsto \ind(x \le 0)
    """
    return x <= 0


def _positive_part(x):
    """
    Positive part of a scalar x^{+} := \max\{ x, 0 \}
    """
    return jnp.max(x, 0)


def _negative_part(x):
    """
    Negative part of a scalar x^{-} := \max\{ -x, 0 \}
    """
    return jnp.max(-x, 0)


def calc_lbda_t(vec_param, z_t_minus_one, lbda_t_minus_one, lbda_bar=4):
    """
    Time-varying dynamics for \lambda_{i,t}

    Return
    ------
    Scalar valued exponential GARCH style process \lambda_{i,t}
    """
    # (1 - c_2 L) \log(\lambda_t - \bar{\lambda})
    rhs = (
        jnp.log(vec_param[0])
        + (
            _negative_part(vec_param[1])
            * jnp.abs(z_t_minus_one)
            * _indicator(z_t_minus_one)
        )
        + (
            _positive_part(vec_param[1])
            * jnp.abs(z_t_minus_one)
            * (1 - _indicator(z_t_minus_one))
        )
    )

    # Solve for lbda_t
    lbda_t_out = lbda_bar + jnp.exp(
        rhs + vec_param[2] * jnp.log(lbda_t_minus_one - lbda_bar)
    )
    return lbda_t_out


def calc_p0_t(vec_param, z_t_minus_one, p0_t_minus_one, p0_bar=0):
    """
    Time-varying dynamics for p_{i,t}

    Return
    ------
    Scalar valued exponential GARCH style process p_{i,t}
    """
    # (1 - c_2 L) \log(p_t - \bar{p})
    rhs = (
        jnp.log(vec_param[0])
        + (
            _negative_part(vec_param[1])
            * jnp.abs(z_t_minus_one)
            * _indicator(z_t_minus_one)
        )
        + (
            _positive_part(vec_param[1])
            * jnp.abs(z_t_minus_one)
            * (1 - _indicator(z_t_minus_one))
        )
    )

    # Solve for p_t
    p_t_out = p0_bar + jnp.exp(rhs + vec_param[2] * jnp.log(p0_t_minus_one - p0_bar))
    return p_t_out


def calc_q0_t(vec_param, z_t_minus_one, q0_t_minus_one, q0_bar=0):
    """
    Time-varying dynamics for q_{i,t}

    Return
    ------
    Scalar valued exponential GARCH style process q_{i,t}
    """
    # (1 - c_2 L) \log(q_t - \bar{q})
    rhs = (
        jnp.log(vec_param[0])
        + (
            _negative_part(vec_param[1])
            * jnp.abs(z_t_minus_one)
            * _indicator(z_t_minus_one)
        )
        + (
            _positive_part(vec_param[1])
            * jnp.abs(z_t_minus_one)
            * (1 - _indicator(z_t_minus_one))
        )
    )

    # Solve for q_t
    q_t_out = q0_bar + jnp.exp(rhs + vec_param[2] * jnp.log(q0_t_minus_one - q0_bar))
    return q_t_out


def calc_sigma_gjr_garch(vec_param, sigma_t_minus_one, epsilon_t_minus_one):
    """
    Glosten, Jagannathan and Runkle (1993)

    Return
    ------
    Scalar-valued \sqrt{\sigma_{i,t}^2}
    """
    omega, beta, alpha, psi = vec_param

    var_t = (
        omega
        + beta * sigma_t_minus_one**2
        + alpha * epsilon_t_minus_one**2
        + psi * epsilon_t_minus_one**2 * _indicator(epsilon_t_minus_one)
    )
    sigma_t = jnp.sqrt(var_t)
    return sigma_t


def calc_q_t(mat_omega_bar, vec_delta, mat_q_t_minus_1, vec_u_t_minus_one):
    """
    Return
    ------
    Matrix-valued Q_t
    """
    # u_{t-1}u_{t-1}^\top
    uuT = jnp.outer(vec_u_t_minus_one, vec_u_t_minus_one)

    vec_q_t_out = (
        mat_omega_bar
        + vec_delta[0] * mat_q_t_minus_1
        + vec_delta[1] * uuT
        + vec_delta[2] * uuT * _indicator(vec_u_t_minus_one)
    )
    return vec_q_t_out


def calc_u_t(vec_sigma_t, vec_epsilon_t):
    """
    u_t = D_t^{-1} \epsilon_t, where D_t = diag(\sigma_{1,t}, \ldots, \sigma_{n,t}),
    so u_t = ( \epsilon_{1,t} / \sigma_{1,t}, \ldots, \epsilon_{n,t} / \sigma_{n,t} )

    Return
    ------
    Vector-valued u_t
    """
    return vec_epsilon_t / vec_sigma_t


def calc_mat_gamma_t(mat_q_t):
    """
    \Gamma_t = (\diag Q_t)^{-1/2} Q_t (\diag Q_t)^{-1/2}

    Return
    ------
    Matrix-valued \Gamma_t
    """
    # \diag(Q_t)^{-1/2} = diag( q_{1,1}^{-1/2}, \ldots, q_{n,n}^{-1/2} )
    diag_q_t_negsqrt = jnp.diag(jnp.power(jnp.diag(mat_q_t), -1 / 2))

    gamma_t_out = jnp.matmul(jnp.matmul(diag_q_t_negsqrt, mat_q_t), diag_q_t_negsqrt)
    return gamma_t_out


def calc_mat_sigma_t(vec_sigma_t, mat_gamma_t):
    """
    \Sigma_t = D_t \Gamma_t D_t

    Return
    ------
    Matrix-valued \Sigma_t
    """
    mat_d_t = jnp.diag(vec_sigma_t)

    sigma_t_out = jnp.matmul(jnp.matmul(mat_d_t, mat_gamma_t), mat_d_t)
    return sigma_t_out


def calc_vec_epsilon_t(vec_rtn_t, vec_mu_t):
    """
    \epsilon_t = vec_rtn_t - vec_mu_t

    Return
    ------
    Vector-valued \epsilon_t
    """
    return vec_rtn_t - vec_mu_t


def vec_z_t(mat_sigma_t, vec_epsilon_t):
    """
    z_t = \Sigma_t^{-1/2} \epsilon_t = \Sigma_t^{-1/2} (R_t - \mu_t)

    Return
    ------
    Vector-valued z_t
    """
    # Compute \Sigma_t^{1/2}
    mat_sigma_sqrt = jscipy.linalg.sqrtm(A=mat_sigma_t)

    # Solve Ax = b. In our case, A = \Sigma_t^{1/2} and b = \epsilon_t. The solution x is z_t.
    z_t = jscipy.linalg.solve(a=mat_sigma_sqrt, b=vec_epsilon_t)

    return z_t


mu = 0
sigma = 1
lbda = 0
p = 10
q = 5
zz = np.linspace(-10, 10, 1000)
hi = sgt_density(zz, mu, sigma, lbda, p, q)


mu = jnp.array([0, 0])
sigma = jnp.array([1, 1])
lbda = jnp.array([0, 0])
p = jnp.array([10, 11])
q = jnp.array([5, 6])
zz = jnp.array([0.2, 0.3])
hihi = mvar_sgt_density(zz, mu, sigma, lbda, p, q)

omega = 0.1
beta = 0.5
alpha = -0.25
psi = 0.5
sigma_t_minus_one = 0.1
epsilon_t_minus_one = -0.33
sigma_t = calc_sigma_gjr_garch(
    omega=omega,
    beta=beta,
    alpha=alpha,
    psi=psi,
    sigma_t_minus_one=sigma_t_minus_one,
    epsilon_t_minus_one=epsilon_t_minus_one,
)

omega_bar = jnp.array([[1, 0], [0, 1]])
delta = jnp.array([0.2, 0.3, 0.4])
q_t_minus_1 = jnp.array([[0.1, 0.3], [0.3, 0.3]])
u_t_minus_one = jnp.array([-0.1, 0.2])
qt = calc_q_t(omega_bar, delta, q_t_minus_1, u_t_minus_one)

vec_sigma_t = jnp.array([0.1, 0.2])
calc_vec_epsilon_t = jnp.array([-0.3, 0.134])
ut = calc_u_t(vec_sigma_t, calc_vec_epsilon_t)

mat_gamma = calc_mat_gamma_t(q_t_minus_1)

mat_sigma = calc_mat_sigma_t(vec_sigma_t, mat_gamma)
