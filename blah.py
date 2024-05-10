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


def _sgt_density(z, lbda, p0, q0):
    """
    SGT density (with zero mean and unit variance)
    """
    # Hard coded
    mu = 0
    sigma = 1

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


def mvar_sgt_density(vec_z, vec_params_lbda, vec_params_p0, vec_params_q0):
    n = len(vec_z)

    density = 1
    for ii in jnp.arange(n):
        d_ = _sgt_density(
            z=vec_z[ii],
            lbda=vec_params_lbda[ii],
            p0=vec_params_p0[ii],
            q0=vec_params_q0[ii],
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


def calc_sigma_t(garch_params, sigma_t_minus_one, epsilon_t_minus_one):
    """
    Glosten, Jagannathan and Runkle (1993)

    Parmeter labels
    ---------------
    omega = garch_params[0]
    beta = garch_params[1]
    alpha = garch_params[2]
    psi = garch_params[3]

    Return
    ------
    Scalar-valued \sqrt{\sigma_{i,t}^2} for asset i
    """
    var_t = (
        garch_params[0]
        + garch_params[1] * sigma_t_minus_one**2
        + garch_params[2] * epsilon_t_minus_one**2
        + garch_params[3] * epsilon_t_minus_one**2 * _indicator(epsilon_t_minus_one)
    )
    sigma_t = jnp.sqrt(var_t)
    return sigma_t


def calc_q_t(mat_q_bar, dcc_params, mat_q_t_minus_1, vec_u_t_minus_one):
    """
    DCC specification of Engle (2002) and Engle and Sheppard (2001)

    Parmeter labels
    ---------------
    delta_1 = dcc_params[0]
    delta_2 = dcc_params[1]

    Return
    ------
    Matrix-valued Q_t
    """
    # u_{t-1}u_{t-1}^\top
    uuT = jnp.outer(vec_u_t_minus_one, vec_u_t_minus_one)

    mat_q_t = (
        (1 - dcc_params[0] - dcc_params[1]) * mat_q_bar
        + dcc_params[0] * uuT
        + dcc_params[1] * mat_q_t_minus_1
    )
    return mat_q_t


def calc_vec_u_t(vec_sigma_t, vec_epsilon_t):
    """
    u_t = D_t^{-1} \epsilon_t, where D_t = diag(\sigma_{1,t}, \ldots, \sigma_{n,t}),
    so u_t = ( \epsilon_{1,t} / \sigma_{1,t}, \ldots, \epsilon_{n,t} / \sigma_{n,t} )

    Return
    ------
    Vector-valued u_t
    """
    vec_u_t = vec_epsilon_t / vec_sigma_t
    return vec_u_t


def calc_mat_gamma_t(mat_q_t):
    """
    \Gamma_t = (\diag Q_t)^{-1/2} Q_t (\diag Q_t)^{-1/2}

    Return
    ------
    Matrix-valued \Gamma_t
    """
    # \diag(Q_t)^{-1/2} = diag( q_{1,1}^{-1/2}, \ldots, q_{n,n}^{-1/2} )
    diag_q_t_negsqrt = jnp.diag(jnp.diag(mat_q_t) ** (-1 / 2))

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


def calc_vec_z_t(mat_sigma_t, vec_epsilon_t):
    """
    z_t = \Sigma_t^{-1/2} \epsilon_t = \Sigma_t^{-1/2} (R_t - \mu_t)

    Return
    ------
    Vector-valued z_t
    """
    # Compute \Sigma_t^{1/2} via spectral theorem
    eigvals, eigvecs = jnp.linalg.eigh(mat_sigma_t)
    sqrt_eigvals = jnp.sqrt(eigvals)
    mat_sqrt_diag = jnp.diag(sqrt_eigvals)
    mat_sigma_sqrt = jnp.matmul(jnp.matmul(eigvecs, mat_sqrt_diag), eigvecs.transpose())

    # Solve Ax = b. In our case, A = \Sigma_t^{1/2} and b = \epsilon_t. The solution x is z_t.
    z_t = jscipy.linalg.solve(a=mat_sigma_sqrt, b=vec_epsilon_t)

    return z_t


# mu = 0
# sigma = 1
# lbda = 0
# p = 10
# q = 5
# zz = np.linspace(-10, 10, 1000)
# hi = sgt_density(zz, mu, sigma, lbda, p, q)
#
#
# mu = jnp.array([0, 0])
# sigma = jnp.array([1, 1])
# lbda = jnp.array([0, 0])
# p = jnp.array([10, 11])
# q = jnp.array([5, 6])
# zz = jnp.array([0.2, 0.3])
# hihi = mvar_sgt_density(zz, mu, sigma, lbda, p, q)


calc_vec_sigma_t = jax.vmap(calc_sigma_t, in_axes=(0, 0, 0))

# Fake asset returns
num_assets = 5
num_time_obs = 1000
key = random.key(51234)
# N x T
mat_rtn = jax.random.normal(key, (num_assets, num_time_obs))


def _gen_garch_init_params(
    num_assets: int,
    key,
    num_agarch_params: int = 4,
):
    """
    Generate the initial random conditions to the
    multivariate DCC model
    """

    # \sigma_{i, 0}
    vec_sigma_0 = jax.random.uniform(key, (num_assets,)) / 2

    # (\varepsilon_{i,0})
    vec_epsilon_0 = mat_rtn[:, 0] - vec_mu

    # (u_{i,0})
    vec_u_0 = calc_vec_u_t(vec_sigma_t=vec_sigma_0, vec_epsilon_t=vec_epsilon_0)

    # Q_0
    _ = jax.random.uniform(key, (num_assets, num_assets)) / 2
    mat_q_0 = jnp.dot(_, _.transpose())

    return (vec_sigma_0, vec_epsilon_0, vec_u_0, mat_q_0)


############################
# Parameters
############################
key = random.key(1234)

# Parameters of the g(.) SGT density
vec_params_lbda = jnp.array([0, 0, 0, 0, 0])
vec_params_p0 = jnp.array([1, 1, 1, 1, 1])
vec_params_q0 = jnp.array([1.1, 1.1, 1.1, 1.1, 1.1])

# Constant mean vector
key, subkey = random.split(key)
vec_mu = jax.random.uniform(subkey, (num_assets,)) / 2

# Parameters of the univariate asymmetric GARCH models of Glosten, Jagannathan
# and Runkle (1993). All these parameters are strictly positive.
# [\omega_i, \beta_i, \alpha_i, \psi_i]
num_agarch_params: int = 4
key, subkey = random.split(key)
garch_params = jax.random.uniform(subkey, (num_assets, num_agarch_params))

# Parameters of the asymmetric DCC model of Engle.
# [\delta_1, \delta_2]
key, subkey = random.split(key)
_ = jax.random.uniform(subkey, (2,))
dcc_params = _ / (2 * jnp.linalg.norm(_, 1))

# \bar{Q}
_ = jax.random.uniform(key, (num_assets, num_assets)) / 2
mat_q_bar = jnp.dot(_, _.transpose())


# # (z_{i,t})
# vec_z_t = mat_rtn[:, tt] - vec_mu
#


@jit
def _calc_log_likelihood_t(
    # Asset returns
    tt,
    mat_rtn,
    # t - 1 terms,
    vec_sigma_t_minus_1,
    vec_epsilon_t_minus_1,
    vec_u_t_minus_one,
    mat_q_t_minus_1,
    # Univar vol params
    vec_mu,
    garch_params,
    # Multivar cov params
    mat_q_bar,
    dcc_params,
    # SGT density params
    vec_params_lbda,
    vec_params_p0,
    vec_params_q0,
):
    """
    Calculate the t-th summand of the log-likelihood function
    """
    #############################################
    ## STEP 1: Construct univariate vol
    #############################################
    # (\sigma_{i,t})
    vec_sigma_t = calc_vec_sigma_t(
        garch_params,
        vec_sigma_t_minus_1,
        vec_epsilon_t_minus_1,
    )

    # (\varepsilon_{i,t})
    vec_epsilon_t = mat_rtn[:, tt] - vec_mu

    # (u_{i,t})
    vec_u_t = calc_vec_u_t(vec_sigma_t=vec_sigma_t, vec_epsilon_t=vec_epsilon_t)

    ###################################################
    ## STEP 2: Construct multivariate covariance matrix
    ###################################################
    # Q_t
    mat_q_t = calc_q_t(
        mat_q_bar=mat_q_bar,
        dcc_params=dcc_params,
        mat_q_t_minus_1=mat_q_t_minus_1,
        vec_u_t_minus_one=vec_u_t_minus_one,
    )
    # \Gamma_t
    mat_gamma_t = calc_mat_gamma_t(mat_q_t=mat_q_t)

    # \Sigma_t
    mat_sigma_t = calc_mat_sigma_t(vec_sigma_t=vec_sigma_t, mat_gamma_t=mat_gamma_t)

    ###################################################
    ## STEP 3: Construct t-th summand of log-likelihood
    ###################################################
    # z_t = \Sigma_t^{-1/2} \varepsilon_t,
    # where \varepsilon_t = R_t - \mu
    vec_z_t = calc_vec_z_t(mat_sigma_t=mat_sigma_t, vec_epsilon_t=vec_epsilon_t)

    # \log g(z_t)
    log_density_t = jnp.log(
        mvar_sgt_density(vec_z_t, vec_params_lbda, vec_params_p0, vec_params_q0)
    )

    # \log\det(\Sigma_t)
    log_det_sigma_t = jnp.log(jnp.linalg.det(mat_sigma_t))

    # Log-likelihood L(R_t) at time t
    log_lik_t = log_density_t - 0.5 * log_det_sigma_t

    ###################################################
    ## STEP 4: Output
    ###################################################
    return log_lik_t, vec_sigma_t, vec_epsilon_t, vec_u_t


def calc_log_likelihood(
    # Asset returns
    mat_rtn,
    # Univar vol params
    vec_mu,
    garch_params,
    # Multivar cov params
    mat_q_bar,
    dcc_params,
    # SGT density params
    vec_params_lbda,
    vec_params_p0,
    vec_params_q0,
    # Hyper params
    key=random.key(1234),
):
    """
    Calculate the log-likelihood fucntion
    """
    num_assets = mat_rtn.shape[0]
    num_time_obs = mat_rtn.shape[1]

    # Initial conditions t = 0
    (vec_sigma_0, vec_epsilon_0, vec_u_0, mat_q_0) = _gen_garch_init_params(
        num_assets=num_assets, key=key
    )

    vec_sigma_t_minus_1 = vec_sigma_0
    vec_epsilon_t_minus_1 = vec_epsilon_0
    vec_u_t_minus_one = vec_u_0
    mat_q_t_minus_1 = mat_q_0

    log_lik = 0
    for tt in range(1, num_time_obs):
        # Calculate the t-th summand of the log-likelihood
        log_lik_t, vec_sigma_t, vec_epsilon_t, vec_u_t = _calc_log_likelihood_t(
            tt=tt,
            mat_rtn=mat_rtn,
            #
            vec_sigma_t_minus_1=vec_sigma_t_minus_1,
            vec_epsilon_t_minus_1=vec_epsilon_t_minus_1,
            vec_u_t_minus_one=vec_u_t_minus_one,
            mat_q_t_minus_1=mat_q_t_minus_1,
            #
            vec_mu=vec_mu,
            garch_params=garch_params,
            #
            mat_q_bar=mat_q_bar,
            dcc_params=dcc_params,
            #
            vec_params_lbda=vec_params_lbda,
            vec_params_p0=vec_params_p0,
            vec_params_q0=vec_params_q0,
        )

        # Update
        log_lik += log_lik_t
        vec_sigma_t_minus_1 = vec_sigma_t
        vec_epsilon_t_minus_1 = vec_epsilon_t
        vec_u_t_minus_one = vec_u_t

    return log_lik


log_lik = calc_log_likelihood(
    mat_rtn=mat_rtn,
    #
    vec_mu=vec_mu,
    garch_params=garch_params,
    #
    mat_q_bar=mat_q_bar,
    dcc_params=dcc_params,
    #
    vec_params_lbda=vec_params_lbda,
    vec_params_p0=vec_params_p0,
    vec_params_q0=vec_params_q0,
)
