import pytest

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

import os
import pathlib

import dcc


jax.config.update("jax_default_device", jax.devices("cpu")[0]) # CPU is fine
jax.config.update("jax_enable_x64", True)

TEST_DATA =pathlib.Path("./tests/test_data_dcc.xlsx")

class TestDCC:
    test_data = TEST_DATA

    num_sample = pd.read_excel(test_data, sheet_name="num_sample", header = None).values.item()
    dim = pd.read_excel(test_data, sheet_name="dim", header = None).values.item()

    mat_epsilon = jnp.array(pd.read_excel(test_data, sheet_name="mat_epsilon", header = None).values)
    mat_sigma = jnp.array(pd.read_excel(test_data, sheet_name="mat_sigma", header = None).values)

    mat_u = jnp.array(pd.read_excel(test_data, sheet_name="mat_u", header = None).values)

    mat_Q_0 = jnp.array(pd.read_excel(test_data, sheet_name="mat_Q_0", header = None).values)
    mat_Qbar = jnp.array(pd.read_excel(test_data, sheet_name="mat_Qbar", header = None).values)

    vec_sigma_init_t0 = jnp.array(pd.read_excel(test_data, sheet_name="vec_sigma_init_t0", header = None).values).reshape(dim, )

    vec_delta = jnp.array(pd.read_excel(test_data, sheet_name="vec_delta", header = None).values).reshape(2, )

    vec_omega = jnp.array(pd.read_excel(test_data, sheet_name="vec_omega", header = None).values).reshape(dim, )
    vec_beta = jnp.array(pd.read_excel(test_data, sheet_name="vec_beta", header = None).values).reshape(dim, )
    vec_alpha = jnp.array(pd.read_excel(test_data, sheet_name="vec_alpha", header = None).values).reshape(dim, )
    vec_psi = jnp.array(pd.read_excel(test_data, sheet_name="vec_psi", header = None).values).reshape(dim, )

    def test_calc_trajectory_uvar_vol_precomputed(self) -> None:
        mat_epsilon = self.mat_epsilon
        vec_sigma_init_t0 = self.vec_sigma_init_t0
        vec_omega = self.vec_omega
        vec_beta = self.vec_beta
        vec_alpha = self.vec_alpha
        vec_psi = self.vec_psi

        num_sample = mat_epsilon.shape[0]
        dim = mat_epsilon.shape[1]

        mat_sigma = jnp.empty(shape=(num_sample, dim))
        mat_sigma = mat_sigma.at[0].set(vec_sigma_init_t0)
        def _body_fun(tt, mat_sigma):
            vec_sigma_t_minus_1 = mat_sigma[tt - 1, :]
            vec_epsilon_t_minus_1 = mat_epsilon[tt - 1, :]

            vec_sigma2_t = dcc._calc_asymmetric_garch_sigma2(
                vec_sigma_t_minus_1=vec_sigma_t_minus_1,
                vec_epsilon_t_minus_1=vec_epsilon_t_minus_1,
                vec_omega=vec_omega,
                vec_beta=vec_beta,
                vec_alpha=vec_alpha,
                vec_psi=vec_psi,
            )
            vec_sigma_t = jnp.sqrt(vec_sigma2_t)

            mat_sigma = mat_sigma.at[tt].set(vec_sigma_t)
            return mat_sigma

        # A slow, but logically correct for-loop calculation
        mat_sigma = jax.lax.fori_loop(
            lower=1, upper=num_sample, body_fun=_body_fun, init_val=mat_sigma
        )

        # Check against the pre-recorded results
        assert jnp.allclose(mat_sigma, self.mat_sigma)


    def test_calc_trajectory_uvar_vol(self) -> None:
        mat_epsilon = self.mat_epsilon
        vec_sigma_init_t0 = self.vec_sigma_init_t0
        vec_omega = self.vec_omega
        vec_beta = self.vec_beta
        vec_alpha = self.vec_alpha
        vec_psi = self.vec_psi

        # Check against the computed results
        calc_mat_sigma = dcc._calc_trajectory_uvar_vol(mat_epsilon=mat_epsilon, 
                                                       vec_sigma_init_t0=vec_sigma_init_t0, 
                                                       vec_omega=vec_omega, 
                                                       vec_beta=vec_beta, 
                                                       vec_alpha=vec_alpha, 
                                                       vec_psi=vec_psi)

        assert jnp.allclose(calc_mat_sigma, self.mat_sigma)


    def test_calc_trajectory_mvar_col(self) -> None:
        mat_epsilon = self.mat_epsilon
        mat_sigma = self.mat_sigma
        mat_Q_0 = self.mat_Q_0
        mat_Qbar = self.mat_Qbar
        vec_delta = self.vec_delta

        num_sample = mat_epsilon.shape[0]
        dim = mat_epsilon.shape[1]

        mat_u = jnp.empty(shape=(num_sample, dim))
        tns_Q = jnp.empty(shape=(num_sample, dim, dim))
        tns_Sigma = jnp.empty(shape=(num_sample, dim, dim))

        vec_u_0 = dcc._calc_normalized_unexpected_excess_rtn(
            vec_sigma=mat_sigma[0, :], vec_epsilon=mat_epsilon[0, :]
        )
        mat_Gamma_0 = dcc._calc_mat_Gamma(mat_Q=mat_Q_0)
        mat_Sigma_0 = dcc._calc_mat_Sigma(vec_sigma=mat_sigma[0, :], mat_Gamma=mat_Gamma_0)

        mat_u = mat_u.at[0].set(vec_u_0)
        tns_Q = tns_Q.at[0].set(mat_Q_0)
        tns_Sigma = tns_Sigma.at[0].set(mat_Sigma_0)


        def _body_fun(tt, carry):
            mat_u, tns_Q, tns_Sigma = carry

            # Compute Q_t
            mat_Q_t = dcc._calc_mat_Q(
                vec_delta=vec_delta,
                vec_u_t_minus_1=mat_u[tt - 1, :],
                mat_Q_t_minus_1=tns_Q[tt - 1, :, :],
                mat_Qbar=mat_Qbar,
            )

            # Compute Gamma_t
            mat_Gamma_t = dcc._calc_mat_Gamma(mat_Q=mat_Q_t)


            # Compute \Sigma_t
            mat_Sigma_t = dcc._calc_mat_Sigma(vec_sigma=mat_sigma[tt, :], mat_Gamma=mat_Gamma_t)

            # Compute u_t
            vec_u_t = dcc._calc_normalized_unexpected_excess_rtn(
                vec_sigma=mat_sigma[tt, :], vec_epsilon=mat_epsilon[tt, :]
            )

            # Bookkeeping
            mat_u = mat_u.at[tt].set(vec_u_t)
            tns_Q = tns_Q.at[tt].set(mat_Q_t)
            tns_Sigma = tns_Sigma.at[tt].set(mat_Sigma_t)

            return mat_u, tns_Q, tns_Sigma

        # A slow, but logically correct for-loop calculation
        carry = jax.lax.fori_loop(
            lower=1,
            upper=num_sample,
            body_fun=_body_fun,
            init_val=(mat_u, tns_Q, tns_Sigma),
        )
        mat_u, tns_Q, tns_Sigma = carry
        
        # Check against pre-computed results
        assert jnp.allclose(mat_u, self.mat_u)

        calc_tns_Sigma = dcc._calc_trajectory_mvar_cov(mat_epsilon=mat_epsilon, mat_sigma = mat_sigma, mat_Q_0=mat_Q_0, vec_delta = vec_delta, mat_Qbar=mat_Qbar)
        assert jnp.allclose(calc_tns_Sigma, tns_Sigma)




if __name__ == "__main__":
    pass






