import torch
import numpy as np
from datetime import datetime as datetime

import casadi as cas

dtype = torch.float64
dtype = torch.float64
# HACK:
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
print("Running", device, "on a", name)
torch.set_default_device(device)

# ##
# ## simple OLS Data Generation Process
# ##
# # True beta
# N = 50000
# K = 100
# b = np.random.randn(K)
# b[0] = b[0] + 3.0
# # True error std deviation
# sigma_e = 1.0
#
# x = np.c_[np.ones(N), np.random.randn(N, K - 1)]
# y = x.dot(b) + sigma_e * np.random.randn(N)
#
# # estimate parameter vector, errors, sd of errors, and se of parameters
# bols = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
# err = y - x.dot(bols)
# sigma_ols = np.sqrt(err.dot(err) / (x.shape[0] - x.shape[1]))
# se = np.sqrt(
#     err.dot(err) / (x.shape[0] - x.shape[1]) * np.diagonal(np.linalg.inv(x.T.dot(x)))
# )
# # put results together for easy viewing
# ols_parms = np.r_[bols, sigma_ols]
# ols_se = np.r_[se, np.nan]
# indexn = ["b" + str(i) for i in range(K)]
# indexn.extend(["sigma"])
#
# X = torch.tensor(x, dtype=dtype)
# Y = torch.tensor(y, dtype=dtype)
#
#
# # initialize parameter vector:
# #  betas in first K positions and sigma in last
# startvals = np.append(np.random.randn(K), 1.0)
# omega = torch.tensor(startvals, dtype=dtype, requires_grad=True)
#
#
# ##
# ## Model Log-Likelihood (times -1)
# ##
# # (note: Y and X are tensors)
# def ols_loglike(omega):
#     # divide omega into beta, sigma
#     beta = omega[:-1]
#     sigma = omega[-1]
#     # xb (mu_i for each observation)
#     mu = torch.mv(X, beta)
#     # this is normal pdf logged and summed over all observations
#     ll = -(Y.shape[0] / 2.0) * torch.log(2.0 * torch.pi * sigma**2) - (
#         1.0 / (2.0 * sigma**2.0)
#     ) * torch.sum(((Y - mu) ** 2.0))
#     return -1.0 * ll
#
#
# gd = torch.optim.SGD([omega], lr=1e-5)
# history_gd = []
# time_start = datetime.now()
# for i in range(100000):
#     gd.zero_grad()
#     objective = ols_loglike(omega=omega)
#     objective.backward()
#     gd.step()
#     history_gd.append(objective.item())
#     if (i > 1) and (np.abs(history_gd[-1] - history_gd[-2]) < 0.00001):
#         print("Convergence achieved in ", i + 1, " iterations")
#         print("-LogL Value: ", objective.item())
#         print("Mean |gradient|: ", torch.abs(omega.grad).mean().item())
#         break
#
# time_pytorch = datetime.now() - time_start
#
#
# hessian = torch.autograd.functional.hessian(ols_loglike, omega)
# se_torch = torch.sqrt(torch.linalg.diagonal(torch.linalg.inv(hessian)))


z = 0.5
theta = cas.MX.sym("theta")


def skew_density(z: float, theta: cas.MX):
    f = z * theta**2
    return cas.Function("f", [theta], [f], ["theta"], ["r"])


f = skew_density(z=z, theta=theta)

r = f(theta)

cas.gradient(r, theta)

x = cas.SX.sym("x")
f = cas.SX.sym("y")
ll = [ii * x for ii in range(1, 5)]
f = ll[0]
for ii in range(1, len(ll)):
    f += ll[ii]


fun = cas.Function("f", [x], [f])
fun(1)


print(r)

# x = cas.SX.sym("x", 1)
# cas.gradient(f, x)

# ghi = cas.gradient(hi, x)
