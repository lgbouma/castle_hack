import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt

from tinygp import GaussianProcess, kernels, transforms

jax.config.update("jax_enable_x64", True)

random = np.random.default_rng(123)
N = 256
t = np.sort(random.uniform(0, 10, N))
theta = random.uniform(-np.pi, np.pi, N)
X = np.vstack((t, theta)).T


def build_gp(params):
    kernel1 = jnp.exp(params["log_amp1"]) * transforms.Linear(
        jnp.exp(params["log_scale1"]), kernels.Matern32()
    )
    kernel2 = jnp.exp(params["log_amp2"]) * transforms.Subspace(
        0,
        kernels.ExpSquared(jnp.exp(params["log_scale2"]))
        * kernels.ExpSineSquared(
            scale=jnp.exp(params["log_period"]),
            gamma=jnp.exp(params["log_gamma"]),
        ),
    )
    kernel = kernel1 + kernel2
    return GaussianProcess(kernel, X, diag=jnp.exp(params["log_diag"]))


true_params = {
    "log_amp1": np.log(2.0),
    "log_scale1": np.log([2.0, 0.8]),
    "log_amp2": np.log(2.0),
    "log_scale2": np.log(3.5),
    "log_period": np.log(2.0),
    "log_gamma": np.log(10.0),
    "log_diag": np.log(0.5),
}
gp = build_gp(true_params)
y = gp.sample(jax.random.PRNGKey(5678))

plt.plot(t, y, ".k")
plt.ylim(-6.5, 6.5)
plt.xlim(0, 10)
plt.xlabel("t")
plt.ylabel("y")
plt.savefig('fake_data.png')
plt.close('all')

# The physical (oscillatory) component is not obvious in this dataset because it is swamped by the systematics. Now, we’ll find the maximum likelihood hyperparameters by numerically minimizing the negative log-likelihood function.

import jaxopt

@jax.jit
def loss(params):
    return -build_gp(params).log_probability(y)


solver = jaxopt.ScipyMinimize(fun=loss)
soln = solver.run(true_params)
print("Maximum likelihood parameters:")
print(soln.params)


# Compute the predictive means - note the "kernel" argument
gp = build_gp(soln.params)
mu1 = gp.condition(y, kernel=gp.kernel.kernel1).gp.loc
mu2 = gp.condition(y, kernel=gp.kernel.kernel2).gp.loc

plt.plot(t, y, ".k", mec="none", alpha=0.3)
plt.plot(t, y - mu1, ".k")
plt.plot(t, mu2)

plt.ylim(-6.5, 6.5)
plt.xlim(0, 10)
plt.xlabel("t")
plt.ylabel("y");
plt.savefig('predictive_means.png', dpi=300)


