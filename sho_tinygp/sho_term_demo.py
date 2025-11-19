import os

import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from tinygp import GaussianProcess, kernels


jax.config.update("jax_enable_x64", True)
os.environ.setdefault(
    "XLA_FLAGS", f"--xla_force_host_platform_device_count={os.cpu_count()}"
)


def build_gp(params, time, noise_sigma):
    period = jnp.exp(params["log_period"])
    Q = jnp.exp(params["log_Q"])
    S0 = jnp.exp(params["log_S0"])
    omega0 = 2 * jnp.pi / period
    kernel = kernels.quasisep.SHO(omega0, Q, S0)
    return GaussianProcess(kernel, time, diag=noise_sigma**2)


if __name__ == "__main__":
    key = jax.random.PRNGKey(123)
    time = jnp.linspace(0.0, 30.0, 500)

    true_period = 5.0  # days
    true_Q = 5.0
    true_S0 = 0.6
    noise_sigma = 5.0e-2

    true_params = {
        "log_period": jnp.log(true_period),
        "log_Q": jnp.log(true_Q),
        "log_S0": jnp.log(true_S0),
    }
    gp_signal = build_gp(true_params, time, noise_sigma=0.0)
    key_signal, key_noise = jax.random.split(key)
    clean_flux = gp_signal.sample(key_signal)
    noisy_flux = clean_flux + noise_sigma * jax.random.normal(
        key_noise, shape=time.shape
    )

    initial_params = {
        "log_period": jnp.log(true_period) + 0.2,
        "log_Q": jnp.log(true_Q / 2),
        "log_S0": jnp.log(true_S0 * 1.5),
        "log_jitter": jnp.log(noise_sigma),
    }

    def objective(params):
        gp = build_gp(
            {
                "log_period": params["log_period"],
                "log_Q": params["log_Q"],
                "log_S0": params["log_S0"],
            },
            time,
            noise_sigma=jnp.exp(params["log_jitter"]),
        )
        return -gp.log_probability(noisy_flux)

    solver = jaxopt.ScipyMinimize(fun=jax.jit(objective))
    soln = solver.run(initial_params)

    map_period = float(jnp.exp(soln.params["log_period"]))
    map_Q = float(jnp.exp(soln.params["log_Q"]))
    map_S0 = float(jnp.exp(soln.params["log_S0"]))

    print(f"True period: {true_period:.3f} days")
    print(f"Preferred period: {map_period:.3f} days")
    print(f"Preferred Q: {map_Q:.3f}")
    print(f"Preferred S0: {map_S0:.3f}")

    gp = build_gp(
        {
            "log_period": soln.params["log_period"],
            "log_Q": soln.params["log_Q"],
            "log_S0": soln.params["log_S0"],
        },
        time,
        noise_sigma=jnp.exp(soln.params["log_jitter"]),
    )
    cond = gp.condition(noisy_flux)
    pred_mean = cond.gp.loc
    uncertainty = jnp.sqrt(cond.gp.variance + noise_sigma**2)

    t_np = np.asarray(time)
    flux_np = np.asarray(noisy_flux)
    mean_np = np.asarray(pred_mean)
    unc_np = np.asarray(uncertainty)

    plt.figure(figsize=(8, 4))
    plt.plot(t_np, flux_np, ".k", ms=3, alpha=0.5, label="fake data")
    plt.plot(t_np, mean_np, color="C0", label="MAP prediction")
    plt.fill_between(
        t_np,
        mean_np - unc_np,
        mean_np + unc_np,
        color="C0",
        alpha=0.2,
        label="1σ interval",
    )
    plt.xlabel("time [days]")
    plt.ylabel("relative flux")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("sho_term_fit.png", dpi=200)
