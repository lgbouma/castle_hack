"""Demonstrate fitting a transit plus rotation-like SHO signal with tinygp."""

import os

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from tinygp import GaussianProcess, kernels

jax.config.update("jax_enable_x64", True)
os.environ.setdefault(
    "XLA_FLAGS", f"--xla_force_host_platform_device_count={os.cpu_count()}"
)


def sho_kernel(params):
    period = jnp.exp(params["log_sho_period"])
    Q = jnp.exp(params["log_Q"])
    S0 = jnp.exp(params["log_S0"])
    omega = 2 * jnp.pi / period
    return kernels.quasisep.SHO(omega, Q, S0)


def build_gp(params, time):
    kernel = sho_kernel(params)
    jitter = jnp.exp(params["log_jitter"])
    return GaussianProcess(kernel, time, diag=jitter**2)


def transit_model(time, params):
    depth = jnp.exp(params["log_depth"])
    period = jnp.exp(params["log_period"])
    duration = jnp.exp(params["log_duration"])
    width = 0.1 * duration + 1e-6
    phase = (time - params["t0"] + 0.5 * period) % period - 0.5 * period
    ingress = jnn.sigmoid((phase + 0.5 * duration) / width)
    egress = jnn.sigmoid((phase - 0.5 * duration) / width)
    profile = ingress - egress
    return -depth * profile


if __name__ == "__main__":
    key = jax.random.PRNGKey(12345)
    time = jnp.linspace(0.0, 30.0, 2000)

    true_params = {
        "baseline": jnp.array(1.0),
        "t0": jnp.array(2.1),
        "log_period": jnp.log(3.0),
        "log_duration": jnp.log(0.32),
        "log_depth": jnp.log(1.e-2),
        "log_sho_period": jnp.log(7.0),
        "log_Q": jnp.log(4.0),
        "log_S0": jnp.log(0.01),
        "log_jitter": jnp.log(1.2e-2),
    }

    kernel = sho_kernel(true_params)
    gp_rot = GaussianProcess(kernel, time, diag=1e-12)
    key_signal, key_noise = jax.random.split(key)
    rotation_signal = gp_rot.sample(key_signal)
    transit_signal = transit_model(time, true_params)
    flux_clean = true_params["baseline"] + transit_signal + rotation_signal
    noise_sigma = jnp.exp(true_params["log_jitter"])
    flux = flux_clean + noise_sigma * jax.random.normal(key_noise, shape=time.shape)

    init_params = {
        "baseline": jnp.array(0.99),
        "t0": jnp.array(1.5),
        "log_period": jnp.log(5.0),
        "log_duration": jnp.log(0.25),
        "log_depth": jnp.log(1e-3),
        "log_sho_period": jnp.log(5.0),
        "log_Q": jnp.log(3.0),
        "log_S0": jnp.log(0.2),
        "log_jitter": jnp.log(1e-2),
    }

    def loss(params):
        transit = transit_model(time, params)
        gp = build_gp(params, time)
        residuals = flux - params["baseline"] - transit
        return -gp.log_probability(residuals)

    solver = jaxopt.ScipyMinimize(fun=jax.jit(loss))
    soln = solver.run(init_params)

    map_period = float(jnp.exp(soln.params["log_period"]))
    map_depth = float(jnp.exp(soln.params["log_depth"]))
    map_duration = float(jnp.exp(soln.params["log_duration"]))
    map_t0 = float(soln.params["t0"])
    map_rotation_period = float(jnp.exp(soln.params["log_sho_period"]))

    print("Recovered parameters (MAP):")
    print(f"Transit period: {map_period:.3f} d")
    print(f"Transit duration: {map_duration:.3f} d")
    print(f"Transit depth: {map_depth*1e3:.2f} ppt")
    print(f"T0: {map_t0:.3f} d")
    print(f"Rotation period: {map_rotation_period:.3f} d")

    map_baseline = soln.params["baseline"]
    transit_map = transit_model(time, soln.params)
    gp_map = build_gp(soln.params, time)
    conditioned = gp_map.condition(flux - map_baseline - transit_map)
    rotation_map = conditioned.gp.loc
    rot_std = jnp.sqrt(conditioned.gp.variance + jnp.exp(2 * soln.params["log_jitter"]))
    model_flux = map_baseline + transit_map + rotation_map

    time_np = np.asarray(time)
    flux_np = np.asarray(flux)
    model_np = np.asarray(model_flux)
    rot_np = np.asarray(rotation_map)
    rot_std_np = np.asarray(rot_std)
    transit_np = np.asarray(transit_map)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 9),
        gridspec_kw={"height_ratios": [2.0, 2.0, 1.0, 1.0]},
        sharex=True,
    )
    ax_components, ax_data, ax_resid, ax_text = axes

    true_period = float(jnp.exp(true_params["log_period"]))
    true_t0 = float(true_params["t0"])
    n_min = int(np.floor((time_np.min() - true_t0) / true_period))
    n_max = int(np.ceil((time_np.max() - true_t0) / true_period))
    cycles = np.arange(n_min, n_max + 1)
    true_transit_times = true_t0 + true_period * cycles
    mask_times = (true_transit_times >= time_np.min()) & (true_transit_times <= time_np.max())
    true_transit_times = true_transit_times[mask_times]

    ax_components.plot(time_np, flux_np, ".k", ms=2, alpha=0.4, label="data")
    ax_components.plot(
        time_np,
        map_baseline + rotation_map,
        color="C0",
        label="rotation component",
    )
    ax_components.plot(
        time_np,
        map_baseline + transit_map,
        color="C1",
        label="transit component",
    )
    if true_transit_times.size > 0:
        ax_components.plot(
            true_transit_times,
            np.full_like(true_transit_times, np.max(flux_np) + 0.002),
            "v",
            color="C3",
            ms=5,
            label="true transits",
        )
    ax_components.set_ylabel("relative flux")
    ax_components.legend(loc="best")

    ax_data.plot(time_np, flux_np, ".k", ms=2, alpha=0.5, label="data")
    ax_data.plot(time_np, model_np, color="C0", label="MAP model")
    ax_data.fill_between(
        time_np,
        model_np - rot_std_np,
        model_np + rot_std_np,
        color="C0",
        alpha=0.2,
        label="1σ credible region",
    )
    ax_data.set_ylabel("relative flux")
    ax_data.legend(loc="best")
    if true_transit_times.size > 0:
        ax_data.plot(
            true_transit_times,
            np.full_like(true_transit_times, np.max(flux_np) + 0.002),
            "v",
            color="C3",
            ms=5,
            label="_nolegend_",
        )

    ax_resid.plot(time_np, flux_np - model_np, ".k", ms=2, alpha=0.5)
    ax_resid.axhline(0, color="C1", lw=1)
    ax_resid.set_xlabel("time [days]")
    ax_resid.set_ylabel("residual")

    assumed_vals = {
        "Transit period": float(jnp.exp(true_params["log_period"])),
        "Transit duration": float(jnp.exp(true_params["log_duration"])),
        "Transit depth (ppt)": float(jnp.exp(true_params["log_depth"]) * 1e3),
        "T0": float(true_params["t0"]),
        "Rotation period": float(jnp.exp(true_params["log_sho_period"])),
        "Q": float(jnp.exp(true_params["log_Q"])),
        "S0": float(jnp.exp(true_params["log_S0"])),
        "Jitter": float(jnp.exp(true_params["log_jitter"])),
    }
    recovered_vals = {
        "Transit period": map_period,
        "Transit duration": map_duration,
        "Transit depth (ppt)": map_depth * 1e3,
        "T0": map_t0,
        "Rotation period": map_rotation_period,
        "Q": float(jnp.exp(soln.params["log_Q"])),
        "S0": float(jnp.exp(soln.params["log_S0"])),
        "Jitter": float(jnp.exp(soln.params["log_jitter"])),
    }
    text_lines = ["Assumed vs. recovered parameters:"]
    for key in assumed_vals:
        text_lines.append(
            f"{key}: true={assumed_vals[key]:.4f}, MAP={recovered_vals[key]:.4f}"
        )
    ax_text.text(
        0.01,
        0.98,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
    )
    ax_text.set_axis_off()

    fig.tight_layout()
    fig.savefig("sho_transit_fit.png", dpi=200)

    phase = ((time_np - map_t0 + 0.5 * map_period) % map_period) - 0.5 * map_period
    order = np.argsort(phase)
    detrended = flux_np - rot_np - float(map_baseline)

    n_bins = 100
    bin_edges = np.linspace(-0.5 * map_period, 0.5 * map_period, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_index = np.digitize(phase, bin_edges) - 1
    mask = (bin_index >= 0) & (bin_index < n_bins)
    bin_index = bin_index[mask]
    detrended_masked = detrended[mask]
    counts = np.bincount(bin_index, minlength=n_bins)
    sums = np.bincount(bin_index, weights=detrended_masked, minlength=n_bins)
    sums_sq = np.bincount(bin_index, weights=detrended_masked**2, minlength=n_bins)
    means = np.zeros(n_bins)
    stderr = np.zeros(n_bins)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid]
    variances = sums_sq[valid] / counts[valid] - means[valid] ** 2
    variances = np.clip(variances, a_min=0.0, a_max=None)
    stderr[valid] = np.sqrt(variances / counts[valid])

    plt.figure(figsize=(7, 4))
    plt.plot(
        phase[order],
        detrended[order],
        ".",
        color="0.85",
        ms=2,
        alpha=0.6,
        label="data",
    )
    plt.errorbar(
        bin_centers[valid],
        means[valid],
        yerr=stderr[valid],
        fmt="o",
        color="C0",
        ms=4,
        capsize=2,
        label="binned (100/bin)",
    )
    plt.plot(phase[order], transit_np[order], color="C1", lw=1.5, label="MAP transit")
    plt.xlabel("phase [days]")
    plt.ylabel("detrended flux")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("sho_transit_phase_fold.png", dpi=200)
