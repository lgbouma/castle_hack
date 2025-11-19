"""Sample a transit plus rotation-like SHO signal with tinygp + NumPyro."""

import os

import corner
import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from tinygp import GaussianProcess, kernels

jax.config.update("jax_enable_x64", True)
os.environ.setdefault(
    "XLA_FLAGS", f"--xla_force_host_platform_device_count={os.cpu_count()}"
)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(os.cpu_count())

SHORT_RUN = True  # Set to False to broaden priors and run a longer MCMC.


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


def evaluate_components(params, time, flux):
    transit = transit_model(time, params)
    gp = build_gp(params, time)
    conditioned = gp.condition(flux - params["baseline"] - transit)
    rotation = conditioned.gp.loc
    model = params["baseline"] + transit + rotation
    return transit, rotation, model, params["baseline"]


if __name__ == "__main__":
    key = jax.random.PRNGKey(12345)
    time = jnp.linspace(0.0, 30.0, 2000)

    true_params = {
        "baseline": jnp.array(1.0),
        "t0": jnp.array(2.1),
        "log_period": jnp.log(3.0),
        "log_duration": jnp.log(0.32),
        "log_depth": jnp.log(1.0e-2),
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

    prior_centers = {
        "baseline": float(true_params["baseline"]),
        "t0": float(true_params["t0"]),
        "log_period": float(true_params["log_period"]),
        "log_duration": float(true_params["log_duration"]),
        "log_depth": float(true_params["log_depth"]),
        "log_sho_period": float(true_params["log_sho_period"]),
        "log_Q": float(true_params["log_Q"]),
        "log_S0": float(true_params["log_S0"]),
        "log_jitter": float(true_params["log_jitter"]),
    }
    if SHORT_RUN:
        prior_scales = {
            "baseline": 0.01,
            "t0": 0.1,
            "log_period": 0.05,
            "log_duration": 0.1,
            "log_depth": 0.3,
            "log_sho_period": 0.05,
            "log_Q": 0.2,
            "log_S0": 0.4,
            "log_jitter": 0.2,
        }
        num_warmup, num_samples = 10, 10
    else:
        prior_scales = {
            "baseline": 0.05,
            "t0": 0.3,
            "log_period": 0.2,
            "log_duration": 0.3,
            "log_depth": 0.7,
            "log_sho_period": 0.2,
            "log_Q": 0.5,
            "log_S0": 0.7,
            "log_jitter": 0.4,
        }
        num_warmup, num_samples = 1500, 2000

    def numpyro_model(time, flux):
        params = {}
        params["baseline"] = numpyro.sample(
            "baseline", dist.Normal(prior_centers["baseline"], prior_scales["baseline"])
        )
        params["t0"] = numpyro.sample(
            "t0", dist.Normal(prior_centers["t0"], prior_scales["t0"])
        )
        for name in [
            "log_period",
            "log_duration",
            "log_depth",
            "log_sho_period",
            "log_Q",
            "log_S0",
            "log_jitter",
        ]:
            params[name] = numpyro.sample(
                name, dist.Normal(prior_centers[name], prior_scales[name])
            )
        transit = transit_model(time, params)
        gp = build_gp(params, time)
        residuals = flux - params["baseline"] - transit
        numpyro.factor("gp_log_prob", gp.log_probability(residuals))

    nuts = NUTS(numpyro_model, dense_mass=True)
    rng_key = jax.random.PRNGKey(2024)
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=2,
        progress_bar=True,
    )
    mcmc.run(rng_key, time, flux)
    samples = mcmc.get_samples()

    param_keys = [
        "baseline",
        "t0",
        "log_period",
        "log_duration",
        "log_depth",
        "log_sho_period",
        "log_Q",
        "log_S0",
        "log_jitter",
    ]

    num_total_samples = samples["baseline"].shape[0]
    rng = np.random.default_rng(0)
    draw_count = min(25, num_total_samples)
    draw_indices = rng.choice(num_total_samples, size=draw_count, replace=False)

    model_draws = []
    rotation_draws = []
    transit_draws = []
    baseline_draws = []
    for idx in draw_indices:
        draw_params = {k: samples[k][idx] for k in param_keys}
        transit_d, rotation_d, model_d, base_d = evaluate_components(
            draw_params, time, flux
        )
        model_draws.append(np.asarray(model_d))
        rotation_draws.append(np.asarray(rotation_d))
        transit_draws.append(np.asarray(transit_d))
        baseline_draws.append(float(base_d))
    model_draws = np.asarray(model_draws)
    rotation_draws = np.asarray(rotation_draws)
    transit_draws = np.asarray(transit_draws)
    baseline_draws = np.asarray(baseline_draws)

    model_lo = np.percentile(model_draws, 16, axis=0)
    model_hi = np.percentile(model_draws, 84, axis=0)

    median_params = {k: jnp.median(samples[k]) for k in param_keys}
    (
        median_transit,
        median_rotation,
        median_model,
        median_baseline,
    ) = evaluate_components(median_params, time, flux)

    time_np = np.asarray(time)
    flux_np = np.asarray(flux)
    median_model_np = np.asarray(median_model)
    median_rotation_np = np.asarray(median_rotation)
    median_transit_np = np.asarray(median_transit)

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
    mask_times = (true_transit_times >= time_np.min()) & (
        true_transit_times <= time_np.max()
    )
    true_transit_times = true_transit_times[mask_times]

    ax_components.plot(time_np, flux_np, ".k", ms=2, alpha=0.35, label="data")
    for rot, base in zip(rotation_draws, baseline_draws):
        ax_components.plot(
            time_np,
            base + rot,
            color="C0",
            alpha=0.2,
        )
    for tran, base in zip(transit_draws, baseline_draws):
        ax_components.plot(
            time_np,
            base + tran,
            color="C1",
            alpha=0.2,
        )
    ax_components.plot(
        time_np, median_baseline + median_rotation_np, color="C0", label="rotation draw"
    )
    ax_components.plot(
        time_np, median_baseline + median_transit_np, color="C1", label="transit draw"
    )
    if true_transit_times.size > 0:
        ax_components.plot(
            true_transit_times,
            np.full_like(true_transit_times, np.max(flux_np) + 0.002),
            "^",
            color="C3",
            ms=5,
            label="true transits",
        )
    ax_components.set_ylabel("relative flux")
    ax_components.legend(loc="best")

    ax_data.plot(time_np, flux_np, ".k", ms=2, alpha=0.5, label="data")
    for draw in model_draws:
        ax_data.plot(time_np, draw, color="C0", alpha=0.08)
    ax_data.plot(time_np, median_model_np, color="C0", label="posterior median")
    ax_data.fill_between(
        time_np, model_lo, model_hi, color="C0", alpha=0.2, label="68% posterior"
    )
    if true_transit_times.size > 0:
        ax_data.plot(
            true_transit_times,
            np.full_like(true_transit_times, np.max(flux_np) + 0.002),
            "^",
            color="C3",
            ms=5,
            label="_nolegend_",
        )
    ax_data.set_ylabel("relative flux")
    ax_data.legend(loc="best")

    ax_resid.plot(time_np, flux_np - median_model_np, ".k", ms=2, alpha=0.5)
    ax_resid.axhline(0, color="C1", lw=1)
    ax_resid.set_xlabel("time [days]")
    ax_resid.set_ylabel("residual")

    posterior_medians = {
        "Transit period": float(jnp.exp(jnp.median(samples["log_period"]))),
        "Transit duration": float(jnp.exp(jnp.median(samples["log_duration"]))),
        "Transit depth (ppt)": float(
            1e3 * jnp.exp(jnp.median(samples["log_depth"]))
        ),
        "T0": float(jnp.median(samples["t0"])),
        "Rotation period": float(jnp.exp(jnp.median(samples["log_sho_period"]))),
        "Q": float(jnp.exp(jnp.median(samples["log_Q"]))),
        "S0": float(jnp.exp(jnp.median(samples["log_S0"]))),
        "Jitter": float(jnp.exp(jnp.median(samples["log_jitter"]))),
    }
    true_values = {
        "Transit period": float(jnp.exp(true_params["log_period"])),
        "Transit duration": float(jnp.exp(true_params["log_duration"])),
        "Transit depth (ppt)": float(jnp.exp(true_params["log_depth"]) * 1e3),
        "T0": float(true_params["t0"]),
        "Rotation period": float(jnp.exp(true_params["log_sho_period"])),
        "Q": float(jnp.exp(true_params["log_Q"])),
        "S0": float(jnp.exp(true_params["log_S0"])),
        "Jitter": float(jnp.exp(true_params["log_jitter"])),
    }
    text_lines = ["Assumed vs posterior medians:"]
    for key in true_values:
        text_lines.append(
            f"{key}: true={true_values[key]:.4f}, med={posterior_medians[key]:.4f}"
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

    median_period = posterior_medians["Transit period"]
    phase = ((time_np - posterior_medians["T0"] + 0.5 * median_period) % median_period) - (
        0.5 * median_period
    )
    order = np.argsort(phase)
    detrended = flux_np - median_rotation_np - float(median_baseline)

    n_bins = 100
    bin_edges = np.linspace(-0.5 * median_period, 0.5 * median_period, n_bins + 1)
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
    plt.plot(
        phase[order],
        median_transit_np[order],
        color="C1",
        lw=1.5,
        label="posterior median transit",
    )
    plt.xlabel("phase [days]")
    plt.ylabel("detrended flux")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("sho_transit_phase_fold.png", dpi=200)

    corner_samples = np.column_stack(
        [
            np.asarray(samples["baseline"]),
            np.asarray(samples["t0"]),
            np.asarray(jnp.exp(samples["log_period"])),
            np.asarray(jnp.exp(samples["log_duration"])),
            np.asarray(1e3 * jnp.exp(samples["log_depth"])),
            np.asarray(jnp.exp(samples["log_sho_period"])),
            np.asarray(jnp.exp(samples["log_Q"])),
            np.asarray(jnp.exp(samples["log_S0"])),
            np.asarray(jnp.exp(samples["log_jitter"])),
        ]
    )
    corner_labels = [
        "baseline",
        "t0",
        "period [d]",
        "duration [d]",
        "depth [ppt]",
        "rot period [d]",
        "Q",
        "S0",
        "jitter",
    ]
    corner_truths = [
        prior_centers["baseline"],
        prior_centers["t0"],
        np.exp(prior_centers["log_period"]),
        np.exp(prior_centers["log_duration"]),
        1e3 * np.exp(prior_centers["log_depth"]),
        np.exp(prior_centers["log_sho_period"]),
        np.exp(prior_centers["log_Q"]),
        np.exp(prior_centers["log_S0"]),
        np.exp(prior_centers["log_jitter"]),
    ]
    fig_corner = corner.corner(
        corner_samples,
        labels=corner_labels,
        truths=corner_truths,
        show_titles=True,
        title_kwargs={"fontsize": 10},
    )
    fig_corner.savefig("sho_transit_corner.png", dpi=200)
