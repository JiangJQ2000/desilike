from __future__ import annotations

from glob import glob
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from desilike.samples import Chain
from getdist import plots


CHAIN_DIR = Path(__file__).resolve().parent / "chains" / "Y1_mu_OmDE_physical_velocileptors_mu0prior_emu_scale1.5"
MCMC_PATTERN = "mcmc_beds_emu_max_physical_velocileptors_Y1_mu0_l02_data-Y1_*.npy"
IMP_PATTERN = "mcmc_beds_emu_max_physical_velocileptors_Y1_mu0_l02_data-Y1_imp_*.npy"
BURNIN = 0.3

MCMC_GETDIST_BASE = CHAIN_DIR / "mcmc_merged_burn0p3_getdist"
IMP_GETDIST_BASE = CHAIN_DIR / "imp_merged_burn0p3_getdist"
OUT_FIG = CHAIN_DIR / "triangle_cosmo_mu_mcmc_vs_imp_vs_imp_reweighted_getdist.svg"

OLD_PRIORS = {
    "n_s": {"loc": 0.9649, "scale": 0.02},
    "omega_b": {"loc": 0.02237, "scale": 0.00055},
}

NEW_PRIORS = {
    "n_s": {"loc": 0.9649, "scale": 0.042},
    "omega_b": {"loc": 0.02218, "scale": 0.00055},
}


def load_merged_chain(files):
    chains = [Chain.load(fn).remove_burnin(BURNIN) for fn in files]
    if not chains:
        raise ValueError("No chain files were loaded.")
    return Chain.concatenate(chains)


def select_cosmo_mu_params(chain):
    varied_names = chain.params(varied=True).names()
    preferred = ["h", "omega_cdm", "omega_b", "logA", "n_s", "H0", "Omega_m", "sigma8_m", "S8"]
    params = [name for name in preferred if name in varied_names]
    params += [name for name in varied_names if name.startswith("mu") and name not in params]
    if not params:
        raise ValueError("No cosmology/mu parameters found in varied parameters.")
    if not any(name.startswith("mu") for name in params):
        raise ValueError("No mu parameter found in varied parameters.")
    return params


def gaussian_logpdf(x, loc, scale):
    return -0.5 * ((x - loc) / scale) ** 2 - np.log(scale * np.sqrt(2.0 * np.pi))


def reweight_importance_with_new_priors(imp_gd):
    params = imp_gd.getParams()
    if not hasattr(params, "n_s"):
        raise ValueError("Importance samples do not contain parameter n_s.")
    if not hasattr(params, "omega_b"):
        raise ValueError("Importance samples do not contain parameter omega_b (Omega_b h^2).")

    ns = np.asarray(params.n_s)
    omega_b = np.asarray(params.omega_b)

    dlogprior_ns = gaussian_logpdf(ns, **NEW_PRIORS["n_s"]) - gaussian_logpdf(ns, **OLD_PRIORS["n_s"])
    dlogprior_ob = gaussian_logpdf(omega_b, **NEW_PRIORS["omega_b"]) - gaussian_logpdf(omega_b, **OLD_PRIORS["omega_b"])
    dlogprior = dlogprior_ns + dlogprior_ob

    # reweightAddingLogLikes multiplies weights by exp(-logLikes), so pass -dlogprior.
    imp_gd.reweightAddingLogLikes(-dlogprior)
    return imp_gd


def main():
    mcmc_files = sorted(glob(str(CHAIN_DIR / MCMC_PATTERN)))
    mcmc_files = [fn for fn in mcmc_files if "_imp_" not in Path(fn).stem and "profiles" not in Path(fn).stem]
    imp_files = sorted(glob(str(CHAIN_DIR / IMP_PATTERN)))
    imp_files = [fn for fn in imp_files if "profiles" not in Path(fn).stem]

    if not mcmc_files:
        raise FileNotFoundError(f"No MCMC files found with pattern: {CHAIN_DIR / MCMC_PATTERN}")
    if not imp_files:
        raise FileNotFoundError(f"No importance files found with pattern: {CHAIN_DIR / IMP_PATTERN}")

    print(f"[info] MCMC files: {len(mcmc_files)}")
    print(f"[info] IMP  files: {len(imp_files)}")

    mcmc_chain = load_merged_chain(mcmc_files)
    imp_chain = load_merged_chain(imp_files)
    print(f"[info] MCMC merged size after burn-in: {mcmc_chain.size}")
    print(f"[info] IMP  merged size after burn-in: {imp_chain.size}")

    plot_params = [name for name in select_cosmo_mu_params(mcmc_chain) if name in imp_chain.params().names()]
    if not plot_params:
        raise ValueError("No common cosmology/mu parameters between MCMC and importance chains.")
    if not any(name.startswith("mu") for name in plot_params):
        raise ValueError("No common mu parameter between MCMC and importance chains.")

    print(f"[info] Plot parameters: {plot_params}")

    mcmc_chain.write_getdist(base_fn=str(MCMC_GETDIST_BASE), params=plot_params)
    imp_chain.write_getdist(base_fn=str(IMP_GETDIST_BASE), params=plot_params)
    print(f"[info] Wrote getdist files: {MCMC_GETDIST_BASE}.*")
    print(f"[info] Wrote getdist files: {IMP_GETDIST_BASE}.*")

    mcmc_gd = mcmc_chain.to_getdist(params=plot_params, label="MCMC")
    imp_gd = imp_chain.to_getdist(params=plot_params, label="Importance")
    imp_rw_gd = imp_chain.to_getdist(params=plot_params, label="Importance + new ns/BBN priors")
    imp_rw_gd = reweight_importance_with_new_priors(imp_rw_gd)
    print(
        "[info] Reweighted importance with priors: "
        "n_s~N(0.9649,0.042^2), omega_b(=Omega_b h^2)~N(0.02218,0.00055^2)"
    )

    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.alpha_filled_add = 0.45
    g.triangle_plot(
        [mcmc_gd, imp_gd, imp_rw_gd],
        params=plot_params,
        filled=True,
        contour_colors=["#1f77b4", "#d62728", "#2ca02c"],
        legend_labels=["MCMC", "Importance sampling (old priors)", "Importance sampling (reweighted)"],
    )
    g.export(str(OUT_FIG))
    print(f"[info] Saved triangle: {OUT_FIG}")


if __name__ == "__main__":
    main()
