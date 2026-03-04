from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from desilike.samples import Chain
from getdist import loadMCSamples, plots
from getdist.mcsamples import MCSamples


BURNIN = 0.3
ROOT = Path(__file__).resolve().parent
OUT_FIG = ROOT / "chains" / "triangle_cosmo_mu_7cases_getdist.pdf"
COBAYA_SIGMA0_ABS_MAX = 0.01
COBAYA_MUSIGMA_DIR = (
    Path.home()
    / "projects"
    / "desi"
    / "y1kp7"
    / "fs"
    / "cobaya"
    / "isitgr"
    / "run1"
    / "base_mu_sigma"
    / "desi-reptvelocileptors-fs-bao-all_schoneberg2024-bbn_planck2018-ns10"
)


@dataclass(frozen=True)
class ChainCase:
    tag: str
    label: str
    pattern: str
    exclude_imp_from_wildcard: bool = False
    loader: str = "desilike"
    sigma0_abs_max: float | None = None


CASES = [
    # ChainCase(
    #     tag="1",
    #     label="① Y1 FS fkpt (physical_velocileptors), w/ emu",
    #     pattern=str(
    #         ROOT
    #         / "chains"
    #         / "Y1_mu_OmDE_physical_velocileptors_mu0prior_emu_scale1.5"
    #         / "mcmc_beds_emu_max_physical_velocileptors_Y1_mu0_l02_data-Y1_*.npy"
    #     ),
    #     exclude_imp_from_wildcard=True,
    # ),
    # ChainCase(
    #     tag="2",
    #     label="② Y1 FS fkpt (physical_velocileptors), IS ①→w/o emu",
    #     pattern=str(
    #         ROOT
    #         / "chains"
    #         / "Y1_mu_OmDE_physical_velocileptors_mu0prior_emu_scale1.5"
    #         / "mcmc_beds_emu_max_physical_velocileptors_Y1_mu0_l02_data-Y1_imp_*.npy"
    #     ),
    # ),
    # ChainCase(
    #     tag="3",
    #     label="③ Y1 FS fkpt (physical_velocileptors), DR1 ns&BBN prior, w/ emu",
    #     pattern=str(
    #         ROOT
    #         / "chains"
    #         / "DR1_OmDE_mu0prior"
    #         / "mcmc_beds_emu_max_physical_velocileptors_Y1_mu0_l02_data-Y1_*.npy"
    #     ),
    #     exclude_imp_from_wildcard=True,
    # ),
    ChainCase(
        tag="4",
        label="④ Y1 FS+BAO fkpt (physical_velocileptors), DR1 ns&BBN prior, w/ emu",
        pattern=str(
            ROOT
            / "chains"
            / "DR1_w_bao_recon_OmDE_mu0prior"
            / "mcmc_beds_emu_max_physical_velocileptors_Y1_mu0_l02_data-Y1_*.npy"
        ),
        exclude_imp_from_wildcard=True,
    ),
    ChainCase(
        tag="5",
        label="⑤ Y1 FS+BAO fkpt (physical_velocileptors), DR1 ns&BBN prior, IS ④→w/o emu",
        pattern=str(
            ROOT
            / "chains"
            / "DR1_w_bao_recon_OmDE_mu0prior"
            / "mcmc_beds_emu_max_physical_velocileptors_Y1_mu0_l02_data-Y1_imp_*.npy"
        ),
    ),
    # ChainCase(
    #     tag="6",
    #     label="⑥ Y1 FS+BAO REPTvelocileptors (physical), DR1 ns&BBN prior, w/ emu",
    #     pattern=str(
    #         ROOT
    #         / "chains"
    #         / "DR1_w_bao_recon_OmDE_mu0prior_reptvelocileptors"
    #         / "mcmc_eds_emu_max_physical_Y1_mu0_l02_data-Y1_*.npy"
    #     ),
    #     exclude_imp_from_wildcard=True,
    # ),
    # ChainCase(
    #     tag="7",
    #     label="⑦ Y1 FS+BAO fkpt (APscaling), DR1 ns&BBN prior, w/ emu",
    #     pattern=str(
    #         ROOT
    #         / "chains"
    #         / "DR1_w_bao_recon_OmDE_mu0prior_fkpt"
    #         / "mcmc_beds_emu_max_APscaling_Y1_mu0_l02_data-Y1_*.npy"
    #     ),
    #     exclude_imp_from_wildcard=True,
    # ),
    # ChainCase(
    #     tag="8",
    #     label="Y1 FS fkpt (APscaling), w/ emu",
    #     pattern=str(
    #         ROOT
    #         / "chains"
    #         / "DR1_w_bao_recon_OmDE_mu0prior_fkpt_prior5"
    #         / "mcmc_beds_emu_max_APscaling_Y1_mu0_l02_data-Y1_*.npy"
    #     ),
    #     exclude_imp_from_wildcard=True,
    # ),
    ChainCase(
        tag="9",
        label=f"y1kp7/fs/cobaya/isitgr/run1/base_mu_sigma/desi-reptvelocileptors-fs-bao-all_schoneberg2024-bbn_planck2018-ns10, keep |Sigma0| <= {COBAYA_SIGMA0_ABS_MAX:.2f}",
        pattern=str(COBAYA_MUSIGMA_DIR / "chain"),
        loader="cobaya_mu_sigma",
        sigma0_abs_max=COBAYA_SIGMA0_ABS_MAX,
    ),
]


def _load_desilike_case(case: ChainCase) -> Chain:
    files = sorted(glob(case.pattern))
    files = [fn for fn in files if "profiles" not in Path(fn).stem]
    if case.exclude_imp_from_wildcard:
        files = [fn for fn in files if "_imp_" not in Path(fn).stem]
    if not files:
        raise FileNotFoundError(f"[case {case.tag}] No files found for pattern: {case.pattern}")

    chains = [Chain.load(fn).remove_burnin(BURNIN) for fn in files]
    merged = Chain.concatenate(chains)
    print(f"[case {case.tag}] files={len(files)} merged_size={merged.size}")
    return merged


def _load_cobaya_mu_sigma_case(case: ChainCase) -> MCSamples:
    sigma0_abs_max = COBAYA_SIGMA0_ABS_MAX if case.sigma0_abs_max is None else case.sigma0_abs_max
    samples = loadMCSamples(case.pattern, settings={"ignore_rows": BURNIN}, no_cache=True)
    params = samples.getParams()
    sigma0 = np.asarray(params.Sigma0)
    mask = np.abs(sigma0) <= sigma0_abs_max
    kept, total = int(np.count_nonzero(mask)), mask.size
    if kept == 0:
        raise ValueError(f"[case {case.tag}] No points satisfy |Sigma0| <= {sigma0_abs_max:.4g}")
    samples.filter(mask)

    params = samples.getParams()
    h = params.H0 / 100.0
    samples.addDerived(h, name="h", label="h")
    samples.addDerived(params.omch2, name="omega_cdm", label=r"\omega_{\mathrm{cdm}}")
    samples.addDerived(params.ombh2, name="omega_b", label=r"\omega_{\mathrm{b}}")
    samples.addDerived(params.ns, name="n_s", label=r"n_{\mathrm{s}}")
    samples.addDerived(params.omegam, name="Omega_m", label=r"\Omega_{\mathrm{m}}")
    samples.addDerived(params.sigma8, name="sigma8_m", label=r"\sigma_{8,\mathrm{m}}")
    samples.addDerived(params.sigma8 * np.sqrt(params.omegam / 0.3), name="S8", label=r"S_8")
    samples.updateBaseStatistics()
    samples.label = case.label
    print(
        f"[case {case.tag}] root={case.pattern} kept={kept}/{total} "
        f"({kept / total:.2%}) with |Sigma0| <= {sigma0_abs_max:.4g}, merged_size={samples.samples.shape[0]}"
    )
    return samples


def load_case(case: ChainCase) -> Chain | MCSamples:
    if case.loader == "desilike":
        return _load_desilike_case(case)
    if case.loader == "cobaya_mu_sigma":
        return _load_cobaya_mu_sigma_case(case)
    raise ValueError(f"Unknown loader '{case.loader}' for case {case.tag}")


def choose_plot_params(chains: list[Chain]) -> list[str]:
    common = set(chains[0].params(varied=True).names())
    for chain in chains[1:]:
        common &= set(chain.params(varied=True).names())

    preferred = ["h", "omega_cdm", "omega_b", "logA", "n_s", "H0", "Omega_m", "sigma8_m", "S8"]
    params = [name for name in preferred if name in common]
    params += [name for name in sorted(common) if name.startswith("mu") and name not in params]

    if not params:
        raise ValueError("No common cosmology/mu parameters found across all cases.")
    if not any(name.startswith("mu") for name in params):
        raise ValueError("No common mu parameter found across all cases.")
    return params


def main():
    loaded = [load_case(case) for case in CASES]
    desilike_chains = [item for item in loaded if isinstance(item, Chain)]
    plot_params = choose_plot_params(desilike_chains)
    print(f"[info] Plot parameters: {plot_params}")

    samples = []
    for case, item in zip(CASES, loaded):
        if isinstance(item, Chain):
            samples.append(item.to_getdist(params=plot_params, label=case.label))
        else:
            item.label = case.label
            samples.append(item)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    g = plots.get_subplot_plotter(width_inch=10)
    g.settings.alpha_filled_add = 0.3
    g.settings.figure_legend_frame = True
    contour_ls = ["--" if case.loader == "cobaya_mu_sigma" else "-" for case in CASES]
    g.triangle_plot(
        samples,
        params=plot_params,
        filled=False,
        contour_colors=["#d62728", "#2ca02c", "#9467bd", "#1f77b4", ],
        contour_ls=contour_ls,
        legend_labels=[case.label for case in CASES],
        markers={"mu0": 0},
    )
    g.export(str(OUT_FIG))
    print(f"[info] Saved triangle: {OUT_FIG}")


if __name__ == "__main__":
    main()
