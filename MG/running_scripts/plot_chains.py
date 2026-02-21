from glob import glob
from pathlib import Path
import argparse
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"desilike\.samples\.plotting")

from desilike.samples import Chain, plotting, diagnostics


def parse_args():
    p = argparse.ArgumentParser(description="Plot desilike chains (triangle/trace/GR/IACT/Geweke).")
    p.add_argument("--chains-dir", type=Path, default="chains")
    p.add_argument("--chain-prefix", type=str, default="fs_folps_isitgr_fkptjax")
    p.add_argument("--out-dir", type=Path, default=None, help="Defaults to --chains-dir.")
    return p.parse_args()


def main():
    args = parse_args()
    chains_dir = args.chains_dir
    out_dir = args.out_dir or chains_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = chains_dir / f"{args.chain_prefix}_*.npy"
    fns = sorted(glob(str(pattern)))
    fns = [fn for fn in fns if "profiles" not in Path(fn).stem]
    if not fns:
        print(f"[plot] No chain files matched: {pattern}", file=sys.stderr)
        print("[plot] Check --chains-dir / --chain-prefix, or where you ran the sampler.", file=sys.stderr)
        return 1

    chains = [Chain.load(fn) for fn in fns]
    chains = [c.remove_burnin(0.3) for c in chains]

    params = chains[0].params(varied=True)
    gr_eigen = diagnostics.gelman_rubin(chains, params=params, method="eigen")
    gr_diag = diagnostics.gelman_rubin(chains, params=params, method="diag")
    rminus1_eigen = float(np.max(gr_eigen) - 1.0)
    rminus1_diag = float(np.max(gr_diag) - 1.0)
    print(f"[GR] max eigen R-1 = {rminus1_eigen:.6g}")
    print(f"[GR] max diag  R-1 = {rminus1_diag:.6g}")
    rminus1_path = out_dir / "gelman_rubin_rminus1.txt"
    with rminus1_path.open("w") as f:
        f.write("# param  R-1 (diag)\n")
        for param, val in zip(params, gr_diag - 1.0):
            f.write(f"{param.name} {val:.8e}\n")

    plotting.plot_triangle(chains, fn=out_dir / "triangle.svg")
    plotting.plot_trace(chains[0], fn=out_dir / "trace.svg")
    plotting.plot_gelman_rubin(chains, fn=out_dir / "gelman_rubin.svg", multivariate=True, threshold=0.01, offset=-1)
    plotting.plot_autocorrelation_time(chains[0], fn=out_dir / "iact.svg")
    plotting.plot_geweke(chains, fn=out_dir / "geweke.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
