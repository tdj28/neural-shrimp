"""Build the optional gdmap CUDA library into a new directory outside Git.

Usage: python tools/build_gdmap_cuda.py OUT_DIR [--variant mirror|contract]

The default ``mirror`` variant (--fmad=false with explicit intrinsics) is the
qualified build; ``contract`` lets the compiler fuse multiply-adds and exists
only for the FMA comparison in docs/gdmap-cuda-qualification.md. Compilation
does not initialize CUDA. The directory receives gdmap_cuda.so and a JSON build
receipt; an existing build there is never overwritten.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_shrimp.gdmaps_cuda import DEFAULT_ARCH, ROOT, VARIANTS, build  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="mirror")
    parser.add_argument("--host-compiler", default="g++-12")
    parser.add_argument("--arch", default=DEFAULT_ARCH, help="CUDA architecture, e.g. sm_90 for H100")
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    if out.is_relative_to(ROOT):
        parser.error("build outputs belong outside the Git checkout")
    receipt = build(out, variant=args.variant, host_compiler=args.host_compiler, arch=args.arch)
    print(f"built {receipt['library']}")
    print(f"sha256 {receipt['library_sha256']}")
    print(f"receipt {receipt['receipt']}")


if __name__ == "__main__":
    main()
