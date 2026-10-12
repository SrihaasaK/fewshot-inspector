"""Download, SHA-256-verify and extract MVTec AD (≈5.3 GB, CC BY-NC-SA 4.0).

    python scripts/download_mvtec.py --dest data/mvtec_ad
    python scripts/download_mvtec.py --dest data/mvtec_ad --verify-only

Routes (docs/method_notes.md §5): anomalib's mydrive URL, then a pinned Hugging Face mirror of the
identical tarball. The dataset is for non-commercial use; cite Bergmann et al. (CVPR 2019, IJCV 2021).
"""

from __future__ import annotations

import argparse
import logging

from inspector.data import download_mvtec, resolve_root, verify_mvtec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Download and verify MVTec AD.")
    ap.add_argument("--dest", default="data/mvtec_ad")
    ap.add_argument("--keep-archive", action="store_true", help="keep the .tar.xz after extraction")
    ap.add_argument("--verify-only", action="store_true", help="only check an existing extraction")
    args = ap.parse_args()
    if args.verify_only:
        counts = verify_mvtec(resolve_root(args.dest))
        logging.info("OK: %d categories, %d train / %d test images", len(counts),
                     sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values()))
        return
    download_mvtec(args.dest, keep_archive=args.keep_archive)


if __name__ == "__main__":
    main()
