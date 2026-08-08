from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_oof_targeted_matrix import detector_state_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Prove that full KD and exported deploy checkpoints contain the same detector tensors.")
    parser.add_argument("--full", required=True)
    parser.add_argument("--deploy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    full, deploy = Path(args.full), Path(args.deploy)
    full_sha = detector_state_fingerprint(full)
    deploy_sha = detector_state_fingerprint(deploy)
    report = {
        "format": 1,
        "full": str(full.resolve()),
        "deploy": str(deploy.resolve()),
        "full_detector_sha256": full_sha,
        "deploy_detector_sha256": deploy_sha,
        "parity": full_sha == deploy_sha,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["parity"]:
        raise SystemExit("Detector parity failed; do not evaluate this deployment checkpoint.")


if __name__ == "__main__":
    main()
