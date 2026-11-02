from __future__ import annotations

import argparse
from pathlib import Path

from .backend import DEFAULT_RUN_ROOT, run_backend
from .experiments import format_experiment_matrix
from .status import format_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe-Fair MACPO experiment CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("prepare-quick", "prepare-full", "train-quick"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
        cmd.add_argument(
            "--clone-sources",
            action="store_true",
            help="clone missing upstream repositories (requires network access)",
        )
        cmd.add_argument("--cpu", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)

    sub.add_parser("paper-matrix")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "paper-matrix":
        print(format_experiment_matrix())
        return

    if args.command == "status":
        print(format_status(args.run_root))
        return

    quick = args.command != "prepare-full"
    run_training = args.command == "train-quick"
    run_backend(
        run_root=args.run_root,
        quick=quick,
        run_training=run_training,
        clone_official_repos=args.clone_sources,
        use_cuda=not args.cpu,
    )


if __name__ == "__main__":
    main()
