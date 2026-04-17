from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

REQUIRED_SUBDIRS = ["trainA", "trainB", "testA", "testB"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CycleGAN training from an already-prepared local repo and dataset."
    )
    p.add_argument("--base_dir", type=Path, default=Path.cwd(),
                   help="ee655 folder containing data/ and pytorch-CycleGAN-and-pix2pix/")
    p.add_argument("--data_dir", type=Path, default=None,
                   help="Dataset root containing trainA/trainB/testA/testB. Defaults to <base_dir>/data")
    p.add_argument("--repo_dir", type=Path, default=None,
                   help="Path to cloned pytorch-CycleGAN-and-pix2pix repo. Defaults to <base_dir>/pytorch-CycleGAN-and-pix2pix")
    p.add_argument("--experiment_name", type=str, default="monet_bg_model")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--load_size", type=int, default=286)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=30)
    p.add_argument("--n_epochs_decay", type=int, default=30)
    p.add_argument("--gpu_ids", type=str, default="0")
    p.add_argument("--resume", action="store_true",
                   help="Resume training using --continue_train")
    p.add_argument("--run_test", action="store_true",
                   help="Run test.py after training completes")
    return p.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def validate_dataset(data_dir: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    print(f"Checking dataset at: {data_dir}")
    for sub in REQUIRED_SUBDIRS:
        subdir = data_dir / sub
        if not subdir.exists():
            raise FileNotFoundError(f"Missing required folder: {subdir}")
        files = [p for p in subdir.iterdir() if p.is_file()]
        if len(files) == 0:
            raise RuntimeError(f"Folder is empty: {subdir}")
        print(f" - {sub}: {len(files)} files")


def validate_repo(repo_dir: Path) -> None:
    train_py = repo_dir / "train.py"
    test_py = repo_dir / "test.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found in repo: {train_py}")
    if not test_py.exists():
        raise FileNotFoundError(f"test.py not found in repo: {test_py}")
    print(f"Using repo at: {repo_dir}")



def main() -> None:
    args = parse_args()

    base_dir = args.base_dir.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else (base_dir / "data")
    repo_dir = args.repo_dir.resolve() if args.repo_dir else (base_dir / "pytorch-CycleGAN-and-pix2pix")

    print("=== Configuration ===")
    print(f"base_dir: {base_dir}")
    print(f"data_dir: {data_dir}")
    print(f"repo_dir: {repo_dir}")
    print(f"experiment_name: {args.experiment_name}")
    print(f"gpu_ids: {args.gpu_ids}")
    print(f"resume: {args.resume}")
    print(f"run_test: {args.run_test}")

    validate_dataset(data_dir)
    validate_repo(repo_dir)

    train_cmd = [
        sys.executable, "train.py",
        "--dataroot", str(data_dir),
        "--name", args.experiment_name,
        "--model", "cycle_gan",
        "--batch_size", str(args.batch_size),
        "--preprocess", "resize_and_crop",
        "--load_size", str(args.load_size),
        "--crop_size", str(args.crop_size),
        "--no_flip",
        "--n_epochs", str(args.n_epochs),
        "--n_epochs_decay", str(args.n_epochs_decay),
        "--gpu_ids", args.gpu_ids,
    ]
    if args.resume:
        train_cmd.append("--continue_train")

    run(train_cmd, cwd=repo_dir)

    ckpt_dir = repo_dir / "checkpoints" / args.experiment_name
    print(f"\nCheckpoints directory: {ckpt_dir}")
    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.iterdir()):
            print(" -", p.name)
    else:
        print("Checkpoint directory not found yet.")

    if args.run_test:
        test_cmd = [
            sys.executable, "test.py",
            "--dataroot", str(data_dir),
            "--name", args.experiment_name,
            "--model", "cycle_gan",
            "--phase", "test",
            "--epoch", "latest",
            "--gpu_ids", args.gpu_ids,
            "--no_dropout",
        ]
        run(test_cmd, cwd=repo_dir)
        results_dir = repo_dir / "results" / args.experiment_name / "test_latest" / "images"
        print(f"Test results directory: {results_dir}")


if __name__ == "__main__":
    main()
