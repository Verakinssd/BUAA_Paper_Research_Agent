"""Bundle the code versions into one file for a cloud box (the GitHub repo is private).

    python -m benchmarks.pack_for_cloud --out cloud.bundle
    scp cloud.bundle root@HOST:/root/            # then, on the box:
    git init -q paper && cd paper && git fetch -q ../cloud.bundle 'refs/cloud/*:refs/cloud/*'
    git checkout -q -b work refs/cloud/base && bash benchmarks/cloud_setup.sh
    python -m benchmarks.make_variants --root ../variants \\
        --base refs/cloud/base --pr2 refs/cloud/pr2 --pr3 refs/cloud/pr3

Only committed work is packed (the box has no access to your working tree). The printed commit ids
are what the experiments record, so check them against what you meant to run.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from benchmarks.make_variants import REPO, git, rev


def pack(repo, out, refs):
    """refs: {short name: git ref}. Returns {name: sha}. Temporary refs/cloud/* are removed afterwards."""
    shas = {}
    try:
        for name, ref in refs.items():
            git(repo, "update-ref", f"refs/cloud/{name}", git(repo, "rev-parse", ref).stdout.strip())
            shas[name] = rev(repo, f"refs/cloud/{name}")
        git(repo, "bundle", "create", str(out), *[f"refs/cloud/{name}" for name in refs])
    finally:
        for name in refs:
            git(repo, "update-ref", "-d", f"refs/cloud/{name}", check=False)
    return shas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="cloud.bundle")
    parser.add_argument("--base", default="HEAD", help="基线分支（默认当前提交）")
    parser.add_argument("--pr2", default="origin/lsy")
    parser.add_argument("--pr3", default="origin/dev/lxy")
    args = parser.parse_args(argv)
    dirty = git(REPO, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        print("警告：工作区有未提交的改动，它们不会进入 bundle：\n" + dirty, file=sys.stderr)
    shas = pack(REPO, Path(args.out).resolve(), {"base": args.base, "pr2": args.pr2, "pr3": args.pr3})
    print(f"已打包 {args.out}（{Path(args.out).stat().st_size / 1e6:.1f} MB）：" + "，".join(f"{k}={v}" for k, v in shas.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
