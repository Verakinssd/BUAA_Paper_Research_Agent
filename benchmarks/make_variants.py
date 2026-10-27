"""Check out the code versions an experiment compares, each as its own working tree.

    python -m benchmarks.make_variants --root ../variants
        base  = the benchmark branch itself (main + benchmark tooling, behaviour unchanged)
        pr2   = base + PR #2 (staged concurrency)
        pr3   = base + PR #3 (cross-encoder rerank)
        both  = base + PR #2 + PR #3

Variants are built by merging the PR branches INTO the benchmark branch, so every checkout also has
the benchmark / evaluation tooling and the shared frozen data. A merge that conflicts is aborted and
reported instead of being resolved for you. Trees are detached (no branch names are created) and get
a copy of the local .env. Also writes variants-pr2.json for `benchmarks.ab_e2e` and checkouts.json.

On a cloud box built from `benchmarks.pack_for_cloud`, pass the bundle's refs:
    --base refs/cloud/base --pr2 refs/cloud/pr2 --pr3 refs/cloud/pr3
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GIT_IDENTITY = ["-c", "user.name=benchmark", "-c", "user.email=benchmark@localhost", "-c", "commit.gpgsign=false"]


def git(cwd, *args, check=True):
    done = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8")
    if check and done.returncode:
        raise RuntimeError(f"git {' '.join(args)} 失败：{(done.stderr or done.stdout).strip()}")
    return done


def rev(cwd, ref):
    return git(cwd, "rev-parse", "--short", ref).stdout.strip()


def build(repo, root, base, merges, env_file=None):
    """merges: {variant: [refs to merge into base]}. Returns {variant: {path, sha, ok, error}}."""
    repo, root = Path(repo), Path(root).resolve()  # absolute: ab_e2e runs from other directories
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, refs in merges.items():
        path = root / name
        if path.exists():
            raise FileExistsError(f"{path} 已存在；换一个 --root，或先 `git worktree remove {path}`")
        git(repo, "worktree", "add", "--detach", str(path), base)
        entry = {"path": str(path), "merged": [], "ok": True, "error": ""}
        for ref in refs:
            merged = git(path, *GIT_IDENTITY, "merge", "--no-edit", "--no-ff", ref, check=False)
            if merged.returncode:
                conflicts = git(path, "diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
                git(path, "merge", "--abort", check=False)
                entry.update(ok=False, error=f"合并 {ref} 冲突：{', '.join(conflicts) or merged.stderr.strip()[:200]}")
                # A half-merged tree would silently measure the wrong code; do not leave it around.
                git(repo, "worktree", "remove", "--force", str(path), check=False)
                entry["path"] = ""
                break
            entry["merged"].append(ref)
        if entry["ok"]:
            entry["sha"] = rev(path, "HEAD")
            if env_file and Path(env_file).is_file():
                shutil.copyfile(env_file, path / ".env")
        out[name] = entry
    return out


def write_configs(root, built):
    """variants-pr2.json (for ab_e2e) and checkouts.json (for rank_eval / stage runs)."""
    root = Path(root)
    (root / "checkouts.json").write_text(json.dumps(built, ensure_ascii=False, indent=2), encoding="utf-8")
    if built.get("base", {}).get("ok") and built.get("pr2", {}).get("ok"):
        variants = [
            {"label": "base-concurrent", "repo": built["base"]["path"], "concurrent": True},
            {"label": "base-control", "repo": built["base"]["path"], "concurrent": True, "control": True},
            {"label": "base-serial", "repo": built["base"]["path"], "concurrent": False},
            {"label": "pr2-staged", "repo": built["pr2"]["path"], "concurrent": True,
             "env": {"PIPELINE_MODE": "staged"}}]
        (root / "variants-pr2.json").write_text(json.dumps(variants, ensure_ascii=False, indent=2), encoding="utf-8")
        return root / "variants-pr2.json"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="放置各版本工作树的目录（建议在仓库之外）")
    parser.add_argument("--base", default="feat/inference-benchmark")
    parser.add_argument("--pr2", default="origin/lsy")
    parser.add_argument("--pr3", default="origin/dev/lxy")
    parser.add_argument("--env", default=str(REPO / ".env"), help="复制到每个工作树的 .env")
    args = parser.parse_args(argv)
    merges = {"base": [], "pr2": [args.pr2], "pr3": [args.pr3], "both": [args.pr2, args.pr3]}
    built = build(REPO, args.root, args.base, merges, args.env)
    for name, entry in built.items():
        print(f"{name:<5} {entry.get('sha', '-'):<8} {'OK' if entry['ok'] else '冲突，已移除'}  "
              f"{entry['path'] or entry['error']}")
    config = write_configs(args.root, built)
    if config:
        print(f"端到端 A/B 配置：{config}")
    return 0 if all(e["ok"] for e in built.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
