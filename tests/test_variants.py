import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from benchmarks import make_variants, pack_for_cloud
from benchmarks.make_variants import GIT_IDENTITY, git


def commit(repo, branch, filename, content, base=None):
    if base:
        git(repo, "checkout", "-q", "-B", branch, base)
    (Path(repo) / filename).write_text(content, encoding="utf-8")
    git(repo, "add", filename)
    git(repo, *GIT_IDENTITY, "commit", "-q", "-m", f"{branch}: {filename}")


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "base")
        (self.repo / "a.txt").write_text("a\n", encoding="utf-8")
        (self.repo / "b.txt").write_text("b\n", encoding="utf-8")
        git(self.repo, "add", ".")
        git(self.repo, *GIT_IDENTITY, "commit", "-q", "-m", "base")
        commit(self.repo, "pr2", "a.txt", "a from pr2\n", base="base")
        commit(self.repo, "pr3", "b.txt", "b from pr3\n", base="base")
        commit(self.repo, "clash", "a.txt", "a from clash\n", base="base")
        git(self.repo, "checkout", "-q", "base")
        self.root = Path(self.tmp.name) / "variants"

    def read(self, name, filename):
        return (self.root / name / filename).read_text(encoding="utf-8")

    def test_variants_contain_exactly_the_merged_changes(self):
        env = Path(self.tmp.name) / "env"
        env.write_text("SECRET_FREE=1\n", encoding="utf-8")
        built = make_variants.build(self.repo, self.root, "base",
                                    {"base": [], "pr2": ["pr2"], "pr3": ["pr3"], "both": ["pr2", "pr3"]}, env)
        self.assertTrue(all(b["ok"] for b in built.values()), built)
        self.assertEqual((self.read("base", "a.txt"), self.read("base", "b.txt")), ("a\n", "b\n"))
        self.assertEqual((self.read("pr2", "a.txt"), self.read("pr2", "b.txt")), ("a from pr2\n", "b\n"))
        self.assertEqual((self.read("pr3", "a.txt"), self.read("pr3", "b.txt")), ("a\n", "b from pr3\n"))
        self.assertEqual((self.read("both", "a.txt"), self.read("both", "b.txt")), ("a from pr2\n", "b from pr3\n"))
        self.assertEqual(built["both"]["merged"], ["pr2", "pr3"])
        self.assertEqual(self.read("pr2", ".env"), "SECRET_FREE=1\n")
        self.assertEqual(git(self.repo, "branch", "--list", "pr2-*").stdout.strip(), "")  # detached, no new branches

    def test_a_conflicting_merge_is_aborted_and_reported(self):
        built = make_variants.build(self.repo, self.root, "base", {"clash": ["pr2", "clash"]})
        entry = built["clash"]
        self.assertFalse(entry["ok"])
        self.assertIn("a.txt", entry["error"])
        self.assertEqual(entry["merged"], ["pr2"])
        self.assertEqual(entry["path"], "")
        self.assertFalse((self.root / "clash").exists())  # a half-merged tree would measure the wrong code
        self.assertNotIn("clash", git(self.repo, "worktree", "list").stdout)

    def test_existing_directory_is_never_reused(self):
        (self.root / "base").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            make_variants.build(self.repo, self.root, "base", {"base": []})

    def test_ab_config_is_written_only_when_base_and_pr2_built(self):
        built = make_variants.build(self.repo, self.root, "base", {"base": [], "pr2": ["pr2"]})
        path = make_variants.write_configs(self.root, built)
        variants = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([v["label"] for v in variants], ["base-concurrent", "base-control", "base-serial", "pr2-staged"])
        self.assertTrue(variants[1]["control"])
        self.assertEqual(variants[3]["env"], {"PIPELINE_MODE": "staged"})
        self.assertEqual(variants[0]["repo"], variants[1]["repo"])
        self.assertTrue((self.root / "checkouts.json").is_file())
        other = Path(self.tmp.name) / "other"
        broken = make_variants.build(self.repo, other, "base", {"base": [], "pr2": ["pr2", "clash"]})
        self.assertIsNone(make_variants.write_configs(other, broken))


class PackTests(unittest.TestCase):
    def test_bundle_carries_the_named_refs_and_leaves_no_temporary_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            git(repo, "init", "-q", "-b", "base")
            (repo / "f.txt").write_text("1\n", encoding="utf-8")
            git(repo, "add", ".")
            git(repo, *GIT_IDENTITY, "commit", "-q", "-m", "base")
            commit(repo, "pr2", "f.txt", "2\n", base="base")
            commit(repo, "pr3", "g.txt", "3\n", base="base")
            bundle = Path(tmp) / "x.bundle"
            shas = pack_for_cloud.pack(repo, bundle, {"base": "base", "pr2": "pr2", "pr3": "pr3"})
            self.assertEqual(git(repo, "for-each-ref", "refs/cloud").stdout.strip(), "")
            box = Path(tmp) / "box"
            box.mkdir()
            git(box, "init", "-q", "-b", "work")
            git(box, "fetch", "-q", str(bundle), "refs/cloud/*:refs/cloud/*")
            for name, ref in (("base", "base"), ("pr2", "pr2"), ("pr3", "pr3")):
                self.assertEqual(shas[name], git(box, "rev-parse", "--short", f"refs/cloud/{name}").stdout.strip())
                self.assertEqual(shas[name], git(repo, "rev-parse", "--short", ref).stdout.strip())
            # and the box can build the variants from those refs
            git(box, "checkout", "-q", "-b", "work2", "refs/cloud/base")
            built = make_variants.build(box, Path(tmp) / "variants", "refs/cloud/base",
                                        {"both": ["refs/cloud/pr2", "refs/cloud/pr3"]})
            self.assertTrue(built["both"]["ok"], built)
            self.assertEqual((Path(tmp) / "variants" / "both" / "g.txt").read_text(), "3\n")


if __name__ == "__main__":
    unittest.main()
