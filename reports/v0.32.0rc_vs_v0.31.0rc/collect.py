#!/usr/bin/env python3
"""Collect immutable Git evidence; optionally cache public GitHub PR metadata."""

import argparse
import concurrent.futures
import datetime as dt
import difflib
import json
import re
import ssl
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = "upstream/releases/v0.31.0rc"
NEW = "upstream/releases/v0.32.0rc"
EXTRA = set()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO).decode("utf-8", "replace")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def patch_ids(revision):
    log = subprocess.Popen(["git", "log", "--no-merges", "-p", revision], cwd=REPO, stdout=subprocess.PIPE)
    result = subprocess.check_output(["git", "patch-id", "--stable"], stdin=log.stdout)
    log.stdout.close()
    if log.wait():
        raise RuntimeError("git log failed")
    return {sha: patch_id for patch_id, sha in (line.split() for line in result.decode().splitlines())}


def commits(revision):
    raw = git("log", "--reverse", "--format=%H%x1f%cI%x1f%an%x1f%s%x1f%b%x1e", revision)
    result = []
    for entry in raw.split("\x1e"):
        if not entry.strip():
            continue
        sha, date, author, subject, body = entry.strip("\n").split("\x1f", 4)
        prs = re.findall(r"\(#(\d+)\)", subject)
        files = []
        for line in git("show", "--format=", "--numstat", "--find-renames", sha).splitlines():
            add, delete, name = line.split("\t", 2)
            files.append(
                dict(
                    path=name,
                    added=int(add) if add.isdigit() else 0,
                    deleted=int(delete) if delete.isdigit() else 0,
                    binary=add == "-",
                )
            )
        patch_path = HERE / "evidence" / "commits" / f"{sha}.patch"
        if not patch_path.exists():
            patch_path.write_text(git("show", "--format=fuller", "--find-renames", sha))
        result.append(
            dict(
                sha=sha,
                date=date,
                author=author,
                subject=subject,
                body=body,
                pr=int(prs[-1]) if prs else None,
                files=files,
                patch=str(patch_path.relative_to(HERE)),
            )
        )
    return result


def norm(subject):
    value = re.sub(r"\[[^]]*\]|\(#\d+\)|\(from #\d+\)", "", subject.lower())
    return re.sub(r"\s+", " ", value).strip()


def fetch_pr(number):
    path = HERE / "evidence" / "prs" / f"{number}.json"
    if path.exists():
        return number, "cached"
    url = f"https://api.github.com/repos/vllm-project/vllm-ascend/pulls/{number}"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "vllm-branch-analysis"})
        with urllib.request.urlopen(
            request, timeout=30, context=ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
        ) as response:
            obj = json.load(response)
        # Keep author-provided evidence, not bot-generated reviews or account details.
        selected = {
            key: obj.get(key)
            for key in [
                "number",
                "title",
                "body",
                "html_url",
                "state",
                "merged",
                "merged_at",
                "merge_commit_sha",
                "updated_at",
                "additions",
                "deletions",
                "changed_files",
            ]
        }
        selected.update(fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(), api_url=url)
        write_json(path, selected)
        return number, "ok"
    except Exception as error:
        return number, str(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--github", action="store_true")
    args = parser.parse_args()
    (HERE / "evidence" / "commits").mkdir(parents=True, exist_ok=True)
    (HERE / "evidence" / "prs").mkdir(parents=True, exist_ok=True)
    added = commits(f"{OLD}..{NEW}")
    old_only = commits(f"{NEW}..{OLD}")
    newer_ids = patch_ids(f"{OLD}..{NEW}")
    older_ids = patch_ids(f"{NEW}..{OLD}")
    index = {}
    for commit in old_only:
        pid = older_ids.get(commit["sha"])
        if pid:
            index.setdefault(pid, []).append(commit["pr"])
    for commit in added:
        commit["patch_id"] = newer_ids.get(commit["sha"])
        commit["equivalent_old_prs"] = index.get(commit["patch_id"], [])
    for commit in old_only:
        commit["patch_id"] = older_ids.get(commit["sha"])
        commit["equivalent_new_prs"] = [
            c["pr"] for c in added if commit["patch_id"] and c["patch_id"] == commit["patch_id"]
        ]
        scored = sorted(
            ((difflib.SequenceMatcher(None, norm(commit["subject"]), norm(c["subject"])).ratio(), c) for c in added),
            key=lambda item: item[0],
            reverse=True,
        )
        commit["similar_new"] = [
            dict(pr=c["pr"], subject=c["subject"], score=round(score, 3)) for score, c in scored[:3]
        ]
    numstat = []
    for line in git("diff", "--numstat", "--find-renames", OLD, NEW).splitlines():
        add, delete, name = line.split("\t", 2)
        numstat.append(
            dict(
                path=name,
                added=int(add) if add.isdigit() else 0,
                deleted=int(delete) if delete.isdigit() else 0,
                binary=add == "-",
            )
        )
    meta = dict(
        repo=".",
        remote=git("remote", "get-url", "upstream").strip(),
        old_ref=OLD,
        new_ref=NEW,
        old_sha=git("rev-parse", OLD).strip(),
        new_sha=git("rev-parse", NEW).strip(),
        merge_base=git("merge-base", OLD, NEW).strip(),
        old_tip=git("log", "-1", "--format=%cI", OLD).strip(),
        new_tip=git("log", "-1", "--format=%cI", NEW).strip(),
        collected_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        added_count=len(added),
        old_only_count=len(old_only),
        diff_shortstat=git("diff", "--shortstat", OLD, NEW).strip(),
        remote_heads=git(
            "ls-remote", "upstream", "refs/heads/releases/v0.32.0rc", "refs/heads/releases/v0.31.0rc"
        ).strip(),
    )
    write_json(HERE / "git_data.json", dict(meta=meta, added=added, old_only=old_only, net_files=numstat))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("Patch equivalents:", sum(bool(c["equivalent_old_prs"]) for c in added))
    if args.github:
        selected = sorted({c["pr"] for c in added if "[performance]" in c["subject"].lower() or c["pr"] in EXTRA})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for number, status in pool.map(fetch_pr, selected):
                print("PR", number, status, flush=True)


if __name__ == "__main__":
    main()
