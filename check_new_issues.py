#!/usr/bin/env python3
"""
Polls a fixed list of GitHub repos for newly-opened issues since the last run
and pushes a notification via ntfy.sh (free, no-signup push notifications).

State (the timestamp of the last successful check) is stored in
state/last_check.txt and committed back to the repo by the workflow, so each
run only reports issues opened since the previous run.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

# --- Configure your repo list here -----------------------------------------
REPOS = [
    "vllm-project/vllm",
    "vllm-project/guidellm",
    "vllm-project/llm-compressor",
    "ggml-org/llama.cpp",
    "huggingface/transformers",
    "sgl-project/sglang",
    "NVIDIA/TensorRT-LLM",
    "ollama/ollama",
    "huggingface/text-generation-inference",
    "microsoft/DeepSpeed",
    "BerriAI/litellm",
]

# When True, only notify for issues labeled for outside contributors
# (case-insensitive match against WANTED_LABELS below). Set to False to
# get every new issue again.
ONLY_CONTRIBUTOR_LABELS = True
WANTED_LABELS = {"good first issue", "help wanted"}
# -----------------------------------------------------------------------------

STATE_FILE = "state/last_check.txt"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC")
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
MAX_MESSAGE_CHARS = 3800  # stay under ntfy's ~4096 char body limit


def gh_request(url):
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "llm-issue-watcher")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if GITHUB_TOKEN:
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def read_last_check():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            ts = f.read().strip()
            if ts:
                return ts
    # first ever run: only look back 1 hour so we don't spam on setup
    return (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_last_check(ts):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        f.write(ts)


def find_new_issues(since):
    found = []
    for repo in REPOS:
        query = f"repo:{repo} is:issue created:>={since}"
        url = (
            "https://api.github.com/search/issues?q="
            + urllib.parse.quote(query)
            + "&sort=created&order=asc&per_page=50"
        )
        try:
            data = gh_request(url)
        except urllib.error.HTTPError as e:
            print(f"HTTP error checking {repo}: {e.code} {e.reason}", file=sys.stderr)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"Error checking {repo}: {e}", file=sys.stderr)
            continue

        for item in data.get("items", []):
            # Search API returns PRs too since they're "issues" under the hood;
            # PR items have a "pull_request" key, so skip those.
            if "pull_request" in item:
                continue

            labels = [label["name"] for label in item.get("labels", [])]

            if ONLY_CONTRIBUTOR_LABELS:
                normalized = {l.lower() for l in labels}
                if not normalized & WANTED_LABELS:
                    continue

            found.append(
                {
                    "repo": repo,
                    "title": item["title"],
                    "number": item["number"],
                    "url": item["html_url"],
                    "labels": labels,
                }
            )
    return found


def build_message(issues):
    lines = []
    for iss in issues:
        tag = f" [{', '.join(iss['labels'])}]" if iss["labels"] else ""
        lines.append(f"{iss['repo']} #{iss['number']}: {iss['title']}{tag}\n{iss['url']}")
    message = "\n\n".join(lines)
    if len(message) > MAX_MESSAGE_CHARS:
        message = message[:MAX_MESSAGE_CHARS] + f"\n\n...and more (truncated, {len(issues)} total)"
    return message


def send_ntfy(issues):
    if not NTFY_TOPIC:
        print("NTFY_TOPIC not set; skipping notification. Set it as a repo secret.", file=sys.stderr)
        return
    message = build_message(issues)
    title = f"{len(issues)} new issue(s) in tracked LLM repos"
    req = urllib.request.Request(
        f"{NTFY_SERVER}/{NTFY_TOPIC}",
        data=message.encode("utf-8"),
        method="POST",
        headers={
            "Title": title,
            "Priority": "default",
            "Tags": "github,bulb",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()


def main():
    since = read_last_check()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    issues = find_new_issues(since)
    print(f"Checked {len(REPOS)} repos since {since}: found {len(issues)} new issue(s).")

    if issues:
        send_ntfy(issues)

    write_last_check(now)


if __name__ == "__main__":
    main()
