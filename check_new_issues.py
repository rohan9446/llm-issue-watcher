#!/usr/bin/env python3
"""
Polls a fixed list of GitHub repos for newly-opened issues since the last run
and pushes a notification via ntfy.sh (free, no-signup push notifications).

State is stored as JSON in state/state.json and committed back to the repo
by the workflow. It tracks two things:
  - last_check: the timestamp of the last successful check
  - notified:   a rolling list of "repo#number" keys already notified about

Why both: GitHub's search API can lag a few seconds to a minute behind an
issue actually being created (indexing delay). If we only tracked a plain
timestamp and advanced it to "now" every run, an issue created right at the
boundary of a run (created, but not yet indexed when that run queried) would
permanently fall into the gap and never be reported. To fix this we always
look back a bit further than the last check (OVERLAP_SECONDS) so borderline
issues get re-queried on the next run — and we keep a dedupe list so that
overlap doesn't cause the same issue to be notified twice.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

# --- Configure your repo list here -----------------------------------------
# Each entry is (repo, category). category shows up in the notification so
# you can tell at a glance whether it's an AI/ML repo or general software
# engineering ("SDE"). Add your own categories freely — the tag is just
# whatever string you put here.
REPOS = [
    # --- LLM inference & serving (AI) ---
    ("vllm-project/vllm", "AI"),
    ("vllm-project/guidellm", "AI"),
    ("vllm-project/llm-compressor", "AI"),
    ("vllm-project/vllm-omni", "AI"),
    ("ggml-org/llama.cpp", "AI"),
    ("sgl-project/sglang", "AI"),
    ("NVIDIA/TensorRT-LLM", "AI"),
    ("ollama/ollama", "AI"),
    ("huggingface/text-generation-inference", "AI"),
    ("BerriAI/litellm", "AI"),
    ("lm-sys/FastChat", "AI"),

    # --- Training / fine-tuning / model compression (AI) ---
    ("microsoft/DeepSpeed", "AI"),
    ("huggingface/transformers", "AI"),
    ("huggingface/peft", "AI"),
    ("huggingface/accelerate", "AI"),
    ("unslothai/unsloth", "AI"),
    ("axolotl-ai-cloud/axolotl", "AI"),

    # --- Agent frameworks (AI) ---
    ("langchain-ai/langchain", "AI"),
    ("run-llama/llama_index", "AI"),
    ("microsoft/autogen", "AI"),
    ("crewAIInc/crewAI", "AI"),

    # --- Data engineering / pipelines (AI-adjacent) ---
    ("apache/airflow", "AI"),
    ("ray-project/ray", "AI"),
    ("apache/spark", "AI"),

    # --- Vector search / retrieval (AI) ---
    ("milvus-io/milvus", "AI"),
    ("qdrant/qdrant", "AI"),
    ("chroma-core/chroma", "AI"),

    # --- Data science / ML tooling (AI, strong "good first issue" hygiene) ---
    ("pandas-dev/pandas", "AI"),
    ("scikit-learn/scikit-learn", "AI"),
    ("huggingface/datasets", "AI"),
    ("mlflow/mlflow", "AI"),

    # --- ML app-building frameworks (AI) ---
    ("streamlit/streamlit", "AI"),
    ("gradio-app/gradio", "AI"),

    # --- Microsoft AI/ML (AI, confirmed active labeling) ---
    ("microsoft/LightGBM", "AI"),
    ("microsoft/onnxruntime", "AI"),
    ("microsoft/semantic-kernel", "AI"),

    # --- NVIDIA AI/ML (AI, confirmed active labeling; NeMo moved orgs in 2026) ---
    ("NVIDIA-NeMo/NeMo", "AI"),  # was NVIDIA/NeMo, repo transferred to the new NVIDIA-NeMo org

    # --- Data orchestration (AI-adjacent) ---
    ("PrefectHQ/prefect", "AI"),

    # --- Google (AI, label exists but used sparingly) ---
    ("jax-ml/jax", "AI"),  # was google/jax, org renamed

    # --- AMD (AI) ---
    ("ROCm/rocm-libraries", "AI"),  # confirmed active "good first issue" items

    # --- Databricks-adjacent data lake ecosystem (AI-adjacent / data) ---
    ("delta-io/delta-rs", "AI"),  # confirmed active "good-first-issue" usage

    # --- General software engineering (SDE), confirmed active labeling ---
    ("home-assistant/core", "SDE"),
    ("microsoft/vscode", "SDE"),
    ("rust-lang/rust", "SDE"),        # uses "E-easy" instead of "good first issue"
    ("facebook/react", "SDE"),        # uses "good first bug" instead of "good first issue"
    ("electron/electron", "SDE"),
]

# When True, only notify for issues labeled for outside contributors.
# Matching is substring-based and case-insensitive against WANTED_LABEL_PATTERNS
# below, since projects don't standardize on exact label text (e.g. some use
# "good-first-issue", some use "E-help-wanted", some use "beginner friendly").
# Set to False to get every new issue again, everywhere.
ONLY_CONTRIBUTOR_LABELS = True
WANTED_LABEL_PATTERNS = [
    "good first issue",
    "good-first-issue",
    "goodfirstissue",
    "good first bug",  # e.g. facebook/react
    "good second issue",
    "help wanted",
    "help-wanted",
    "helpwanted",
    "beginner",
    "starter",
    "easy",
    "up for grabs",
    "up-for-grabs",
    "first-timers-only",
    "first timers only",
    "contribution(s) welcome",
    "contributions welcome",
    "low-hanging-fruit",
    "low hanging fruit",
]

# Repos that don't label consistently enough for the filter above to be useful
# (e.g. tracked mainly on JIRA instead of GitHub Issues, or just don't label).
# For these specific repos, ONLY_CONTRIBUTOR_LABELS is ignored and you get
# every new issue instead.
#
# NOTE: this is only worth it for genuinely low-volume repos where "every new
# issue" is still a small, relevant stream. For high-volume/uncurated repos
# (NVIDIA/TensorRT-LLM, vllm-project/vllm-omni turned out this way in
# practice — lots of regular bug reports, nothing pickup-able), exempting
# them just means a flood of noise instead of zero signal, which is worse.
# Left those OFF this list on purpose; they fall back to the strict label
# filter and may simply produce very few/no notifications until they label
# something — that's fine.
NO_LABEL_FILTER_REPOS = {
    "apache/spark",           # issue tracking mostly lives on Apache JIRA
}


def matches_wanted_label(labels):
    lowered = [l.lower() for l in labels]
    return any(pattern in label for label in lowered for pattern in WANTED_LABEL_PATTERNS)
# -----------------------------------------------------------------------------

STATE_FILE = "state/state.json"
OLD_STATE_FILE = "state/last_check.txt"  # previous plain-text format, for one-time migration
OVERLAP_SECONDS = 180  # always re-check the last 3 min too, to cover search-index lag
MAX_NOTIFIED_HISTORY = 2000  # cap dedupe list so the state file doesn't grow forever

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


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            data.setdefault("last_check", None)
            data.setdefault("notified", [])
            return data
        except Exception as e:  # noqa: BLE001
            print(f"Could not parse {STATE_FILE}, starting fresh: {e}", file=sys.stderr)

    # One-time migration from the old plain-text timestamp file, if present.
    if os.path.exists(OLD_STATE_FILE):
        with open(OLD_STATE_FILE) as f:
            ts = f.read().strip()
        if ts:
            return {"last_check": ts, "notified": []}

    return {"last_check": None, "notified": []}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state["notified"] = state["notified"][-MAX_NOTIFIED_HISTORY:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def compute_since(last_check):
    if last_check:
        last_dt = datetime.strptime(last_check, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        since_dt = last_dt - timedelta(seconds=OVERLAP_SECONDS)
    else:
        # first ever run: only look back 1 hour so we don't spam on setup
        since_dt = datetime.now(timezone.utc) - timedelta(hours=1)
    return since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def find_new_issues(since):
    found = []
    for repo, category in REPOS:
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

            if ONLY_CONTRIBUTOR_LABELS and repo not in NO_LABEL_FILTER_REPOS:
                if not matches_wanted_label(labels):
                    continue

            found.append(
                {
                    "repo": repo,
                    "category": category,
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
        lines.append(
            f"[{iss['category']}] {iss['repo']} #{iss['number']}: {iss['title']}{tag}\n{iss['url']}"
        )
    message = "\n\n".join(lines)
    if len(message) > MAX_MESSAGE_CHARS:
        message = message[:MAX_MESSAGE_CHARS] + f"\n\n...and more (truncated, {len(issues)} total)"
    return message


def send_ntfy(issues):
    if not NTFY_TOPIC:
        print("NTFY_TOPIC not set; skipping notification. Set it as a repo secret.", file=sys.stderr)
        return
    message = build_message(issues)
    counts_by_category = {}
    for iss in issues:
        counts_by_category[iss["category"]] = counts_by_category.get(iss["category"], 0) + 1
    breakdown = ", ".join(f"{n} {cat}" for cat, n in sorted(counts_by_category.items()))
    title = f"{len(issues)} new issue(s): {breakdown}"
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
    state = load_state()
    since = compute_since(state["last_check"])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    matching = find_new_issues(since)

    already_notified = set(state["notified"])
    new_issues = [
        iss for iss in matching
        if f"{iss['repo']}#{iss['number']}" not in already_notified
    ]

    print(
        f"Checked {len(REPOS)} repos since {since} (includes {OVERLAP_SECONDS}s overlap): "
        f"{len(matching)} matching issue(s), {len(new_issues)} not yet notified."
    )

    if new_issues:
        send_ntfy(new_issues)
        for iss in new_issues:
            state["notified"].append(f"{iss['repo']}#{iss['number']}")

    state["last_check"] = now
    save_state(state)


if __name__ == "__main__":
    main()
