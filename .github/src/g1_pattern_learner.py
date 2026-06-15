import os
import json
import httpx
import pathlib
from datetime import datetime
from google import genai
from google.genai import types
import copy
from pathlib import Path

# --- Config from environment ---
GH_TOKEN = os.environ["GH_TOKEN"]
GH_REPO = os.environ["GH_REPO"]
GH_OWNER = os.environ["GH_OWNER"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
owner, repo = GH_OWNER, GH_REPO

raw_prs = os.environ.get("PR_NUMBERS", "")
pr_numbers = []
for p in raw_prs.split(","):
    p = p.strip()
    if p.isdigit():
        pr_numbers.append(int(p))
pr_numbers = list(dict.fromkeys(pr_numbers))
if not pr_numbers:
    raise ValueError("PR_NUMBERS env var is required. Provide comma-separated PR numbers.")
if len(pr_numbers) > 10:
    raise ValueError("Maximum 10 PR numbers allowed.")
print(f"Training on PRs: {pr_numbers}")

raw_seniors = os.environ.get("SENIOR_REVIEWERS", "")
if raw_seniors.strip():
    SENIOR_REVIEWERS = [s.strip() for s in raw_seniors.split(",") if s.strip()][:10]
else:
    SENIOR_REVIEWERS = []

# Persist senior reviewers to team_config.json
TEAM_CONFIG_FILE = "../pattern_store/team_config.json"
existing_config = {}
if os.path.exists(TEAM_CONFIG_FILE):
    try:
        with open(TEAM_CONFIG_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                existing_config = json.loads(content)
    except json.JSONDecodeError:
        print(f"Warning: team_config.json contains invalid JSON. Using defaults.")
        existing_config = {}
existing_seniors = existing_config.get("senior_reviewers", [])
merged_seniors = list(dict.fromkeys(existing_seniors + SENIOR_REVIEWERS))
existing_config["senior_reviewers"] = merged_seniors
existing_config["last_updated"] = datetime.now().isoformat()
with open(TEAM_CONFIG_FILE, "w", encoding="utf-8") as f:
    f.write(json.dumps(existing_config, indent=2))
SENIOR_REVIEWERS = merged_seniors
print(f"Senior reviewers: {SENIOR_REVIEWERS}")

client = genai.Client(api_key=GEMINI_API_KEY)

CATEGORY_DEFINITIONS = {
    "security": "SQL injection, auth bypass, secrets exposure, CSRF, XSS, insecure dependencies, missing authorization checks, cryptography issues, unsafe deserialization.",
    "logic": "Incorrect conditions, missing error handling, race conditions, edge cases, null handling, async bugs, data integrity issues, performance risks like N+1 queries.",
    "style": "Naming clarity, function complexity, SRP violations, missing docstrings or type hints, dead code, duplicate logic, maintainability issues, inconsistent conventions."
}

PATTERN_EXTRACTION_PROMPT = """You are a senior engineering lead analysing pull request review comments to extract reusable review patterns for a team's automated code reviewer.

You will receive:
1. A list of review comments from a PR, each with author, body, diff_hunk (if available), and seniority flag.
2. The raw PR diff for context.
3. Definitions of three standard categories: security, logic, style.

Your task:

PART A — PATTERN EXTRACTION

For each comment, determine if it describes a reusable review pattern (something this team consistently cares about) or a one-off observation specific to that PR's unique context.

One-off: "This variable name is confusing given what this specific endpoint does."
Pattern: "External API calls must always include timeout and error handling."

For each pattern comment:
1. Classify it against the three category definitions provided. Use those definitions as your primary reference.
2. If it fits one category: assign that category.
3. If it fits multiple categories (cross-cutting): assign all matching categories, set cross_cutting to true, and pick the primary_category it most strongly belongs to.
4. If it fits none: create a new category with a snake_case name and a one-sentence definition.
5. Extract the pattern as a general, reusable rule — not tied to the specific file or variable name in the comment.

For each extracted pattern, output:
{
  "pattern_id": "P-{3-digit-number}",
  "categories": ["security", "logic"],
  "primary_category": "security",
  "cross_cutting": true,
  "description": "All external API calls must define explicit timeouts and handle network errors.",
  "example_violation": "const data = await fetch(url)",
  "example_fix": "const data = await fetchWithTimeout(url, { timeout: 5000 })",
  "source_comment": "{first 100 chars of original comment body}",
  "author": "{author username}",
  "is_senior": true,
  "created_at": "{ISO timestamp}",
  "pr_number": {pr_number}
}

PART B — WHAT THE TEAM DOES NOT CARE ABOUT

Analyse the PR diff and identify issues that a generic reviewer would typically flag but that none of the human reviewers commented on. These represent implicit team tolerances — things this team has decided not to enforce.

For each identified non-pattern:
{
  "description": "Single-letter variable names in short closures are not flagged.",
  "category": "style",
  "evidence": "Diff contains multiple single-letter variables with no reviewer comment."
}

PART C — TONE ANALYSIS

Analyse the tone and style of the review comments. Extract tone per severity level based on how the reviewers actually wrote their comments. Also extract the response style from any reply comments where the author responded to reviewer feedback.

For severity mapping: infer severity from the language used — imperative/urgent language maps to critical/high, collaborative/question-framed maps to medium, nit-prefixed or one-line maps to low.

Output:
{
  "review_tone": {
    "critical": "{describe the tone pattern for critical issues}",
    "high": "{describe the tone pattern for high severity}",
    "medium": "{describe the tone pattern for medium severity}",
    "low": "{describe the tone pattern for low severity}"
  },
  "response_style": {
    "description": "{how the team responds after fixing — terse/verbose, pronouns/no pronouns, etc.}",
    "example": "{best example reply found in the comments}",
    "avoid": "{what the team's responses explicitly avoid}"
  }
}

OUTPUT FORMAT:
Return ONLY valid JSON in this exact structure. No preamble, no explanation, no markdown fences.

{
  "patterns": [...],
  "non_patterns": [...],
  "tone": { "review_tone": {...}, "response_style": {...} }
}
"""

def fetch_pr_data(pr_num: int) -> dict:
    headers = {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    comments_resp = httpx.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}/comments",
        headers=headers
    )
    reviews_resp = httpx.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}/reviews",
        headers=headers
    )
    diff_resp = httpx.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}",
        headers={**headers, "Accept": "application/vnd.github.v3.diff"}
    )
    all_comments = []
    for c in comments_resp.json():
        all_comments.append({
            "source": "inline",
            "author": c["user"]["login"],
            "is_senior": c["user"]["login"] in SENIOR_REVIEWERS,
            "body": c["body"],
            "path": c["path"],
            "line": c.get("line") or c.get("original_line"),
            "diff_hunk": c["diff_hunk"],
            "created_at": c["created_at"],
            "pr_number": pr_num
        })
    for r in reviews_resp.json():
        if r.get("state") in ("CHANGES_REQUESTED", "COMMENTED") and r.get("body"):
            all_comments.append({
                "source": "review",
                "author": r["user"]["login"],
                "is_senior": r["user"]["login"] in SENIOR_REVIEWERS,
                "body": r["body"],
                "path": None,
                "line": None,
                "diff_hunk": None,
                "created_at": r["submitted_at"],
                "pr_number": pr_num
            })
    return {"comments": all_comments, "diff": diff_resp.text}

# Fetch all PRs
all_comments = []
combined_diff_parts = []
for pr_num in pr_numbers:
    print(f"Fetching PR #{pr_num}...")
    data = fetch_pr_data(pr_num)
    all_comments.extend(data["comments"])
    combined_diff_parts.append(f"--- PR #{pr_num} DIFF ---\n{data['diff'][:2000]}")
combined_diff = "\n\n".join(combined_diff_parts)
pr_data = {"comments": all_comments, "diff": combined_diff}
print(f"Fetched {len(all_comments)} total comments across {len(pr_numbers)} PRs")

# Paste extract_patterns(), compute_recency_score(), compute_confidence(),
# rank_patterns(), load_json_store(), merge_patterns(), save_json_store()
# exactly as they are in L9 cells 3-6
def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned

def extract_patterns(pr_data: dict) -> dict:
    comments_json = json.dumps(pr_data["comments"], indent=2)
    diff_preview = pr_data["diff"][:3000]
    user_message = f"""CATEGORY DEFINITIONS:
{json.dumps(CATEGORY_DEFINITIONS, indent=2)}

REVIEW COMMENTS:
{comments_json}

PR DIFF (first 3000 chars for context):
{diff_preview}

PR NUMBERS ANALYSED: {pr_numbers}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=PATTERN_EXTRACTION_PROMPT,
        )
    )

    raw_response = strip_markdown_fences(response.text or "")
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        print("Warning: Gemini returned invalid JSON. Raw response:")
        print(raw_response)
        return {"patterns": [], "non_patterns": [], "tone": {}}

def compute_recency_score(created_at_str: str) -> float:
    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
    now = datetime.now(created_at.tzinfo) if created_at.tzinfo else datetime.now()
    days_since = (now - created_at).days
    if days_since <= 90:
        return 1.0
    if days_since <= 365:
        return 0.6
    return 0.2

def compute_confidence(pattern: dict) -> float:
    seniority_boost = 1.0 if pattern.get("is_senior") else 0.5
    recency = compute_recency_score(pattern["created_at"])
    frequency = pattern.get("frequency", 1)
    return round((frequency * 0.5) + (seniority_boost * 0.3) + (recency * 0.2), 3)

def rank_patterns(patterns: list) -> dict:
    ranked_patterns = []
    for pattern in patterns:
        pattern_copy = copy.deepcopy(pattern)
        pattern_copy["frequency"] = pattern_copy.get("frequency", 1)
        pattern_copy["confidence"] = compute_confidence(pattern_copy)
        ranked_patterns.append(pattern_copy)

    ranked_patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)

    by_category = {}
    primary_category_groups = {}
    for pattern in ranked_patterns:
        categories = pattern.get("categories") or [pattern.get("primary_category", "uncategorized")]
        for category in categories:
            by_category.setdefault(category, []).append(copy.deepcopy(pattern))

        primary_category = pattern.get("primary_category") or categories[0]
        primary_category_groups.setdefault(primary_category, []).append(copy.deepcopy(pattern))

    for category_patterns in by_category.values():
        category_patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
    for category_patterns in primary_category_groups.values():
        category_patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)

    top3_per_category = {
        category: category_patterns[:3]
        for category, category_patterns in primary_category_groups.items()
    }

    return {
        "by_category": by_category,
        "top3_per_category": top3_per_category,
        "all_patterns": ranked_patterns,
    }


def load_json_store(filepath: str, default: dict) -> dict:
    path = Path(filepath)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return copy.deepcopy(default)

def description_prefix(item: dict) -> str:
    return (item.get("description") or "")[:60].strip().casefold()


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def merge_patterns(existing_store: dict, new_ranked: dict, new_non_patterns: list, pr_nums: list) -> dict:
    updated_store = copy.deepcopy(existing_store)
    existing_patterns = updated_store.get("patterns", [])

    for new_pattern in new_ranked["all_patterns"]:
        new_copy = copy.deepcopy(new_pattern)
        new_prefix = description_prefix(new_copy)
        matched_pattern = next(
            (p for p in existing_patterns if description_prefix(p) == new_prefix),
            None,
        )

        if matched_pattern:
            matched_pattern["frequency"] = matched_pattern.get("frequency", 1) + 1
            if parse_iso_datetime(new_copy["created_at"]) > parse_iso_datetime(matched_pattern["created_at"]):
                matched_pattern["created_at"] = new_copy["created_at"]
            matched_pattern["confidence"] = compute_confidence(matched_pattern)
        else:
            existing_patterns.append(new_copy)

    existing_non_patterns = updated_store.get("non_patterns", [])
    for new_non_pattern in new_non_patterns:
        new_copy = copy.deepcopy(new_non_pattern)
        new_prefix = description_prefix(new_copy)
        matched_non_pattern = next(
            (np for np in existing_non_patterns if description_prefix(np) == new_prefix),
            None,
        )

        if matched_non_pattern:
            matched_non_pattern["frequency"] = matched_non_pattern.get("frequency", 1) + 1
        else:
            new_copy["frequency"] = new_copy.get("frequency", 1)
            existing_non_patterns.append(new_copy)

    prs_analysed = updated_store.get("prs_analysed", [])
    for n in pr_nums:
       if n not in prs_analysed:
           prs_analysed.append(n)

    reranked = rank_patterns(existing_patterns)
    updated_store["last_updated"] = datetime.now().isoformat()
    updated_store["prs_analysed"] = prs_analysed
    updated_store["patterns"] = reranked["all_patterns"]
    updated_store["non_patterns"] = existing_non_patterns
    updated_store["top3_per_category"] = reranked["top3_per_category"]
    return updated_store

def save_json_store(filepath: str, data: dict):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))

# Run
extraction_result = extract_patterns(pr_data)
print(f"Extracted {len(extraction_result['patterns'])} patterns")

ranked = rank_patterns(extraction_result["patterns"])

PATTERNS_FILE = "../pattern_store/team_patterns.json"
TONE_FILE = "../pattern_store/team_tone.json"

default_store = {
    "last_updated": "", "prs_analysed": [],
    "patterns": [], "non_patterns": [], "top3_per_category": {}
}
default_tone = {
    "last_updated": "", "prs_analysed": [],
    "review_tone": {
        "critical": {"active": "", "observations": []},
        "high": {"active": "", "observations": []},
        "medium": {"active": "", "observations": []},
        "low": {"active": "", "observations": []}
    },
    "response_style": {"active": {}, "observations": []}
}

existing_store = load_json_store(PATTERNS_FILE, default_store)
updated_store = merge_patterns(existing_store, ranked, extraction_result["non_patterns"], pr_numbers)
save_json_store(PATTERNS_FILE, updated_store)
print(f"✅ team_patterns.json updated. Total patterns: {len(updated_store['patterns'])}")

def merge_tone(existing_tone: dict, new_tone: dict, pr_nums: list) -> dict:
    updated_tone = copy.deepcopy(existing_tone)
    review_tone = new_tone.get("review_tone", {})

    for level in ["critical", "high", "medium", "low"]:
        level_observation = review_tone.get(level)
        if level_observation:
            updated_tone.setdefault("review_tone", {}).setdefault(level, {"active": "", "observations": []})
            observation = f"from PRs {pr_nums}: {level_observation}"
            updated_tone["review_tone"][level].setdefault("observations", []).append(observation)
            updated_tone["review_tone"][level]["active"] = level_observation

    response_style = new_tone.get("response_style", {})
    if response_style:
        response_style_observation = copy.deepcopy(response_style)
        response_style_observation["pr_numbers"] = pr_nums
        updated_tone.setdefault("response_style", {"active": {}, "observations": []})
        updated_tone["response_style"].setdefault("observations", []).append(response_style_observation)
        updated_tone["response_style"]["active"] = copy.deepcopy(response_style)

    prs_analysed = updated_tone.get("prs_analysed", [])
    for n in pr_nums:
       if n not in prs_analysed:
           prs_analysed.append(n)

    updated_tone["prs_analysed"] = prs_analysed
    updated_tone["last_updated"] = datetime.now().isoformat()
    return updated_tone


TONE_FILE = "../pattern_store/team_tone.json"
default_tone = {
    "last_updated": "",
    "prs_analysed": [],
    "review_tone": {
        "critical": {"active": "", "observations": []},
        "high": {"active": "", "observations": []},
        "medium": {"active": "", "observations": []},
        "low": {"active": "", "observations": []}
    },
    "response_style": {"active": {}, "observations": []}
}
existing_tone = load_json_store(TONE_FILE, default_tone)
new_tone_data = extraction_result.get("tone", {})
updated_tone = merge_tone(existing_tone, new_tone_data, pr_numbers)
save_json_store(TONE_FILE, updated_tone)
print(f"✅ team_tone.json updated.")

print("\n" + "="*60)
print(f"G1 Pattern Learner — PRs {pr_numbers} complete")
print(f"Total patterns in store: {len(updated_store['patterns'])}")
print("="*60)