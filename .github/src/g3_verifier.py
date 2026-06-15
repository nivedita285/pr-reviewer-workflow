import os
import json
import httpx
import pathlib
import time
from datetime import datetime
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from google import genai
from google.genai import types

GH_TOKEN = os.environ["GH_TOKEN"]
GH_REPO = os.environ["GH_REPO"]
GH_OWNER = os.environ["GH_OWNER"]
GH_PR_NUMBER = int(os.environ["GH_PR_NUMBER"])
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
# In CI, these come from workflow inputs
POST_RESOLVED = os.environ.get("POST_RESOLVED", "true").lower() == "true"
POST_UNRESOLVED = os.environ.get("POST_UNRESOLVED", "false").lower() == "true"
owner, repo = GH_OWNER, GH_REPO
pr_number = GH_PR_NUMBER
token = GH_TOKEN

client = genai.Client(api_key=GEMINI_API_KEY)

def load_team_tone() -> str:
    path = pathlib.Path("../pattern_store/team_tone.json")
    tone = {}
    if not path.exists():
       return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                tone = json.loads(content)
    except json.JSONDecodeError:
        print(f"Warning: team_tone.json contains invalid JSON. Using defaults.")
        tone = {}
    review_tone = tone.get("review_tone", {})
    response_style = tone.get("response_style", {}).get("active", {})
    if not review_tone and not response_style:
       return ""
    lines = ["TEAM RESPONSE TONE (learned from this team's PR history):"]
    lines.append("Use this tone when writing replies to reviewer comments.\n")
    if response_style:
       lines.append(f"Style: {response_style.get('description', '')}")
       lines.append(f"Example: {response_style.get('example', '')}")
       lines.append(f"Avoid: {response_style.get('avoid', '')}\n")
    if review_tone:
       lines.append("Tone by severity:")
       for level in ["critical", "high", "medium", "low"]:
            entry = review_tone.get(level, {})
            active = entry.get("active", "") if isinstance(entry, dict) else entry
            if active:
               lines.append(f"  {level}: {active}")
    return "\n".join(lines)


TEAM_TONE_CONTEXT = load_team_tone()
if TEAM_TONE_CONTEXT:
    print("✅ team_tone.json loaded.")
else:
    print("⚠️ team_tone.json not found or empty. Running without team tone.")

class VerifyState(TypedDict):
    comments: list[dict]
    diff: str
    resolver_document: str
    replies: list[dict]
    approved: bool
    post_decisions: dict  # {"resolved": bool, "unresolved": bool}

VERIFY_PROMPT = """You are a senior software engineer verifying whether pull request review comments have been addressed by a new commit.

You will receive:
1. A resolver document listing all inline review comments, each with a comment_id, the original diff hunk showing the code at review time, and the reviewer's concern.
2. The latest PR diff showing what the code looks like after the newest commit.

Your task is to analyse each comment against the latest diff and determine its resolution status, then produce a replies block.

ANALYSIS RULES:
- For each comment, locate the relevant file and lines in the latest diff.
- Compare the original diff hunk (what was wrong) against the latest diff (what it looks like now).
- Determine one of three statuses:
  - resolved: The issue described in the comment has been clearly fixed in the new code.
  - partial: Some but not all of the concern has been addressed, or the fix is incomplete.
  - unresolved: The code at that location has not changed meaningfully, or the issue persists.
- Do not assume a fix was made unless you can see evidence of it in the latest diff.
- If the file or lines are not present in the latest diff at all, mark as unresolved.

REPLIES BLOCK:
After your analysis, output ONLY the following delimited block. No preamble, no explanation, no other text.

===REPLIES===
[
  {
    "comment_id": 123456789,
    "status": "resolved",
    "reply": "commit-style reply text here"
  }
]
===END_REPLIES===

Rules for each reply:
- Every comment_id from the resolver document must appear exactly once.
- status must be exactly one of: "resolved", "partial", "unresolved".
- reply text rules:
  - No first-person pronouns (no I, We, Us).
  - No introductory phrase summarising the whole file or PR.
  - Describe only what changed (or did not change) for this specific comment.
  - Use past tense, active voice for resolved/partial: "Fixed X by doing Y", "Replaced X with Y".
  - For unresolved: "No change detected at {file} line {line}. Issue remains."
  - For partial: describe what was fixed and what remains.
  - Name the specific file and line.
  - Maximum 30 words.
- Output valid JSON inside the ===REPLIES=== delimiters. No trailing commas.

Your output must contain ONLY the ===REPLIES=== block. Nothing before it, nothing after it."""

def node_a(state: VerifyState) -> dict:
    comments_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    }

    response = httpx.get(comments_url, headers=headers)
    response.raise_for_status()
    inline_comments = response.json()

    comments = []
    for c in inline_comments:
        comments.append(
            {
                "type": "inline",
                "author": c["user"]["login"],
                "comment_id": c["id"],
                "file": c["path"],
                "line": c.get("line") or c.get("original_line"),
                "diff_hunk": c["diff_hunk"],
                "body": c["body"],
                "url": c["html_url"],
            }
        )

    diff_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    diff_headers = {
        "Accept": "application/vnd.github.v3.diff",
        "Authorization": f"Bearer {token}",
    }
    diff_response = httpx.get(diff_url, headers=diff_headers)
    diff_response.raise_for_status()
    raw_diff = diff_response.text

    return {"comments": comments, "diff": raw_diff}

def node_b(state: VerifyState) -> dict:
    document = """PR Review Comments — Verification Document
The following inline comments were left on a pull request. A new commit has since been pushed.
"""

    inline_comments = [comment for comment in state["comments"] if comment.get("type") == "inline"]
    for index, comment in enumerate(inline_comments, start=1):
        document += f"""

Issue {index} (Inline — {comment['file']} line {comment['line']})
Author: {comment['author']}
comment_id: {comment['comment_id']}
Reference: {comment['url']}
Code context (diff hunk):
{comment['diff_hunk']}
Comment / suggestion:
{comment['body']}
You will compare these comments against the latest PR diff to determine which issues have been resolved.
"""

    os.makedirs("resolver_documents", exist_ok=True)
    repo_name = repo.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resolver_documents/{repo_name}_pr{pr_number}_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(document)
    print(f"✅ Resolver document saved to {filename}")

    return {"resolver_document": document}

def node_c(state: VerifyState) -> dict:
    effective_prompt = VERIFY_PROMPT
    if TEAM_TONE_CONTEXT:
        effective_prompt = VERIFY_PROMPT + "\n\n" + TEAM_TONE_CONTEXT

    user_message = f"""RESOLVER DOCUMENT:
{state['resolver_document']}

LATEST PR DIFF:
{state['diff']}"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=effective_prompt,
        )
    )

    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    parsed_replies = []
    if "===REPLIES===" in text and "===END_REPLIES===" in text:
        replies_text = text.split("===REPLIES===", 1)[1]
        replies_text = replies_text.split("===END_REPLIES===", 1)[0].strip()
        try:
            parsed_replies = json.loads(replies_text)
        except json.JSONDecodeError:
            parsed_replies = []
            print("⚠️ Could not parse REPLIES block. Replies will be empty.")
    else:
        print("⚠️ Could not find REPLIES block. Replies will be empty.")

    return {"replies": parsed_replies}

def node_d(state: VerifyState) -> dict:
    repo_name = repo.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resolver_responses/{repo_name}_pr{pr_number}_{timestamp}_verify.json"
    os.makedirs("resolver_responses", exist_ok=True)

    resolver_doc = [
        {
            "comment_id": r["comment_id"],
            "status": r.get("status", "unresolved"),
            "reply": r["reply"]
        }
        for r in state["replies"]
    ]
    with open(filename, "w", encoding="utf-8") as f:
        f.write(json.dumps(resolver_doc, indent=2))
    print(f"✅ Verification responses saved to {filename}")

    resolved = [r for r in resolver_doc if r["status"] == "resolved"]
    partial = [r for r in resolver_doc if r["status"] == "partial"]
    unresolved = [r for r in resolver_doc if r["status"] == "unresolved"]
    not_resolved = partial + unresolved

    def build_group_preview(items, emoji, label):
        if not items:
            return f"\n{emoji} {label}: none\n"
        lines = [f"\n{emoji} {label} ({len(items)}):"]
        for item in items:
            lines.append(f"  comment_id: {item['comment_id']}")
            lines.append(f"  reply: {item['reply']}")
            lines.append("  " + "-" * 38)
        return "\n".join(lines)

    preview = "Verification Report — " + str(len(resolver_doc)) + " comments analysed\n"
    preview += f"✅ Resolved: {len(resolved)}  ⚠️ Partial: {len(partial)}  ❌ Unresolved: {len(unresolved)}\n"
    preview += "=" * 60
    preview += build_group_preview(resolved, "✅", "RESOLVED")
    preview += build_group_preview(not_resolved, "⚠️❌", "PARTIAL / UNRESOLVED")

    return {
           "replies": state["replies"],
           "approved": POST_RESOLVED or POST_UNRESOLVED,
           "post_decisions": {
                "resolved": POST_RESOLVED,
                "unresolved": POST_UNRESOLVED
            }
       }

def node_e(state: VerifyState) -> dict:
    decisions = state.get("post_decisions", {"resolved": True, "unresolved": True})
    post_resolved = decisions.get("resolved", True)
    post_unresolved = decisions.get("unresolved", True)

    for item in state["replies"]:
        comment_id = int(item["comment_id"])
        reply = str(item["reply"])
        status = item.get("status", "unresolved")

        is_resolved = status == "resolved"
        should_post = (is_resolved and post_resolved) or (not is_resolved and post_unresolved)

        if not should_post:
            print(f"⏭️  Skipped comment {comment_id} (status: {status}, posting disabled for this group)")
            continue

        response = httpx.post(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments/{comment_id}/replies",
            headers={
                "Authorization": f"Bearer {GH_TOKEN}",
                "Accept": "application/vnd.github+json"
            },
            json={"body": reply}
        )
        if response.status_code not in (200, 201):
            print(f"⚠️ Failed to post reply to comment {comment_id}: {response.status_code} — {response.text}")
        else:
            print(f"✅ Reply posted to comment {comment_id} (status: {status})")

    return {}
# Paste load_team_tone(), VerifyState, VERIFY_PROMPT,
# node_a, node_b, node_c exactly from L10.
#
# For node_d: replace the interrupt block entirely with:
#
#   return {
#       "replies": state["replies"],
#       "approved": POST_RESOLVED or POST_UNRESOLVED,
#       "post_decisions": {
#           "resolved": POST_RESOLVED,
#           "unresolved": POST_UNRESOLVED
#       }
#   }
#   And still save the resolver_responses file — keep that part.
#
# node_e: paste exactly from L10, no changes needed.

# Paste graph assembly exactly from L10.

graph_builder = StateGraph(VerifyState)

graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b)
graph_builder.add_node("node_c", node_c)
graph_builder.add_node("node_d", node_d)
graph_builder.add_node("node_e", node_e)

graph_builder.add_edge(START, "node_a")
graph_builder.add_edge("node_a", "node_b")
graph_builder.add_edge("node_b", "node_c")
graph_builder.add_edge("node_c", "node_d")


def route_after_d(state: VerifyState) -> str:
    return "node_e" if state.get("approved", False) else END


graph_builder.add_conditional_edges("node_d", route_after_d)
graph_builder.add_edge("node_e", END)

app = graph_builder.compile(checkpointer=MemorySaver(), interrupt_before=[])

config = {"configurable": {"thread_id": f"pr-verify-{pr_number}-{int(time.time())}"}}
initial_state = VerifyState(
    comments=[], diff="", resolver_document="",
    replies=[], approved=False, post_decisions={}
)
result = app.invoke(initial_state, config)
print("Verification complete.")