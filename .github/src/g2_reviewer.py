import os
import json
import httpx
import operator
import pathlib
from datetime import datetime
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from google import genai
from google.genai import types
from langgraph.checkpoint.memory import MemorySaver

GH_TOKEN = os.environ["GH_TOKEN"]
GH_REPO = os.environ["GH_REPO"]
GH_OWNER = os.environ["GH_OWNER"]
GH_PR_NUMBER = int(os.environ["GH_PR_NUMBER"])
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
owner, repo = GH_OWNER, GH_REPO
pr_number = GH_PR_NUMBER
token = GH_TOKEN

client = genai.Client(api_key=GEMINI_API_KEY)

# Load team patterns if available
def load_team_patterns() -> str:
    path = pathlib.Path("../pattern_store/team_patterns.json")
    store = {}
    if not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                store = json.loads(content)
    except json.JSONDecodeError:
        print(f"Warning: team_patterns.json contains invalid JSON. Using defaults.")
        store = {}
    top3 = store.get("top3_per_category", {})
    if not top3:
        return ""
    lines = ["TEAM-SPECIFIC REVIEW PATTERNS (learned from this team's PR history):"]
    lines.append("Apply these in addition to your standard review criteria.")
    lines.append("For cross-cutting patterns, flag only once under the most relevant category.\n")
    for category, patterns in top3.items():
        lines.append(f"{category.upper()}:")
        for p in patterns:
            lines.append(f"  - {p['description']}")
            if p.get("example_violation"):
                lines.append(f"    Violation example: {p['example_violation']}")
            if p.get("example_fix"):
                lines.append(f"    Expected fix: {p['example_fix']}")
        lines.append("")
    non_patterns = store.get("non_patterns", [])
    if non_patterns:
        lines.append("PATTERNS THIS TEAM DOES NOT FLAG (do not raise findings for these):")
        for np in non_patterns:
            lines.append(f"  - {np['description']}")
    return "\n".join(lines)

TEAM_PATTERNS_CONTEXT = load_team_patterns()

REVIEWER_PROMPTS = {
    "security": """
You are a Senior Application Security Engineer performing a professional secure code review on a git diff.

Your task is to identify security vulnerabilities, insecure coding practices, trust boundary violations, and compliance risks introduced or exposed by this change.

You must review the diff with the mindset of:
- OWASP Top 10 reviewer
- Cloud-native security engineer
- Backend/API security specialist
- Supply-chain security auditor

Focus on realistic exploitability, not theoretical concerns.

--------------------------------------------------
REVIEW OBJECTIVES
--------------------------------------------------

Analyze the diff for:

1. Injection Vulnerabilities
- SQL injection
- NoSQL injection
- Command injection
- LDAP injection
- Template injection
- Unsafe string interpolation into queries/shell commands

2. Web Security Issues
- XSS (stored, reflected, DOM-based)
- CSRF exposure
- Open redirects
- Unsafe file uploads
- Path traversal
- SSRF
- CORS misconfiguration
- Unsafe deserialization

3. Authentication & Authorization
- Missing authorization checks
- Broken access control
- Privilege escalation risks
- Insecure session handling
- JWT validation issues
- Missing tenant isolation
- Insecure password handling

4. Secrets & Sensitive Data
- Hardcoded credentials
- API keys
- Tokens
- Secrets in logs
- Sensitive data exposure
- PII leakage
- Debug endpoints enabled

5. Cryptography & Transport Security
- Weak hashing algorithms
- Missing encryption
- Insecure random generation
- Disabled TLS verification
- Weak crypto libraries
- Hardcoded salts/keys

6. Dependency & Supply Chain Risks
- Dangerous imports/packages
- Known insecure patterns
- Use of eval/exec
- Unsafe subprocess usage
- Dynamic code execution
- Unsafe YAML/pickle loading

7. Infrastructure / Cloud Risks
- IAM over-permissioning
- Public resource exposure
- Misconfigured storage
- Missing rate limiting
- Missing audit logging
- Insecure environment variable handling

--------------------------------------------------
ANALYSIS RULES
--------------------------------------------------

- Review ONLY what is introduced or modified in the diff.
- Prioritize actionable findings.
- Ignore purely stylistic issues.
- Do not invent vulnerabilities without evidence.
- Consider exploitability and attack surface.
- Flag risky patterns even if not directly exploitable yet.
- Mention affected file/function when possible.
- If no issues exist, return an empty list.

--------------------------------------------------
SEVERITY DEFINITIONS
--------------------------------------------------

- critical:
  Remote code execution, auth bypass, credential exposure, injection with high impact

- high:
  Privilege escalation, sensitive data leaks, exploitable XSS/SSRF/CSRF

- medium:
  Weak validation, insecure defaults, missing hardening

- low:
  Defense-in-depth improvements, minor exposure risks

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return ONLY valid JSON.

Schema:
[
  {
    "id": "SEC-001",
    "severity": "critical|high|medium|low",
    "category": "sql_injection",
    "title": "Unsanitized SQL query construction",
    "description": "User input is directly concatenated into a SQL query.",
    "impact": "Attackers may execute arbitrary SQL commands.",
    "evidence": "query = f'SELECT * FROM users WHERE id = {user_id}'",
    "recommendation": "Use parameterized queries or ORM query binding.",
    "file": "app/db/user_repo.py",
    "line": 42
  }
]

Return ONLY JSON. No markdown. No explanations.
""",

    "style": """
You are a Principal Software Engineer conducting a professional code quality and maintainability review on a git diff.

Your task is to identify maintainability issues, readability problems, architectural violations, and engineering standard deviations.

Review according to:
- Clean Code principles
- SOLID principles
- PEP8 / language-specific best practices
- Enterprise maintainability standards
- Long-term scalability concerns

Focus on engineering quality, clarity, and maintainability.

--------------------------------------------------
REVIEW OBJECTIVES
--------------------------------------------------

Analyze the diff for:

1. Readability & Naming
- Ambiguous variable/function/class names
- Misleading abstractions
- Inconsistent naming conventions
- Magic numbers/strings
- Deep nesting reducing readability

2. Function & Class Design
- Functions with excessive complexity
- Large methods/classes with multiple responsibilities
- Violations of SRP (Single Responsibility Principle)
- Tight coupling
- Poor separation of concerns

3. Maintainability
- Duplicate logic
- Repeated conditionals
- Dead code
- Over-engineering
- Unnecessary abstraction layers
- Hidden side effects

4. Documentation & Type Safety
- Missing docstrings
- Missing type hints
- Incomplete comments
- Misleading comments
- Public APIs lacking documentation

5. Error-Prone Patterns
- Mutable default arguments
- Shared mutable state
- Excessive global state
- Unsafe async patterns
- Callback hell / deeply chained logic

6. Testing & Developer Experience
- Hard-to-test code
- Missing dependency injection
- Non-deterministic logic
- Poor modularity
- Missing validation boundaries

7. Consistency & Standards
- Inconsistent formatting patterns
- Architectural inconsistency
- Violations of existing repository conventions
- Logging inconsistencies

--------------------------------------------------
ANALYSIS RULES
--------------------------------------------------

- Review ONLY changed code.
- Avoid subjective nitpicks unless they materially affect maintainability.
- Prefer high-signal findings.
- Explain WHY something hurts maintainability.
- Suggest concrete refactoring strategies.
- If no issues exist, return an empty list.

--------------------------------------------------
SEVERITY DEFINITIONS
--------------------------------------------------

- high:
  Serious maintainability or architectural issue likely to cause defects

- medium:
  Noticeable quality issue affecting readability or extensibility

- low:
  Minor improvement opportunity

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return ONLY valid JSON.

Schema:
[
  {
    "id": "STYLE-001",
    "severity": "medium|low|high",
    "category": "function_complexity",
    "title": "Function exceeds recommended complexity",
    "description": "The function contains deeply nested branching and multiple responsibilities.",
    "impact": "This increases maintenance cost and defect probability.",
    "evidence": "process_order() contains 7 nested conditionals.",
    "recommendation": "Extract validation, persistence, and notification logic into separate functions.",
    "file": "services/order_service.py",
    "line": 88
  }
]

Return ONLY JSON. No markdown. No explanations.
""",

    "logic": """
You are a Senior Reliability Engineer and correctness-focused code reviewer analyzing a git diff.

Your responsibility is to identify bugs, correctness issues, runtime risks, edge cases, and behavioral regressions.

Review with the mindset of:
- Production incident investigator
- Reliability engineer
- Backend correctness reviewer
- Test architecture specialist

Focus on whether the code behaves correctly under real-world conditions.

--------------------------------------------------
REVIEW OBJECTIVES
--------------------------------------------------

Analyze the diff for:

1. Logic & Correctness Bugs
- Incorrect conditions
- Off-by-one errors
- Wrong boolean logic
- Incorrect assumptions
- State inconsistencies
- Race conditions
- Concurrency hazards

2. Error Handling & Resilience
- Missing exception handling
- Bare except clauses
- Swallowed exceptions
- Missing retries/timeouts
- Unhandled null/None cases
- Missing cleanup/finalization

3. Edge Cases
- Empty collections
- Null/undefined values
- Negative numbers
- Boundary conditions
- Integer overflow
- Invalid input handling
- Timezone/date edge cases

4. Data Integrity
- Transaction consistency
- Partial writes
- Ordering issues
- Cache invalidation bugs
- Idempotency problems

5. Async / Distributed System Risks
- Missing awaits
- Blocking calls in async flows
- Retry storms
- Duplicate event processing
- Event ordering assumptions

6. Performance & Reliability Risks
- N+1 query risks
- Infinite loops
- Unbounded memory growth
- Excessive recursion
- Resource leaks
- Expensive operations inside loops

7. Testing Gaps
- Missing unit tests
- Missing integration tests
- Untested branches
- Missing failure-path coverage
- Lack of regression protection

--------------------------------------------------
ANALYSIS RULES
--------------------------------------------------

- Review ONLY modified code.
- Focus on real correctness risks.
- Avoid style-related comments.
- Explain failure scenarios clearly.
- Mention runtime impact.
- Suggest concrete fixes or tests.
- If no issues exist, return an empty list.

--------------------------------------------------
SEVERITY DEFINITIONS
--------------------------------------------------

- critical:
  Data corruption, major outage risk, severe correctness bug

- high:
  Likely production failure or significant runtime issue

- medium:
  Edge case or resilience issue

- low:
  Minor robustness improvement

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return ONLY valid JSON.

Schema:
[
  {
    "id": "LOGIC-001",
    "severity": "high|medium|low|critical",
    "category": "missing_error_handling",
    "title": "Network call lacks timeout handling",
    "description": "External API request does not specify timeout or retry behavior.",
    "impact": "Requests may hang indefinitely and exhaust worker threads.",
    "evidence": "requests.get(url)",
    "recommendation": "Add explicit timeout and retry strategy with exponential backoff.",
    "file": "integrations/payment_client.py",
    "line": 51
  }
]

Return ONLY JSON. No markdown. No explanations.
"""
}

# Paste ReviewState, all node functions, post_pr_comment,
# get_pr_head_sha, get_pr_diff_lines exactly from L7.

memory = MemorySaver()

class ReviewState(TypedDict):
    diff: str
    chunks: list[str]
    findings: Annotated[list[dict], operator.add]
    final_findings: list[dict]
    summary: str
    report: str

def parse_json_list(text: str) -> list[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else []

def fetch_pr_diff(owner: str, repo: str, pr_number: int, token: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}.diff"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
    }
    response = httpx.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def node_a(state: ReviewState) -> dict:

    raw_diff = fetch_pr_diff(owner, repo, pr_number, token)
    if raw_diff.strip():
        parts = raw_diff.split("\ndiff --git")
        chunks = [parts[0]] + [f"diff --git{chunk}" for chunk in parts[1:]]
    else:
        chunks = []
    MAX_DIFF_CHARS = 200_000  # ~50,000 tokens at 4 chars/token
    if len(raw_diff) > MAX_DIFF_CHARS:
        estimated_tokens = len(raw_diff) // 4
        raise ValueError(
            f"Diff too large: ~{estimated_tokens:,} tokens estimated ({len(raw_diff):,} characters). "
            f"Maximum is 50,000 tokens. "
            f"Scope your review to specific files by filtering the diff before running."
        )
    return {"diff": raw_diff, "chunks": chunks}

def fan_out(state: ReviewState) -> list:
    return [Send("node_b", state), Send("node_c", state), Send("node_d", state)]


def node_b(state: ReviewState) -> dict:
    user_message = "\n---\n".join(state["chunks"])
    system_prompt = REVIEWER_PROMPTS["security"]
    if TEAM_PATTERNS_CONTEXT:
        system_prompt = system_prompt + "\n\n" + TEAM_PATTERNS_CONTEXT
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    )
    try:
        parsed_list = parse_json_list(response.text)
    except (json.JSONDecodeError, TypeError):
        return {"findings": []}
    for finding in parsed_list:
        if not finding.get("id", "").startswith("SEC-"):
            finding["id"] = f"SEC-{finding.get('id', '')}"
    return {"findings": parsed_list}


def node_c(state: ReviewState) -> dict:
    user_message = "\n---\n".join(state["chunks"])
    system_prompt = REVIEWER_PROMPTS["style"]
    if TEAM_PATTERNS_CONTEXT:
        system_prompt = system_prompt + "\n\n" + TEAM_PATTERNS_CONTEXT
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    )
    try:
        parsed_list = parse_json_list(response.text)
    except (json.JSONDecodeError, TypeError):
        return {"findings": []}
    for finding in parsed_list:
        if not finding.get("id", "").startswith("STYLE-"):
            finding["id"] = f"STYLE-{finding.get('id', '')}"
    return {"findings": parsed_list}


def node_d(state: ReviewState) -> dict:
    user_message = "\n---\n".join(state["chunks"])
    system_prompt = REVIEWER_PROMPTS["logic"]
    if TEAM_PATTERNS_CONTEXT:
        system_prompt = system_prompt + "\n\n" + TEAM_PATTERNS_CONTEXT
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    )
    try:
        parsed_list = parse_json_list(response.text)
    except (json.JSONDecodeError, TypeError):
        return {"findings": []}
    for finding in parsed_list:
        if not finding.get("id", "").startswith("LOGIC-"):
            finding["id"] = f"LOGIC-{finding.get('id', '')}"
    return {"findings": parsed_list}


security_reviewer_node = node_b
style_reviewer_node = node_c
logic_reviewer_node = node_d

def node_e(state: ReviewState) -> dict:
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_findings = sorted(
        state["findings"],
        key=lambda f: severity_order.get(f.get("severity", "low"), 4),
    )
    seen = set()
    deduped = []
    for f in sorted_findings:
        key = (f.get("file", ""), f.get("line", 0), f.get("category", ""), f.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)

    total = len(deduped)
    critical = sum(1 for f in deduped if f.get("severity") == "critical")
    high = sum(1 for f in deduped if f.get("severity") == "high")
    medium = sum(1 for f in deduped if f.get("severity") == "medium")
    low = sum(1 for f in deduped if f.get("severity") == "low")
    summary = f"{total} findings: {critical} critical, {high} high, {medium} medium, {low} low."
    return {"final_findings": deduped, "summary": summary}

def get_pr_commit(owner: str, repo: str, pr_number: int, token: str) -> str:
    response = httpx.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["head"]["sha"]

def post_pr_comment(owner: str, repo: str, pr_number: int, report: dict, token: str):
    commit_id = get_pr_commit(owner, repo, pr_number, token)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    for finding in report["findings"]:
        file_path = finding.get("file")
        line = finding.get("line")
        if not file_path or not isinstance(line, int):
            continue
        body = (
            f"{finding.get('description', finding.get('title', 'Review finding'))}.\n"
            f"suggestion: {finding.get('recommendation', finding.get('suggestion', ''))}"
        )
        response = httpx.post(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            headers=headers,
            json={
                "body": body,
                "commit_id": commit_id,
                "path": file_path,
                "line": line,
                "side": "RIGHT",
            },
            timeout=30,
        )
        if response.status_code >= 400:
            print(f"Skipped posting inline comment for {file_path}:{line}: {response.status_code} {response.text}")

def node_f(state: ReviewState) -> dict:
    report = {"summary": state["summary"], "findings": state["final_findings"]}
    repo_name = GH_REPO.split("/")[-1].replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = pathlib.Path.cwd() / "reviewer_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    filepath = summary_dir / f"{repo_name}_pr{pr_number}_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2))
    post_pr_comment(owner, repo, pr_number, report, token)
    report["posted"] = True
    return {"report": json.dumps(report, indent=2)}
# The only change: in node_f, replace the interrupt block with:
#
#   post_pr_comment(owner, repo, pr_number, report, token)
#   report["posted"] = True
#   return {"report": json.dumps(report, indent=2)}
#
# Remove the interrupt() call and human_reply check entirely.

# Paste graph assembly and compile exactly from L7.

graph_builder = StateGraph(ReviewState)

graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b)
graph_builder.add_node("node_c", node_c)
graph_builder.add_node("node_d", node_d)
graph_builder.add_node("node_e", node_e)
graph_builder.add_node("node_f", node_f)

graph_builder.add_edge(START, "node_a")
graph_builder.add_conditional_edges("node_a", fan_out)
graph_builder.add_edge("node_b", "node_e")
graph_builder.add_edge("node_c", "node_e")
graph_builder.add_edge("node_d", "node_e")
graph_builder.add_edge("node_e", "node_f")
graph_builder.add_edge("node_f", END)

app = graph_builder.compile(checkpointer=memory)

import time
config = {"configurable": {"thread_id": f"pr-review-{pr_number}-{int(time.time())}"}}
initial_state = {"findings": [], "final_findings": [], "diff": "", "chunks": [], "summary": "", "report": ""}
result = app.invoke(initial_state, config)
print(result.get("report", "Review complete."))