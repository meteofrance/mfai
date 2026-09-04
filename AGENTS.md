# AGENTS.md

## Chat answer guidelines

These instructions apply to LLM agents answering to the user's chat prompts.

- Be respectful.
- Be concise, do not give examples unless prompted to. Short answers.
- Do not over explain, assume user expertise and give precise explanations when prompted.
- If a code example is requested, answer only with the code, no explanations.
- When making asumptions about the project's environment, check the `pyproject.toml` file.

## Python coding guidelines

These instructions apply to LLM agents writing or editing Python code in this repository.

### Keep it simple

- Prefer the simplest solution that works. Do not over-engineer.
- Do not add abstractions, base classes, or indirection unless the code actually needs them.
- Solve the problem at hand; avoid speculative generality and unused features.

### Follow best practices

- Follow PEP 8 style and the conventions already used in the surrounding code.
- Use type hints on all function signatures and public variables.
- Prefer the standard library and Python 3 idioms (e.g. f-strings, `pathlib`, dataclasses) over third-party utilities where the stdlib suffices.
- Write clear, descriptive names for variables, functions, and classes.
- Keep functions short and focused on a single responsibility.
- Do not add emojis.
- Do not write code containing security vulnerability
- If you propose the usage of third party package, check their cybersecurity status (known vulnerabilities, use appropriate versions, etc)
- Warn the user if you think there is a cybersecurity risk with either the code you propose or the problem you are asked to resolve 
- Keep line length shorter or equal to 79 characters.
- Write code following the structure double line break, comment, code pragraph. Like so:
```py

# Comment explaining what
my = python_code()
can = be_multiple_lines()  # Comment explaining why

# Second paragraph separated by a double line break
_with = the_next_python_code_paragraph()
```

### Respect dev context
You are writting code in the mfai library, wich should be compatible for all versions of python >= 3.10. Keep it easy to maintain. This libairy is unit tested with a coverage > 85 %. When introducing new features, ensure it is tested.

### Python writting rules

#### 1. Docstrings (Google style)

- Add a docstring to every public module, class, and function.
- Format docstrings following Google's syntax:

```
"""Summary, that can
be multiline.

Args:
    arg_name: Description of the argument.

Returns:
    return_type: Description of the return value.
"""
```

- Start with a summary that can be multiline ending in a period.
- Add a blank line after the summary when there is more content.
- Use an `Args:` section with one indented `name: description.` line per parameter. Describe what each parameter is or does; do not repeat its type in the description (types come from annotations).
- Add a `Returns:` section describing the return value. Use `Yields:` for generators.
- Omit `Args`/`Returns` sections only when there are no parameters or return value.
- Do not use `:data:` / `:func:` / `:class:` or oother Sphinx references.

### 2. Look Before You Leap

The single most important rule in Dignified Python is to check conditions proactively rather than relying on exceptions for control flow. We call this Look Before You Leap (LBYL), in contrast to the Easier to Ask for Forgiveness than Permission (EAFP) pattern that many LLMs default to.

// WRONG: Exception as control flow
try:
    value = mapping[key]
    process(value)
except KeyError:
    pass

// CORRECT: Check first
if key in mapping:
    value = mapping[key]
    process(value)

LBYL makes intent explicit. The reader can see immediately what conditions are being checked and what happens in each case. EAFP obscures this by burying the logic inside exception handlers.

Exceptions are still acceptable at error boundaries, when interacting with third-party APIs that provide no alternative, or when adding context before re-raising. A good example is interacting with external services where we lack control over the underlying behavior:

// ACCEPTABLE: Third-party API forces exception handling
def _get_bigquery_sample(sql_client, table_name):
    """
    BigQuery's TABLESAMPLE doesn't work on views.
    There's no reliable way to determine a priori whether
    a table supports TABLESAMPLE.
    """
    try:
        return sql_client.run_query(f"SELECT * FROM {table_name} TABLESAMPLE...")
    except Exception:
        return sql_client.run_query(f"SELECT * FROM {table_name} ORDER BY RAND()...")

### 3. Never Swallow Exceptions

A common issue that arises from LLM pattern matching is silent error swallowing. Many models overuse broad try and except blocks, and one of the most problematic variants is catching every exception and ignoring it entirely.

// WRONG: Silent exception swallowing
try:
    risky_operation()
except:
    pass

// CORRECT: Let exceptions bubble up
risky_operation()

‍
Although the first version will run, it hides failures that may be critical to the correctness of your system. Debugging issues introduced by swallowed exceptions can be extremely difficult since the original error is lost and the failure often surfaces far away from the root cause. The Python community has long discussed restricting or discouraging bare except clauses, as seen in proposals like PEP 760.

Dignified Python encourages code that is explicit. If an operation can fail in a meaningful way, that failure should be visible and actionable. That means allowing exceptions to propagate naturally unless there is a compelling and clearly defined reason to handle them.
### 4. Magic Methods Must Be O(1)

Performance is an area where agents often fall short. LLMs tend to focus on producing code that works, not code that is efficient. Without explicit guidance, they may introduce subtle performance issues that only become visible once the code is used at scale.

Magic methods like__len__, __bool__, and __contains__ are called frequently and implicitly. They must run in constant time.

// WRONG: __len__ doing iteration
def __len__(self) -> int:
    return sum(1 for _ in self._items)

// CORRECT: O(1) __len__
def __len__(self) -> int:
    return self._count

‍
The first implementation is correct but inefficient. Each call requires iterating over the entire collection, which can introduce significant overhead when used in loops, conditionals, or membership checks. The same rule applies to properties, which should never perform I/O or expensive computation.
### 5. Check Existence Before Resolution

When working with pathlib, LLMs often forget that certain methods can fail on non-existent paths. The rule is simple: always check .exists() before calling .resolve() or .is_relative_to().

from pathlib import Path

// WRONG: resolve() can raise OSError on non-existent paths
wt_path_resolved = wt_path.resolve()
if current_dir.is_relative_to(wt_path_resolved):
    current_worktree = wt_path_resolved

// CORRECT: Check exists() first
if wt_path.exists():
    wt_path_resolved = wt_path.resolve()
    if current_dir.is_relative_to(wt_path_resolved):
        current_worktree = wt_path_resolved

‍This follows directly from the LBYL principle. Instead of catching exceptions after the fact, we verify preconditions before calling methods that might fail.
### 6. Defer Import-Time Computation

Module-level code runs when the module is imported. Side effects at import time cause slower startup, test brittleness, circular import issues, and unpredictable behavior based on import order.

from pathlib import Path
from functools import cache

// WRONG: Path computed at import time
SESSION_FILE = Path("scratch/current-session-id")

// CORRECT: Defer with @cache
@cache
def _session_file_path() -> Path:
    """Return path to session ID file (cached after first call)."""
    return Path("scratch/current-session-id")

‍
The @cache decorator ensures the computation happens only once, but not until the function is first called. This pattern works for any resource initialization: configuration loading, database connections, or path construction.
### 7. Verify Your Casts at Runtime

typing.cast() is a compile-time only construct. It tells the type checker to trust you but performs no runtime verification. If your assumption is wrong, you will get silent misbehavior instead of a clear error.

from typing import Any, cast
from collections.abc import MutableMapping

// WRONG: Blind cast
cast(dict[str, Any], doc)["key"] = value

// CORRECT: Assert before cast
assert isinstance(doc, MutableMapping), f"Expected MutableMapping, got {type(doc)}"
cast(dict[str, Any], doc)["key"] = value

‍
When the cost of the assertion is trivial (O(1) checks like isinstance), always add it. Skip the assertion only when you have just performed a type guard, or in measured performance-critical hot paths with documented justification.
### 8. Use Literal Types for Fixed Values

When strings represent a fixed set of valid values, such as error codes, status values, or command types, model them in the type system using Literal. This catches typos at type-check time, enables IDE autocomplete, and documents valid values directly in the code.

from typing import Literal
from dataclasses import dataclass

// WRONG: Bare strings
issues.append(("orphen-state", "desc"))  # Typo goes unnoticed!

// CORRECT: Literal type
IssueCode = Literal["orphan-state", "orphan-dir", "missing-branch"]

@dataclass(frozen=True)
class Issue:
    code: IssueCode
    message: str

issues.append(Issue(code="orphan-state", message="desc"))  # Type-checked!

‍
Before using a bare str type, ask: Is this string compared with == or in anywhere? Is there a fixed set of valid values? Would a typo cause a bug? If any answer is yes, use Literal instead.
### 9. Declare Variables Close to Use

Variables should be declared as close as possible to where they are used. Avoid early declarations that pollute scope and obscure data flow.

// WRONG: Variable declared 20 lines before use
def process_data(ctx, items):
    result_path = compute_result_path(ctx)
    # ... 20 lines of other logic ...
    save_to_path(transformed, result_path)

// CORRECT: Inline at call site
def process_data(ctx, items):
    # ... other logic ...
    save_to_path(transformed, compute_result_path(ctx))

‍
This reduces cognitive load because readers do not need to scroll back to understand where a value came from. It also makes data flow visible at a glance. The exception is when a value is used multiple times or when inlining would hurt readability.
### 10. Keyword Arguments for Complex Functions

Functions with five or more parameters must use keyword-only arguments. Use the * separator after the first positional parameter to enforce this at the language level.

// WRONG: Positional chaos - what do these values mean?
response = fetch_data(api_url, 30.0, 3, {"Accept": "application/json"}, token)

// CORRECT: Keyword-only after first param
def fetch_data(
    url,
    *,
    timeout: float,
    retries: int,
    headers: dict[str, str],
    auth_token: str,
) -> Response:
    ...

# Call site is self-documenting
response = fetch_data(
    api_url,
    timeout=30.0,
    retries=3,
    headers={"Accept": "application/json"},
    auth_token=token,
)

‍
This improves call-site readability by forcing explicit parameter names. The first parameter (often self, ctx, or the primary subject of the function) can remain positional.

### 11. Default Values Are Dangerous

Avoid default parameter values unless absolutely necessary. They are a significant source of bugs because callers forget to pass a parameter and get unexpected results.

# DANGEROUS: Caller forgets encoding, gets wrong behavior
def process_file(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)

content = process_file(legacy_latin1_file)  # Bug: should be encoding="latin-1"

# SAFER: Require explicit choice
def process_file(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding)

content = process_file(legacy_latin1_file, encoding="latin-1")

‍
When a default value is never overridden anywhere in your codebase, eliminate the parameter entirely or hardcode the value. Acceptable uses of defaults include truly optional behavior where the default is correct for 95%+ of callers, or temporary backwards compatibility when adding parameters to existing APIs.