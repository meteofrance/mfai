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
- Keep line length shorter or equal to 88 characters.
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


### Post contribution rules
- After any modification to a Python file, load the `python-code-formating` skill and
  run `uvx ruff@0.15.20 format` then `uvx ruff@0.15.20 check --fix` on the changed files.
- Fix any errors the check raises, then re-run both until no errors are raised.
- Load the `unit-test` skill and run the relevant tests to validate the changed code.

### Python writing rules

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
- Do not use `:data:` / `:func:` / `:class:` or other Sphinx references.

#### 2. Prefer explicit checks over exceptions (LBYL)

Check conditions proactively before acting. Do not use exceptions for normal control flow
(Look Before You Leap, LBYL). Reserve exceptions for error boundaries, third-party APIs
that offer no alternative, or re-raising with added context.

```py
# WRONG: exception as control flow
try:
    value = mapping[key]
    process(value)
except KeyError:
    pass

# CORRECT: check first
if key in mapping:
    value = mapping[key]
    process(value)
```

#### 3. Never swallow exceptions

Let failures propagate instead of catching and ignoring them. Never use bare `except:`
or catch broad exceptions and discard them silently, as hidden failures are hard to
debug and surface far from their root cause.

```py
# WRONG: silent exception swallowing
try:
    risky_operation()
except:
    pass

# CORRECT: let exceptions bubble up
risky_operation()
```

#### 4. Keep magic methods O(1)

`__len__`, `__bool__`, `__contains__`, and properties are called frequently and
implicitly. Implement them in constant time and avoid I/O or expensive computation.

```py
# WRONG: O(n) __len__
def __len__(self) -> int:
    return sum(1 for _ in self._items)

# CORRECT: O(1) __len__
def __len__(self) -> int:
    return self._count
```

#### 5. Check existence before pathlib resolution

When using `pathlib`, check `.exists()` before calling `.resolve()` or `.is_relative_to()`,
which can raise `OSError` on non-existent paths.

```py
# WRONG: .resolve() raises on non-existent paths
wt_path_resolved = wt_path.resolve()
if current_dir.is_relative_to(wt_path_resolved):
    current_worktree = wt_path_resolved

# CORRECT: check exists() first
if wt_path.exists():
    wt_path_resolved = wt_path.resolve()
    if current_dir.is_relative_to(wt_path_resolved):
        current_worktree = wt_path_resolved
```

#### 6. Defer import-time computation

Avoid side effects and expensive work at module import time. Defer resource
initialization (paths, config, connections) until first use with `@cache`.

```py
# WRONG: path computed at import time
SESSION_FILE = Path("scratch/current-session-id")

# CORRECT: compute lazily on first use, then cache
@cache
def _session_file_path() -> Path:
    """Return path to session ID file (cached after first call)."""
    return Path("scratch/current-session-id")
```

#### 7. Verify casts at runtime

`typing.cast()` performs no runtime verification. When casting, add a cheap
`isinstance()` guard so a wrong assumption fails loudly instead of misbehaving
silently.

```py
# WRONG: blind cast
cast(dict[str, Any], doc)["key"] = value

# CORRECT: assert before cast
assert isinstance(doc, MutableMapping), f"Expected MutableMapping, got {type(doc)}"
cast(dict[str, Any], doc)["key"] = value
```

#### 8. Use Literal types for fixed values

When a string belongs to a fixed set of valid values, model it with `Literal` in the
type system so typos are caught at type-check time.

```py
# WRONG: bare strings, typos go unnoticed
issues.append(("orphen-state", "desc"))

# CORRECT: Literal type
IssueCode = Literal["orphan-state", "orphan-dir", "missing-branch"]

@dataclass(frozen=True)
class Issue:
    code: IssueCode
    message: str

issues.append(Issue(code="orphan-state", message="desc"))
```

#### 9. Declare variables close to use

Declare variables as close as possible to where they are used. Do not pollute scope
with early declarations that obscure data flow, unless a value is reused or inlining
hurts readability.

```py
# WRONG: declared far from use
def process_data(ctx, items):
    result_path = compute_result_path(ctx)
    # ... 20 lines of other logic ...
    save_to_path(transformed, result_path)

# CORRECT: inline at call site
def process_data(ctx, items):
    # ... other logic ...
    save_to_path(transformed, compute_result_path(ctx))
```

#### 10. Use keyword-only arguments for complex functions

Functions with five or more parameters must use keyword-only arguments, enforced with
`*` after the first positional parameter, so call sites are self-documenting.

```py
def fetch_data(
    url,
    *,
    timeout: float,
    retries: int,
    headers: dict[str, str],
    auth_token: str,
) -> Response:
    ...
```

#### 11. Avoid default parameter values

Do not use default parameter values unless truly necessary. Defaults cause unexpected
behavior when callers forget to pass a parameter. If a default is never overridden,
eliminate the parameter or hardcode the value.

```py
# WRONG: caller forgets encoding, gets wrong behavior
def process_file(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)

# CORRECT: require explicit choice
def process_file(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding)
```
