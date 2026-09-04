# AGENTS.md

## Chat answer guidelines

These instructions pply to LLM agents answering to the user's chat prompts.

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

### Docstrings (Google style)

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
