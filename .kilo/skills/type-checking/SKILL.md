---
name: type-checking
description: Project static type checking. Use after any Python modification to check validity.
---

# Type Checking

Run the project's type checking with pyright to validate code changes.

## When to use

Run after any Python modification. Type hinting configuration is specified in the project's `pyprojec.toml` file.

## Running the tests

Run the full test suite:

```bash
uv run pyright
```

To run on specific file:

```bash
uv run pyright <file_name_where_appropriate_tests_are_present>.py
```