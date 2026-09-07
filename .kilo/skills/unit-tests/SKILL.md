---
name: unit-test
description: Project unit testing. Use after any Python modification to check validity.
---

# Unit testing

Run the project's unit tests with pytest to validate code changes.

## When to use

Run the tests after any Python modification. The mfai library is unit tested with a
coverage above 85 %, so every code change should be validated by the existing tests
before being considered done.

## Running the tests

Run the full test suite:

```bash
uv run pytest
```

The tests live under the repo's `tests/` directory. When you modify a specific feature,
you can run only the tests for that feature to get faster feedback:

```bash
uv run pytest tests/<file_name_where_appropriate_tests_are_present>.py
```

Replace `<file_name_where_appropriate_tests_are_present>` with the test file covering
the feature you changed. Run the relevant subset first while iterating, then run the
full suite (or at least the touched module's tests) before finishing to make sure
nothing else broke.
