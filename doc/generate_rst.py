"""
This script automatically generates RST files for the mfai Sphinx documentation.
Instead of relying solely on sphinx-apidoc (which generates a flat, unstructured output),
this script introspects the mfai codebase at build time to find classes matching specific
inheritance criteria, and generates structured autosummary RST files grouped by category
(Models, Losses, Lightning, Transforms, Metrics).

This script is meant to be run before the Sphinx build step, either locally or in CI.
"""

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Literal, Sequence, Generator

sys.path.insert(0, str(Path(__file__).parent.parent))


def is_subclass_of_names(cls: type, base_names: Sequence[str | list[str]]) -> bool:
    """Check if a class inherits from any of the given base class names.

    Supports both OR and AND conditions:
    - A string entry means OR: the class must inherit from at least one of them.
    - A list entry means AND: the class must inherit from all of them.

    Args:
        cls: The class to check.
        base_names: A list of conditions. Each condition is either a string (OR)
            or a list of strings (AND). Example: ["BaseModel", ["ModelABC", "Module"]]
            means: inherits from BaseModel OR (ModelABC AND Module).

    Returns:
        True if the class matches any condition, False otherwise.
    """
    if not inspect.isclass(cls):
        return False
    mro_names = [base.__name__ for base in cls.__mro__[1:]]
    for condition in base_names:
        if isinstance(condition, list):
            if all(name in mro_names for name in condition):
                return True
        else:
            if condition in mro_names:
                return True
    return False



def iter_all_modules(package_path: str) -> Generator[str, None, None]:
    """Recursively yield all module names within a given package.

    Args:
        package_path: The dotted path of the package to introspect
            (e.g. 'mfai.pytorch.models').

    Yields:
        Fully qualified module name strings.
    """
    try:
        pkg = importlib.import_module(package_path)
    except ImportError:
        return

    yield package_path

    if hasattr(pkg, "__path__"):
        for mod_info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package_path}."):
            yield mod_info.name


def get_classes_matching(package_path: str, base_names: Sequence[str | list[str]]) -> list[str]:
    """Find all classes in a package that match the given inheritance conditions.

    Only classes that are defined directly in their module (not re-exported
    from another module) are included.

    Args:
        package_path: The dotted path of the package to search
            (e.g. 'mfai.pytorch.losses').
        base_names: Inheritance conditions passed to `is_subclass_of_names`.

    Returns:
        A list of fully qualified class names (e.g. 'mfai.pytorch.losses.dice.DiceLoss').
    """
    matches = []
    for module_name in iter_all_modules(package_path):
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if name.startswith("_"):
                continue
            if getattr(obj, "__module__", None) != module_name:
                continue
            if is_subclass_of_names(obj, base_names):
                matches.append(f"{module_name}.{name}")

    return matches


def write_rst(title: str, sections: list[dict[Literal["title", "classes"], str]], output_path: Path) -> None:
    """Generate a RST file with autosummary blocks for the given sections.

    Each section produces a titled subsection with an autosummary directive
    listing the provided classes. Sections with no classes are skipped.

    Args:
        title: The top-level title of the RST page.
        sections: A list of dicts, each with:
            - 'title' (str): the section heading.
            - 'classes' (list[str]): fully qualified class names to include.
        output_path: The Path where the RST file will be written.
    """
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    for section in sections:
        if not section["classes"]:
            continue

        if section.get("title") and section["title"] != title:
            lines.append(section["title"])
            lines.append("-" * len(section["title"]))
            lines.append("")

        lines.append(".. autosummary::")
        lines.append("   :toctree: generated")
        lines.append("   :nosignatures:")
        lines.append("")
        for cls in section["classes"]:
            lines.append(f"   ~{cls}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


if __name__ == "__main__":
    write_rst(
        title="Models",
        sections=[
            {"title": "Models", "classes": ["mfai.pytorch.models.base.ModelABC"]},
            {
                "title": "Models",
                "classes": get_classes_matching(
                    "mfai.pytorch.models",
                    [["ModelABC", "Module"], "BaseModel"],
                ),
            },
        ],
        output_path=Path("doc/api/models.rst"),
    )

    write_rst(
        title="Losses",
        sections=[
            {
                "title": "Losses",
                "classes": get_classes_matching(
                    "mfai.pytorch.losses",
                    ["Module"],
                ),
            }
        ],
        output_path=Path("doc/api/losses.rst"),
    )

    write_rst(
        title="Lightning",
        sections=[
            {
                "title": "Lightning Modules",
                "classes": get_classes_matching(
                    "mfai.pytorch.lightning_modules",
                    ["LightningModule"],
                ),
            },
            {
                "title": "Callbacks",
                "classes": get_classes_matching(
                    "mfai.pytorch",
                    ["Callback"],
                ),
            },
        ],
        output_path=Path("doc/api/lightning.rst"),
    )

    write_rst(
        title="Transforms",
        sections=[
            {
                "title": "Transforms",
                "classes": get_classes_matching(
                    "mfai.pytorch.transforms",
                    ["object"],
                ),
            }
        ],
        output_path=Path("doc/api/transforms.rst"),
    )

    write_rst(
        title="Metrics",
        sections=[
            {
                "title": "Metrics",
                "classes": get_classes_matching(
                    "mfai.pytorch.metrics",
                    ["Metric"],
                ),
            }
        ],
        output_path=Path("doc/api/metrics.rst"),
    )
