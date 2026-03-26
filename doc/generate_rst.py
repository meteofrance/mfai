"""
This script automatically generates RST files for the mfai Sphinx documentation.
Instead of relying solely on sphinx-apidoc (which generates a flat, unstructured output),
this script introspects the mfai codebase at build time to find classes matching specific
inheritance criteria, and generates structured autosummary RST files grouped by category
(Models, Losses, Lightning, Transforms, Metrics).

Each class in the Models section gets an inline Mermaid class diagram showing:
- The class itself with its attributes and methods
- Its parents (from MRO, filtered to relevant ones)
- Its children (other classes in the same section that inherit from it)

Requires sphinxcontrib-mermaid in conf.py extensions.

This script is meant to be run before the Sphinx build step, either locally or in CI.
"""

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Generator, Literal, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers — class discovery (unchanged from original)
# ---------------------------------------------------------------------------

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


def get_classes_matching(
    package_path: str, base_names: Sequence[str | list[str]]
) -> list[str]:
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


# ---------------------------------------------------------------------------
# Mermaid diagram generation
# ---------------------------------------------------------------------------

# Base classes shown as <<abstract>> roots in diagrams (no members displayed)
ABSTRACT_ROOTS = {"ModelABC", "BaseModel", "Module", "LightningModule", "ModelSettings"}

# Methods excluded from diagrams (too noisy)
EXCLUDED_METHODS = {
    "__init__", "__str__", "__repr__", "__eq__", "__hash__",
    "__lt__", "__le__", "__gt__", "__ge__", "__ne__",
    "__class__", "__module__", "__dict__", "__weakref__",
}


def _visibility(name: str) -> str:
    """Return Mermaid visibility prefix from Python naming convention."""
    if name.startswith("__"):
        return "-"
    if name.startswith("_"):
        return "#"
    return "+"


def _type_hint(annotation) -> str:
    """Return a short, readable type hint string."""
    if annotation is inspect.Parameter.empty:
        return ""
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    hint = str(annotation)
    # Clean up typing noise: typing.Optional[X] -> Optional[X]
    hint = hint.replace("typing.", "")
    return hint


def _class_block(cls: type) -> list[str]:
    """
    Generate the Mermaid `class Foo { ... }` block for a given class.

    Args:
        cls: The class object.
    """
    name = cls.__name__

    lines = [f"    class {name} {{"]

    # Attributes — from __annotations__ defined directly on the class (not inherited)
    annotations = cls.__dict__.get("__annotations__", {})
    for attr_name, attr_type in annotations.items():
        if attr_name.startswith("__"):
            continue
        vis = _visibility(attr_name)
        type_str = _type_hint(attr_type)
        type_prefix = f"{type_str} " if type_str else ""
        lines.append(f"        {vis}{type_prefix}{attr_name}")

    # Methods — defined directly on this class (not inherited)
    for method_name, method_obj in inspect.getmembers(cls, predicate=inspect.isfunction):
        if method_name in EXCLUDED_METHODS:
            continue
        if method_name not in cls.__dict__:
            continue
        vis = _visibility(method_name)
        sig = inspect.signature(method_obj)
        return_type = _type_hint(sig.return_annotation)
        ret_str = f" {return_type}" if return_type else ""
        lines.append(f"        {vis}{method_name}(){ret_str}")

    lines.append("    }")
    return lines


def _resolve_class(fqn: str) -> type | None:
    """Import and return a class from its fully qualified name."""
    module_path, class_name = fqn.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name, None)
    except ImportError:
        return None


def diagram_for_class(fqn: str, all_fqns: list[str]) -> str:
    """
    Generate a Mermaid classDiagram centered on a single class showing:
    - The class itself with its own attributes and methods
    - Its parents (from MRO, up to ABSTRACT_ROOTS)
    - Its children (other classes in all_fqns that inherit from it)

    Args:
        fqn: Fully qualified name of the focal class.
        all_fqns: All classes in the section (used to find children).

    Returns:
        A Mermaid classDiagram string (without the ```mermaid fence).
    """
    cls = _resolve_class(fqn)
    if cls is None:
        return ""

    lines = ["classDiagram"]

    # --- Focal class (full detail) ---
    lines += _class_block(cls)

    # --- Parents (walk MRO, stop at object) ---
    for parent in cls.__mro__[1:]:
        if parent is object:
            break
        parent_name = parent.__name__
        if parent_name in ABSTRACT_ROOTS:
            # Show as abstract root, no members
            lines += _class_block(parent)
            lines.append(f"    <<abstract>> {parent_name}")
            lines.append(f"    {parent_name} <|-- {cls.__name__} : herits")
        else:
            # Show with members if it's a known project class
            parent_fqn = next((f for f in all_fqns if f.endswith(f".{parent_name}")), None)
            if parent_fqn:
                parent_cls = _resolve_class(parent_fqn)
                if parent_cls:
                    lines += _class_block(parent_cls)
                    lines.append(f"    {parent_name} <|-- {cls.__name__} : hérite")

    # --- Children (classes in all_fqns that inherit from cls) ---
    for child_fqn in all_fqns:
        if child_fqn == fqn:
            continue
        child_cls = _resolve_class(child_fqn)
        if child_cls is None:
            continue
        if cls in child_cls.__mro__[1:] and child_cls.__name__ != cls.__name__:
            lines += _class_block(child_cls)
            lines.append(f"    {cls.__name__} <|-- {child_cls.__name__} : hérite")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RST generation
# ---------------------------------------------------------------------------

def write_rst(
    title: str,
    sections: list[dict[Literal["title", "classes"], str | list[str]]],
    output_path: Path,
    with_diagrams: bool = False,
) -> None:
    """Generate a RST file with optional Mermaid diagrams and autosummary blocks.

    Each section produces a titled subsection. When `with_diagrams=True`, each
    class gets an individual ``.. mermaid::`` directive before being listed in
    the autosummary block.

    Args:
        title: The top-level title of the RST page.
        sections: A list of dicts, each with:
            - 'title' (str): the section heading.
            - 'classes' (list[str]): fully qualified class names to include.
        output_path: The Path where the RST file will be written.
        with_diagrams: If True, insert a Mermaid class diagram for each class.
    """
    # Collect all class FQNs across the page (needed to find children)
    all_fqns = [cls for section in sections for cls in section.get("classes", [])]

    lines: list[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    for section in sections:
        if not section["classes"]:
            continue

        if section.get("title") and section["title"] != title:
            if isinstance(section["title"], list):
                raise ValueError("Title should be a string")
            lines.append(section["title"])
            lines.append("-" * len(section["title"]))
            lines.append("")

        if with_diagrams:
            # One subsection per class: diagram + autosummary entry
            for fqn in section["classes"]:
                class_name = fqn.rsplit(".", 1)[-1]

                lines.append(class_name)
                lines.append("~" * len(class_name))
                lines.append("")

                # Mermaid diagram
                diagram = diagram_for_class(fqn, all_fqns)
                if diagram:
                    lines.append(".. mermaid::")
                    lines.append("")
                    for diagram_line in diagram.splitlines():
                        lines.append(f"   {diagram_line}")
                    lines.append("")

                # Autosummary for this single class
                lines.append(".. autosummary::")
                lines.append("   :toctree: generated")
                lines.append("   :nosignatures:")
                lines.append("")
                lines.append(f"   ~{fqn}")
                lines.append("")

        else:
            # Original behaviour: one autosummary block for all classes
            lines.append(".. autosummary::")
            lines.append("   :toctree: generated")
            lines.append("   :nosignatures:")
            lines.append("")
            for cls in section["classes"]:
                lines.append(f"   ~{cls}")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Written {output_path} ({len(all_fqns)} classes)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
        with_diagrams=True,
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
