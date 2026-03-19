import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def is_subclass_of_names(cls, base_names: list[str | list[str]]):
    try:
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
    except TypeError:
        return False


def iter_all_modules(package_path: str):
    try:
        pkg = importlib.import_module(package_path)
    except ImportError:
        return

    yield package_path

    if hasattr(pkg, "__path__"):
        for mod_info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package_path}."):
            yield mod_info.name


def get_classes_matching(package_path: str, base_names: list[str]):
    matches = []
    for module_name in iter_all_modules(package_path):
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if name.startswith("_"):
                continue
            if getattr(obj, "__module__", None) != module_name:
                continue
            if is_subclass_of_names(obj, base_names):
                matches.append(f"{module_name}.{name}")

    return matches


def write_rst(title: str, sections: list[dict], output_path: Path):
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
