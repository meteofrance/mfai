"""Scripts that runs the py2puml tool on the mfai module

py2puml does not support type hinting Literal[]

To circumvent those limitations, create a modified copy of the source code.
"""

import os
from pathlib import Path
import pkgutil
import re
import shutil
import subprocess
from typing import Iterator


LIB_PATH: Path = Path('../../mfai/')
TEMP_PATH: Path = Path('./mfai/')


def clean_script(script_path: Path) -> str:
    """Reads and clean a script from the synthax unsuported by py2puml.
    """
    with open(script_path, 'r') as file:
        script: str = file.read()

        # typing.Literal unsuported: https://github.com/lucsorel/py2puml/issues/91
        script = re.sub(': .*Literal\[.*\].*=', '=', script)
        script = re.sub(': .*Literal\[.*\].*,', ',', script)
        script = re.sub(': .*Literal\[.*\].*$', '', script)

    return script


def get_packages(package_name: str) -> Iterator[pkgutil.ModuleInfo]:
    for module_info in pkgutil.walk_packages(package_name):
        if module_info.ispkg:
            yield from get_packages(module_info.name)
            yield module_info



if __name__ == '__main__':
    # walk, copy and clean all files in mfai
    for root_path, dir_paths, file_names in os.walk(LIB_PATH):
        root_path = Path(root_path)

        for file_name in file_names:
            file_name = Path(file_name)
            if file_name.suffix != '.py':
                continue
        
            script_cleaned: str = clean_script(root_path / file_name)
            out_dir = TEMP_PATH / root_path.relative_to(LIB_PATH)
            if not out_dir.exists():
                os.makedirs(out_dir)
            with open(out_dir / file_name, 'w+') as file:
                file.write(script_cleaned)

    # path, module name
    py2puml_args: list[tuple[str, str]] = [
        ("mfai/", "mfai")
    ]

    # Run py2puml
    for path, name in py2puml_args:
        puml: subprocess.CompletedProcess = subprocess.run(
            ["python3", "-m", "py2puml", str(path), str(name)],
            capture_output=True,
            )
        
        with open(f'{name}.puml', 'w+') as file:
            file.write(puml.stdout.decode("utf-8"))

    # Clean
    if TEMP_PATH.exists:
        shutil.rmtree(TEMP_PATH)