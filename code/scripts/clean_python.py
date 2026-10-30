"""Launch project commands without host sitecustomize/usercustomize hooks.

Use with python -I -S. site.main still initializes venv paths and installed
packages (including .pth files); only the two host customization hooks are skipped.
"""

import importlib.util
import os
from pathlib import Path
import runpy
import sys


def python_command(executable=None):
    """Explicit clean interpreter prefix, also usable by nested subprocesses."""
    return [str(executable or sys.executable), "-X", "utf8", "-I", "-S",
            str(Path(__file__).resolve())]


def pip_source():
    spec = importlib.util.find_spec("pip")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("The notebook interpreter must provide pip for bootstrap")
    return str(Path(next(iter(spec.submodule_search_locations))).resolve())


def main():
    if not sys.flags.no_site or not sys.flags.isolated:
        raise RuntimeError("clean_python.py requires -I -S")
    import site

    site.execsitecustomize = lambda: None
    site.execusercustomize = lambda: None
    site.main()
    # Nested project commands can opt into this same interpreter prefix.
    os.environ["PROMPTALIGN_CLEAN_PYTHON"] = "1"
    args = sys.argv[1:]
    if args[:1] == ["--pip-source"]:
        package = Path(args[1])
        spec = importlib.util.spec_from_file_location(
            "pip", package / "__init__.py",
            submodule_search_locations=[str(package)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["pip"] = module
        spec.loader.exec_module(module)
        # Load only parent's pip package, never its site-packages directory.
        args = ["-m", "pip", *args[2:]]
    if args[:1] == ["-I"]:
        args = args[1:]
    if args[:1] == ["-m"]:
        sys.path.insert(0, str(Path.cwd()))
        sys.argv = args[1:]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    elif args[:1] == ["-c"]:
        sys.argv = ["-c", *args[2:]]
        exec(compile(args[1], "<string>", "exec"), {"__name__": "__main__"})
    elif args:
        script = Path(args[0]).resolve()
        sys.path.insert(0, str(script.parent))
        sys.argv = args
        runpy.run_path(str(script), run_name="__main__")
    else:
        raise ValueError("Expected -m module, -c code or script")


if __name__ == "__main__":
    main()
