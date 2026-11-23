"""Test that importing optimagic stays fast.

Importing optimagic must not import packages that are slow to import and only needed
for some features. Those are imported inside the functions that use them.

"""

import subprocess
import sys

SLOW_MODULES = [
    "altair",
    "bokeh",
    "joblib",
    "matplotlib",
    "numba",
    "plotly.express",
    "scipy",
    "sqlalchemy",
]


def test_import_does_not_import_slow_modules():
    code = "import optimagic, sys; print('\\n'.join(sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    imported = set(out.stdout.split())

    unexpected = {
        name
        for name in imported
        if any(name == slow or name.startswith(f"{slow}.") for slow in SLOW_MODULES)
    }
    assert not unexpected, f"`import optimagic` imported slow modules: {unexpected}"
