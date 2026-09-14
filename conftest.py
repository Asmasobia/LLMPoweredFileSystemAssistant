"""
pytest configuration for this project.

WHY THIS FILE EXISTS AT ALL — it looks like boilerplate, and the reason is not
obvious until it bites you:

When pytest collects `tests/test_fs_tools.py` and that directory has no
`__init__.py`, pytest inserts the *test file's own directory* (`tests/`) at the
front of `sys.path` — not the project root. So `import fs_tools` fails with
ModuleNotFoundError even though the file is sitting right there one level up.

The presence of a `conftest.py` at the project root makes pytest treat this
directory as the rootdir, and the explicit `sys.path` insertion below guarantees
the import works no matter which directory pytest was invoked from. That last part
matters more than it sounds: `pytest` and `pytest tests/` and
`python -m pytest tests/test_fs_tools.py` all set up paths slightly differently,
and a test suite that only passes when run one particular way is a test suite
people stop running.

The alternative is installing the project as a package (`pip install -e .`), which
is the better answer for a library. This is a single-directory application, so the
two-line version is proportionate.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# `insert(0, ...)` rather than `append`: if some installed package happens to be
# named `fs_tools`, we want *ours*, not theirs.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
