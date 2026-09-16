import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Point settings at a throwaway data dir before app modules import.
_TMP = tempfile.mkdtemp(prefix="movie-censor-test-")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP, "data"))
os.environ.setdefault("CENSORED_DIR", os.path.join(_TMP, "censored"))
os.environ.setdefault("AUTH_DISABLED", "1")


@pytest.fixture()
def tmpdir_path(tmp_path):
    return str(tmp_path)
