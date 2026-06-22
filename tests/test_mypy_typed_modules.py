import subprocess

from scripts import TYPED_MYPY_PATHS


def test_type_check_typed_modules():
    result = subprocess.run(["mypy", *TYPED_MYPY_PATHS], check=False)
    assert result.returncode == 0, "mypy failed on typed modules"
