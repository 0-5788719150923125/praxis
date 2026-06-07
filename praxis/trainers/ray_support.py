"""Runtime install of the optional Ray dependency, integration-loader style.

Ray is only needed by the ``mono_forward_ray`` trainer and publishes no wheels
for Python >= 3.14, so it is neither a hard dependency nor installed by the
launch script. Like integrations, it installs lazily at runtime - but only when
the parsed args actually select the Ray trainer, and only on an interpreter
where the wheels exist. Must run before anything imports ``ray`` (the trainer
defers its import to ``fit()``, so calling this from main() is early enough).
"""

import importlib.util
import subprocess
import sys

# Interpreter range with upstream Ray wheels. Bump the upper bound when Ray
# starts publishing wheels for newer Pythons.
RAY_PYTHON_MIN = (3, 10)
RAY_PYTHON_MAX = (3, 13)

# Matches the '[ray]' extra in pyproject.toml.
RAY_REQUIREMENT = "ray[default]>=2.30"

_NO_WHEELS_MSG = """\
[ERROR] --trainer-type mono_forward_ray requires the optional 'ray' package,
        and Ray publishes no wheels for Python {version}.

        The typical path is the Docker compose environment (Ubuntu 24.04 +
        Python 3.12, where Ray installs cleanly):

            ./launch compose --mike --trainer-type mono_forward_ray ...

        For single-host training, use --trainer-type mono_forward
        (the in-process profile, no Ray dependency).
"""


def trainer_needs_ray(trainer_type: str) -> bool:
    return trainer_type == "mono_forward_ray"


def ensure_ray(trainer_type: str) -> None:
    """Install Ray at runtime if the selected trainer needs it.

    No-op for non-Ray trainers or when Ray is already importable. Exits with an
    actionable message (rather than an opaque mid-training ImportError) when the
    interpreter has no Ray wheels or the install fails.
    """
    if not trainer_needs_ray(trainer_type):
        return
    if importlib.util.find_spec("ray") is not None:
        return

    version = (sys.version_info[0], sys.version_info[1])
    if not (RAY_PYTHON_MIN <= version <= RAY_PYTHON_MAX):
        sys.exit(_NO_WHEELS_MSG.format(version=f"{version[0]}.{version[1]}"))

    print(f"[ENV] Installing optional Ray dependency ({RAY_REQUIREMENT})...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", RAY_REQUIREMENT]
        )
    except subprocess.CalledProcessError:
        sys.exit(
            "[ERROR] Ray install failed. Install manually with\n"
            f"        pip install '{RAY_REQUIREMENT}'\n"
            "        or run inside Docker: ./launch compose ..."
        )
    if importlib.util.find_spec("ray") is None:
        sys.exit("[ERROR] Ray installed but is not importable in this environment.")
    print("[ENV] Ray installation complete.")
