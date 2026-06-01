"""dodo_rl — RL locomotion package for the Dodo bipedal robot."""

import os
import toml

# Read extension.toml for metadata
EXTENSION_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
EXTENSION_TOML = toml.load(os.path.join(EXTENSION_DIR, "config", "extension.toml"))

# Import sub-packages so gym envs get registered on import
from . import assets  # noqa: F401
from . import tasks   # noqa: F401
