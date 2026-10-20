"""HSR-only field experiment: no gauge/readout configuration or TEST selection."""
from copy import deepcopy
import json
from pathlib import Path

from .config import DEFAULT as FIELD_DEFAULT, validate_config

SCHEMA = "nowcastnet-hsr-field-v5"
DEFAULT = deepcopy(FIELD_DEFAULT)


def load_field_config(path=None):
    config = deepcopy(DEFAULT)
    if path is not None:
        path = Path(path).expanduser().resolve()
        supplied = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict) or set(supplied) - set(config):
            raise ValueError("Unknown field-only config keys; CNN/gauge options are not accepted")
        for key, value in supplied.items():
            if isinstance(config[key], dict):
                if not isinstance(value, dict) or set(value) - set(config[key]):
                    raise ValueError(f"Unknown or invalid field settings in {key}")
                config[key].update(value)
            else:
                config[key] = value
    if config["schema"] != SCHEMA or config["test_issue_policy"] != "hsr_available_windows":
        raise ValueError("Field-only experiment schema/test issue policy changed")
    root = Path(config["project_root"]).expanduser()
    # A review bundle can set project_root='.' without depending on shell cwd.
    if not root.is_absolute():
        root = (path.parent.parent if path is not None else Path(FIELD_DEFAULT["project_root"])) / root
    root = root.resolve()
    for key in ("project_root", "experiment_root", "cache_root", "radar_root", "hsr_cache_root"):
        value = root if key == "project_root" else Path(config[key]).expanduser()
        config[key] = str((value if value.is_absolute() else root / value).resolve())
    validate_config(config)
    return config
