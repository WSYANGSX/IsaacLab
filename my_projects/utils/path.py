from __future__ import annotations

import os
import carb


def get_logs_path():
    isaac_logs_path = os.environ.get("ISAACLAB_LOGS_PATH")

    assert isaac_logs_path, "Can not find 'ISAACLAB_LOGS_PATH' in .bashrc"
    carb.log_info(f"isaaclab logs path: {isaac_logs_path}")

    return isaac_logs_path
