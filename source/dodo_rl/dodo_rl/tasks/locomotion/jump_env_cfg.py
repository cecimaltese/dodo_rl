from typing import Mapping


def jump_weight_fn(obs: Mapping[str, float]) -> float:
    """Weight function for rough_env that encourages jumping.

    The function rewards upward motion and base height above the ground.

    Args:
        obs: observation dictionary with keys such as
            - "base_height" or "z"
            - "base_velocity_z" or "vz"

    Returns:
        A scalar weight to shape the reward for jump behavior.
    """
    base_height = float(obs.get("base_height", obs.get("z", 0.0)))
    vertical_vel = float(obs.get("base_velocity_z", obs.get("vz", 0.0)))

    height_bonus = max(0.0, base_height - 0.15) * 2.0
    velocity_bonus = max(0.0, vertical_vel) * 0.7
    airtime_bonus = 1.0 if base_height > 0.2 else 0.0

    return airtime_bonus + height_bonus + velocity_bonus
