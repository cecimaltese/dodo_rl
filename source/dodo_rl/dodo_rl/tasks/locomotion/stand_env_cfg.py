"""Stand-and-balance env for Dodo (flat terrain).

First deployment target: hold a stable two-leg stance and resist mild pushes —
NOT walking. Built on the flat velocity env, then stripped down so the trained
policy matches what the real robot can actually observe.

Key differences vs DodoFlatEnvCfg
---------------------------------
* Observation drops `base_lin_vel` (can't be estimated reliably from the IMU at
  deployment) and `velocity_commands` (a pure balance policy has no command).
  Remaining policy obs: base_ang_vel, projected_gravity, joint_pos, joint_vel,
  actions — exactly the layout MotorController.to_policy_vector() produces on the
  real robot.
* Commands are zeroed: the velocity-tracking reward then rewards *staying still*.
* Rewards favor staying upright / alive / near the home pose instead of gait.
* A periodic push event is enabled so the policy learns to resist disturbances.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import DodoFlatEnvCfg


@configclass
class DodoStandEnvCfg(DodoFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Observations: match the real robot's balance-policy layout ---
        # base linear velocity is unreliable from the IMU -> never observe it.
        self.observations.policy.base_lin_vel = None
        # no velocity command for a pure balance policy.
        self.observations.policy.velocity_commands = None

        # --- Commands: zero everything (track-zero == stand still) ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # --- Rewards: balance, not gait ---
        # Tracking a zero command -> dense "don't drift / don't rotate" signal.
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        # Survival bonus: every step upright is good.
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=1.0)
        # Stay upright and don't bob / tip.
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        # Hold the home pose (penalize deviation of every joint from default).
        self.rewards.joint_deviation_all = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        # Plant the feet (no sliding); kill the gait/air-time incentive.
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_air_time.weight = 0.0
        # Keep smoothness / effort penalties modest.
        self.rewards.action_rate_l2.weight = -0.01

        # --- Events: start near upright, then push it around a bit ---
        # Start standing still (no spawn velocity, minimal pose offset).
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        # Mild periodic shove so the policy learns to recover.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )


@configclass
class DodoStandEnvCfg_PLAY(DodoStandEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        # Keep push_robot enabled in PLAY — that's how you eyeball push resistance.
        self.events.base_external_force_torque = None
