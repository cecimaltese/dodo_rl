"""Stand-and-balance env for Dodo (flat terrain), hardened to resist pushes.

First deployment target: hold a stable two-leg stance and recover from mild pushes
— NOT walking. Built on the flat velocity env, then stripped down so the trained
policy matches what the real robot can actually observe, and wrapped in domain
randomization so the policy survives the sim-to-sim (IsaacLab -> MuJoCo) and
sim-to-real gaps.

Key differences vs DodoFlatEnvCfg
---------------------------------
* Observation drops `base_lin_vel` (can't be estimated reliably from the IMU at
  deployment) and `velocity_commands` (a pure balance policy has no command).
  Remaining policy obs: base_ang_vel, projected_gravity, joint_pos, joint_vel,
  actions — exactly the layout MotorController.to_policy_vector() produces on the
  real robot. obs_dim = 30.
* Commands are zeroed: the velocity-tracking reward then rewards *staying still*.
* Rewards favor staying upright / alive / near the home pose instead of gait.
* PUSHES: a gentle periodic shove (push_robot) plus a perturbed start pose/velocity
  teach the policy to absorb a disturbance and return to balance.
* DOMAIN RANDOMIZATION (chosen for sim-to-sim/real transfer): added base mass + COM
  offset, perturbed reset pose/velocity, observation noise (enable_corruption), and
  an action/command delay (in the DelayedPDActuatorCfg over in assets/dodo.py).

Why this matters for transfer
------------------------------
The previous "same policy, sim-to-sim" attempt failed because a policy trained
against one engine's exact dynamics doesn't transfer to another. Two fixes work
together: (a) assets/dodo.py now uses an EXPLICIT PD actuator with a command delay
so IsaacLab's actuator matches the MuJoCo PD loop, and (b) the randomization below
forces the policy to be robust to a *range* of dynamics instead of overfitting to
one. MuJoCo (and the real robot) then look like just another sample from that range.
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
        # Keep observation corruption ON during training: the policy must learn to
        # tolerate the IMU / encoder noise it will see on the real robot (and that a
        # clean MuJoCo run lacks). This is the "obs noise" half of obs/action noise.
        self.observations.policy.enable_corruption = True

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

        # =====================================================================
        # Events / domain randomization
        # =====================================================================

        # --- (1) Mild periodic pushes: the core "resist a shove" signal --------
        # push_by_setting_velocity snaps the base to a random velocity a few times
        # per episode; the policy has to absorb the kick and recover balance. Kept
        # GENTLE (per request): small lateral shove + a little yaw spin. Raise the
        # ranges later (e.g. +/-0.6) once it holds these reliably -> a quick way to
        # build a push curriculum without touching the rewards.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),   # a push every few seconds
            params={
                "velocity_range": {
                    "x": (-0.3, 0.3),      # m/s lateral shove (gentle)
                    "y": (-0.3, 0.3),
                    "yaw": (-0.3, 0.3),    # rad/s twist
                }
            },
        )

        # --- (2) Initial pose / velocity perturbation --------------------------
        # Start slightly tilted and drifting so the policy LEARNS to recover rather
        # than only holding a perfect upright pose. Small roll/pitch + base velocity.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.5, 0.5),
            },
            "velocity_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0),
                "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3),
            },
        }
        # Small per-joint spread around the home pose at reset (~+/-10%), so it also
        # learns to recover from a slightly off-nominal joint configuration.
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)

        # --- (3) Added base mass + COM offset ----------------------------------
        # The real body carries a battery + cables and its true COM isn't exactly
        # the model's. Randomize both so balance tolerates a heavier / shifted body.
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="body"),
                "mass_distribution_params": (-0.3, 0.5),  # kg added (robot ~4.7 kg)
                "operation": "add",
            },
        )
        # COM randomization: the event function name/signature varies across IsaacLab
        # versions (it was reworked/removed in some builds), so only wire it up if
        # this build exposes it — otherwise skip cleanly instead of crashing import.
        if hasattr(mdp, "randomize_rigid_body_com"):
            self.events.base_com = EventTerm(
                func=mdp.randomize_rigid_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="body"),
                    "com_range": {
                        "x": (-0.02, 0.02),
                        "y": (-0.02, 0.02),
                        "z": (-0.02, 0.02),
                    },
                },
            )

        # --- (4) Action / command delay -> handled in the actuator model -------
        # assets/dodo.py uses DelayedPDActuatorCfg (explicit PD that matches the
        # MuJoCo PD loop) with a 0-4 physics-step (0-20 ms, ~one 50 Hz cycle) random
        # command delay. That is the "action delay" half of obs/action-delay; it
        # lives on the actuator, not here, so nothing to configure in this file.


@configclass
class DodoStandEnvCfg_PLAY(DodoStandEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # Clean observations for eyeballing behavior (no training noise).
        self.observations.policy.enable_corruption = False
        # Keep push_robot ENABLED in PLAY — that's how you eyeball push resistance.
        # Drop the extra continuous force/torque so the only disturbance you see is
        # the discrete shove from push_robot.
        self.events.base_external_force_torque = None
