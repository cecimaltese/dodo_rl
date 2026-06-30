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
* DOMAIN RANDOMIZATION (for robustness / sim-to-real margin): added base mass + COM
  offset, perturbed reset pose/velocity, and observation noise (enable_corruption).

Why this matters for transfer
------------------------------
The earlier "same policy, sim-to-sim" failure turned out to be two concrete bugs,
both now fixed outside this file: (a) a wrong joint order, and (b) MuJoCo adding
passive joint damping/frictionloss that IsaacLab doesn't have (zeroed in
sim_env.py). With those fixed, the IMPLICIT actuator (assets/dodo.py) transfers as-is.
The randomization below is then layered on top for robustness — it forces the policy
to handle a *range* of dynamics and disturbances rather than overfitting one engine,
which is what buys margin for the real robot.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import DodoFlatEnvCfg


# --- Curriculum stage --------------------------------------------------------------
# Bootstrapping a balance policy with ALL disturbances on from iteration 0 tends to
# stick the mean reward at the termination floor (~ -200 * dt = -4.0): the robot is
# spawned into states it can't yet recover from, falls immediately every episode, and
# never learns. So train in two stages:
#   _STAGE = 0  LEARN TO STAND  — no pushes, near-upright spawn, no mass/COM/noise.
#               Mean reward should climb POSITIVE and episode length reach the cap.
#   _STAGE = 1  FULL            — pushes + start perturbation + mass/COM + obs noise
#               (the push-resistance task). Set this once stage 0 stands, then retrain
#               or RESUME from the stage-0 checkpoint (resume is faster + more stable).
_STAGE = 0


@configclass
class DodoStandEnvCfg(DodoFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Observations: match the real robot's balance-policy layout ---
        # base linear velocity is unreliable from the IMU -> never observe it.
        self.observations.policy.base_lin_vel = None
        # no velocity command for a pure balance policy.
        self.observations.policy.velocity_commands = None
        # Obs corruption (IMU/encoder noise) ON only in the FULL stage — clean obs
        # while it first learns to stand, then noisy for real-robot robustness.
        self.observations.policy.enable_corruption = (_STAGE >= 1)

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
        # --- Anti-wobble / anti-walk tuning ------------------------------------
        # Symptom in PLAY: it stays up but WOBBLES (underdamped — partly because
        # kd=0.5 is intentionally low to match the real motors) and DRIFTS / takes
        # little STEPS instead of holding one spot. These weights push it toward a
        # still, planted stance WITHOUT stiffening it so much it can't catch a push.
        # If it goes sluggish/passive under shoves, dial these back (see
        # REWARD_TUNING.md). Change ONE at a time when tuning further.
        self.rewards.ang_vel_xy_l2.weight = -0.25    # was -0.1 : kill base rocking
        self.rewards.action_rate_l2.weight = -0.05   # was -0.01: smoother, less twitch
        self.rewards.dof_acc_l2.weight = -2.5e-7     # was -1.25e-7: damp joint jitter
        # Sharper "hold still" only in FULL stage (tighter exp kernel penalizes drift
        # harder). Keep the looser 0.5 while bootstrapping so early reward isn't sparse.
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25 if _STAGE >= 1 else 0.5

        # Plant the feet — stop the sliding/stepping that reads as "walking".
        self.rewards.feet_slide.weight = -0.5        # was -0.2
        self.rewards.feet_air_time.weight = 0.0      # no incentive to step at all

        # Hold the home pose (penalize deviation of every joint from default).
        # NOTE: keep this MODEST. Too strong and it rigidly freezes the pose and
        # can't move to recover from a push.
        self.rewards.joint_deviation_all = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        # =====================================================================
        # Events / domain randomization  (gated by _STAGE)
        # =====================================================================

        if _STAGE < 1:
            # ---- STAGE 0: LEARN TO STAND -------------------------------------
            # Strip every disturbance so the policy can bootstrap a stable stance.
            # No pushes, near-upright zero-velocity spawn, no mass/COM/noise.
            self.events.push_robot = None
            self.events.add_base_mass = None
            self.events.base_com = None
            self.events.reset_base.params = {
                "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05),
                               "yaw": (-0.2, 0.2)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                                   "roll": (0.0, 0.0), "pitch": (0.0, 0.0),
                                   "yaw": (0.0, 0.0)},
            }
            self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        else:
            # ---- STAGE 1: FULL push-resistance + domain randomization --------

            # (1) Mild periodic pushes — the core "resist a shove" signal.
            # push_by_setting_velocity snaps the base to a random velocity a few
            # times per episode; the policy must absorb the kick and recover.
            # GENTLE per request; raise the ranges (e.g. +/-0.6) for a harder
            # push curriculum once it holds these reliably.
            self.events.push_robot = EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(3.0, 6.0),
                params={
                    "velocity_range": {
                        "x": (-0.3, 0.3),
                        "y": (-0.3, 0.3),
                        "yaw": (-0.3, 0.3),
                    }
                },
            )

            # (2) Initial pose / velocity perturbation — start slightly tilted and
            # drifting so it LEARNS to recover, not just hold a perfect pose.
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
            self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)

            # (3) Added base mass + COM offset — battery/cables + imperfect COM.
            self.events.add_base_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="body"),
                    "mass_distribution_params": (-0.3, 0.5),  # kg (robot ~4.7 kg)
                    "operation": "add",
                },
            )
            # COM event name/signature varies across IsaacLab versions; only wire it
            # up if this build exposes it, else skip cleanly instead of crashing.
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

            # (4) Obs noise is ON via enable_corruption above. Action/command DELAY
            # stays OFF (needs an explicit-PD actuator; we're on implicit). Add it
            # later as a deliberate sim-to-real step. See assets/dodo.py.


@configclass
class DodoStandEnvCfg_PLAY(DodoStandEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # Clean observations for eyeballing behavior (no training noise).
        self.observations.policy.enable_corruption = False
        # Pushes OFF in PLAY (training keeps them on). This gives a clean "can it
        # just stand still?" demo. To eyeball push resistance instead, comment the
        # next line out and the inherited train-time push_robot stays active.
        self.events.push_robot = None
        # Also drop the continuous external force/torque so PLAY is fully quiet.
        self.events.base_external_force_torque = None
