"""Configuration for the Dodo bipedal robot.

Robot: dodo_daimao
Joints (8 revolute + 2 fixed soles):
    hip_left / hip_right             - hip roll   (axis X), effort 27 Nm
    upper_leg_left / upper_leg_right - hip pitch  (axis Y), effort 27 Nm
    lower_leg_left / lower_leg_right - knee pitch (axis Y), effort  9 Nm
    foot_left / foot_right           - ankle      (axis Y), effort  9 Nm
    foot_sole_left / foot_sole_right - fixed (contact surface)

Total mass: ~4.7 kg
Approximate standing height: ~0.45 m
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

# USD path — we reuse the converted USD from the previous team's project.
# To reconvert from URDF: Isaac Sim → File → Import → URDF → select dodo_daimao.urdf
_USD_PATH = Path(__file__).resolve().parent / "usd" / "dodo_daimao.usd"

DODO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Spawn height: upper_leg(0.199) + lower_leg(0.199) + foot(~0.05) ≈ 0.45m
        # Add margin so the robot doesn't clip through the ground
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            # Bird-like crouch: the knee (lower_leg) folds BACKWARD (negative),
            # not forward/humanoid. Validated upright in MuJoCo at ~0.44 m base
            # height. Must stay in sync with rl_env._DEFAULT_POS_PATTERNS — change
            # both together and retrain (this pose is the policy's action offset
            # and observation reference).
            "hip_.*": 0.0,             # hip roll neutral
            "upper_leg_.*": 0.20,      # hip pitch (thigh forward)
            "lower_leg_.*": -0.50,     # knee folds backward (bird-like)
            "foot_.*": 0.30,           # ankle keeps the sole flat
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # --- Actuators ---------------------------------------------------------------
    # DelayedPDActuatorCfg (NOT ImplicitActuatorCfg) for sim-to-sim/sim-to-real
    # transfer. Two deliberate choices:
    #   1) EXPLICIT PD. The implicit actuator solves the PD drive *inside* PhysX,
    #      which behaves differently from the explicit torque PD that MuJoCo
    #      (sim_env.py) and the real Damiao/ODrive firmware apply. A policy tuned
    #      to the implicit solver fell over on the sim-to-sim swap. DelayedPD
    #      computes tau = kp*(q_des - q) - kd*qd in Python and applies it as a
    #      torque, exactly matching the MuJoCo PD loop -> the two engines now agree.
    #   2) COMMAND DELAY. min_delay/max_delay buffer the command by 0-4 physics
    #      steps (sim dt 0.005 s -> 0-20 ms), i.e. up to ~one 50 Hz control cycle of
    #      comms latency. This is the "action delay" half of the obs/action-delay
    #      randomization; a fresh delay is sampled every reset.
    # Gains kp=30, kd=0.5 match the real Damiao hips and the MuJoCo PD (KP/KD in
    # sim_env.py) — keep all three in sync. effort_limit set both ways so the limit
    # holds whether the build reads effort_limit or effort_limit_sim.
    actuators={
        "hip_upper": DelayedPDActuatorCfg(
            joint_names_expr=["hip_.*", "upper_leg_.*"],
            stiffness=30.0,
            damping=0.5,
            armature=0.01,
            effort_limit=27.0,
            effort_limit_sim=27.0,
            velocity_limit=6.0,
            velocity_limit_sim=6.0,
            min_delay=0,
            max_delay=4,
        ),
        "lower_foot": DelayedPDActuatorCfg(
            joint_names_expr=["lower_leg_.*", "foot_.*"],
            stiffness=30.0,
            damping=0.5,
            armature=0.01,
            effort_limit=9.0,
            effort_limit_sim=9.0,
            velocity_limit=6.0,
            velocity_limit_sim=6.0,
            min_delay=0,
            max_delay=4,
        ),
    },
)
