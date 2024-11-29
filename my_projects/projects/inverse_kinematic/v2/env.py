import torch
import numpy as np
from numpy import ndarray
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import add_reference_to_stage
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.articulations import Articulation
from pxr import UsdGeom
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *


class IKSolverTrain(BaseTask):
    def __init__(self, name: str, offset: ndarray | None = None) -> None:
        super().__init__(name, offset)
        self._robot = None
        self._device = "cuda:0"

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        # 添加地面
        scene.add_default_ground_plane()

        # 添加机器人stage
        self.get_robot()

        # 添加franka和机器人末端到scene中
        self._franka = Articulation(
            prim_path="/World/franka",
            position=self.robot_init_translation,
            orientation=self.robot_init_orientation,
            name="franka",
        )

        self._hands = RigidPrim(prim_path="/World/franka/panda_link7", name="hands")
        self._lfingers = RigidPrim(
            prim_path="/World/franka/panda_leftfinger", name="lfingers"
        )
        self._rfingers = RigidPrim(
            prim_path="/World/franka/panda_rightfinger",
            name="rfingers",
        )
        scene.add(self._franka)
        scene.add(self._hands)

        self._task_objects[self._franka.name] = self._franka
        self._task_objects[self._hands.name] = self._hands

    def init_data(self) -> None:
        def get_env_local_pose(robot_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - robot_pos[0]
            py = world_pos[1] - robot_pos[1]
            pz = world_pos[2] - robot_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor(
                [px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float
            )

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.franka_pos,
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/franka/panda_link7")),
            self._device,
        )
        lfinger_pose = get_env_local_pose(
            self.franka_pos,
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/franka/panda_leftfinger")),
            self._device,
        )
        rfinger_pose = get_env_local_pose(
            self.franka_pos,
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/franka/panda_rightfinger")),
            self._device,
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(
            hand_pose[3:7], hand_pose[0:3]
        )

        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot

        self.gripper_forward_axis = torch.tensor(
            [0, 0, 1], device=self._device, dtype=torch.float
        )
        self.gripper_up_axis = torch.tensor(
            [0, 1, 0], device=self._device, dtype=torch.float
        )

        dof_limits = self._franka._articulation_view.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0]
        self.franka_dof_upper_limits = dof_limits[0, :, 1]

    def get_robot(self):
        self.robot_init_translation = np.array([0.0, 0.0, 0.0])
        self.robot_init_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        robot_usd_path = "omniverse://localhost/Library/robots/franka.usd"
        add_reference_to_stage(robot_usd_path, "/World/franka")

    def get_params(self) -> dict:
        params_representation = dict()
        params_representation["robot_name"] = {
            "value": self._franka.name,
            "modifiable": False,
        }
        return params_representation

    def post_reset(self) -> None:
        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035],
            device=self._device,
        )
        self._franka.set_joint_positions(self.franka_default_dof_pos.cpu().numpy())
        self.franka_pos, self.franka_rot = self._franka.get_world_pose()
        self.init_data()
        self._franka.set_enabled_self_collisions(False)
        print("************* Task reset *****************")
        print("franka dof lower limits:", self.franka_dof_lower_limits)
        print("franka dof upper limits:", self.franka_dof_upper_limits)

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._hands.get_world_pose()
        franka_dof_pos = self._franka.get_joint_positions()
        self.franka_dof_pos = franka_dof_pos

        (
            self.franka_grasp_rot,
            self.franka_grasp_pos,
        ) = self.compute_grasp_transforms(
            torch.tensor(hand_rot, device=self._device),
            torch.tensor(hand_pos, device=self._device),
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
        )

        observations = {
            self._franka.name: {
                "arm_pos": self.franka_dof_pos[0:7],
                "grasp_pos": self.franka_grasp_pos,
                "grasp_rot": self.franka_grasp_rot,
            }
        }
        return observations

    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos