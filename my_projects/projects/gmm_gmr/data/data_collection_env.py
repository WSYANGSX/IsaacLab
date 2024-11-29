from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.tasks import PickPlace

my_world = World(stage_units_in_meters=1.0)
my_task = PickPlace()
my_world.add_task(my_task)
my_world.reset()
task_params = my_task.get_params()
my_franka = my_world.scene.get_object(task_params["robot_name"]["value"])
my_controller = PickPlaceController(
    name="pick_place_controller",
    gripper=my_franka.gripper,
    robot_articulation=my_franka,
)
articulation_controller = my_franka.get_articulation_controller()

i = 0
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
        observations = my_world.get_observations()

        with open(f"my_project/rl_projects/gmm_gmr/data/ee_pose{i}.txt", "a") as f:
            ee_pos = observations[task_params["robot_name"]["value"]][
                "end_effector_position"
            ]
            ee_quat = observations[task_params["robot_name"]["value"]][
                "end_effector_quat"
            ]
            ee_pose = np.concatenate((ee_pos, ee_quat), axis=-1)
            ee_pose = ee_pose.reshape(1, -1)
            np.savetxt(f, ee_pose)

        actions = my_controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]][
                "position"
            ],
            placing_position=observations[task_params["cube_name"]["value"]][
                "target_position"
            ],
            current_joint_positions=observations[task_params["robot_name"]["value"]][
                "joint_positions"
            ],
            end_effector_offset=np.array([0, 0.005, 0]),
        )
        if my_controller.is_done():
            print("done picking and placing")
            reset_needed = True
            i += 1
        articulation_controller.apply_action(actions)
simulation_app.close()
