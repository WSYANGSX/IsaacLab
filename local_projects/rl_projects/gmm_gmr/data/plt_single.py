import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.inf)

with open(
    "/home/yangxf/Ominverse_RL_platform/IsaacLab/my_project/rl_projects/gmm_gmr/data/ee_pose0.txt",
    "r",
) as f1:
    ee_pos1 = np.loadtxt(f1)
    ee_pos1 = np.array(ee_pos1)[:, :3]

with open(
    "/home/yangxf/Ominverse_RL_platform/IsaacLab/my_project/rl_projects/gmm_gmr/data/ee_pose1.txt",
    "r",
) as f2:
    ee_pos2 = np.loadtxt(f2)
    ee_pos2 = np.array(ee_pos2)[:, :3]

with open(
    "/home/yangxf/Ominverse_RL_platform/IsaacLab/my_project/rl_projects/gmm_gmr/data/ee_pose2.txt",
    "r",
) as f3:
    ee_pos3 = np.loadtxt(f3)
    ee_pos3 = np.array(ee_pos3)[:, :3]

x1 = ee_pos1[:, 0] * 1000
y1 = ee_pos1[:, 1] * 1000
z1 = ee_pos1[:, 2] * 1000

x2 = ee_pos1[:, 0] * 1000
y2 = ee_pos1[:, 1] * 1000
z2 = ee_pos1[:, 2] * 1000

x3 = ee_pos1[:, 0] * 1000
y3 = ee_pos1[:, 1] * 1000
z3 = ee_pos1[:, 2] * 1000


figure1 = plt.figure(figsize=(10, 10))
axe1 = plt.axes(projection="3d")

# 绘图
axe1.plot3D(x1, y1, z1, color="blue", label="ee position")
axe1.plot3D(x2, y2, z2, color="red", label="ee position")
axe1.plot3D(x3, y3, z3, color="green", label="ee position")


axe1.set_title("ur10 straightline follow curve")
axe1.set_xlabel("x(mm)")
axe1.set_ylabel("y(mm)")
axe1.set_zlabel("z(mm)")

plt.legend()


# figure2 = plt.figure(figsize=(10, 5))
# plt.plot(
#     range(len(rotdist1)), rotdist1, color="blue", label="without control model switch"
# )
# plt.plot(range(len(rotdist2)), rotdist2, color="red", label="with control model switch")
# plt.title("ur10 PRTPR moving curve")
# plt.xlabel("step")
# plt.ylabel("rot_dist between target and ee")
# plt.axvline(433, linestyle="dashed")
# plt.xticks([433])
# plt.xticks([0, 100, 200, 300, 400, 433, 500, 600, 700])
# plt.legend()
# plt.grid()
# plt.text(435, 0.25, "control model switch")

plt.show()
