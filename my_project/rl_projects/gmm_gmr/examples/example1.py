import numpy as np
import matplotlib.pyplot as plt
from my_project.rl_projects.gmm_gmr.mixtures import GMM_GMR

np.set_printoptions(threshold=np.inf)

with open(
    "/home/yangxf/Ominverse_RL_platform/IsaacLab/my_project/rl_projects/gmm_gmr/data/ee_positions.txt",
    "r",
) as f1:
    ee_pos = np.loadtxt(f1)
    ee_pos = np.array(ee_pos)

with open(
    "/home/yangxf/Ominverse_RL_platform/IsaacLab/my_project/rl_projects/gmm_gmr/data/ee_rotations.txt",
    "r",
) as f2:
    ee_rot = np.loadtxt(f2)
    ee_rot = np.array(ee_rot)

ee_poses = np.concatenate((ee_pos, ee_rot), axis=-1)
print(ee_poses)

gmm_gmr = GMM_GMR(ee_poses, 3)
gmm_gmr.fit()

f, axarr = plt.subplots(3, 1)

for i in range(len(ee_poses)):
    for j in range(3):
        axarr[j].plot(gmm_gmr.trajectories[i, :, j], linestyle=":")

for j in range(3):
    axarr[j].scatter(
        gmm_gmr.centers_temporal, gmm_gmr.centers_spatial[:, j], label="centers"
    )

times, trj = gmm_gmr.generate_trajectory(0.1)
for j in range(3):
    axarr[j].plot(times, trj[:, j], label="estimated")

plt.legend()
plt.show()
