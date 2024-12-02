import h5py
import matplotlib.pyplot as plt


file_path = f"/home/yangxf/Ominverse_RL_platform/IsaacLab/my_project/rl_projects/gmm_gmr/data/can/low_dim_v141.hdf5"
with h5py.File(file_path, "r") as f:
    data_group = f["data"]
    ee_pos = {}
    for i in range(20):
        ee_pos[f"demo_{i}"] = data_group[f"demo_{i}/obs/robot0_eef_pos"][:]

figure1 = plt.figure(figsize=(20, 20))
axe1 = plt.axes(projection="3d")

# 绘图
axe1.set_title("can task trajectories curve")
axe1.set_xlabel("x(mm)")
axe1.set_ylabel("y(mm)")
axe1.set_zlabel("z(mm)")


for key, value in ee_pos.items():
    x = value[:, 0] * 1000
    y = value[:, 1] * 1000
    z = value[:, 2] * 1000

    axe1.plot3D(x, y, z, label=key)


plt.legend()

plt.show()
