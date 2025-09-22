import pandas as pd
import matplotlib.pyplot as plt
from local_projects.projects.gmm_gmr.mixtures import GMM_GMR

data_path = "/home/yangxf/my_projects/IsaacLab/local_projects/projects/gmm_gmr/data/data.xlsx"
data1 = pd.read_excel(data_path, sheet_name="Sheet1", engine="openpyxl")
data2 = pd.read_excel(data_path, sheet_name="Sheet2", engine="openpyxl")
data3 = pd.read_excel(data_path, sheet_name="Sheet3", engine="openpyxl")
data4 = pd.read_excel(data_path, sheet_name="Sheet4", engine="openpyxl")

ee_poses = [data1.to_numpy(), data2.to_numpy(), data3.to_numpy(), data4.to_numpy()]

gmm_gmr = GMM_GMR(ee_poses, 2)
gmm_gmr.fit()

f, axarr = plt.subplots(2, 1)

for i in range(len(ee_poses)):
    for j in range(2):
        axarr[j].plot(gmm_gmr.trajectories[i, :, j], linestyle=":")

for j in range(2):
    axarr[j].scatter(
        gmm_gmr.centers_temporal, gmm_gmr.centers_spatial[:, j], label="centers"
    )

times, trj = gmm_gmr.generate_trajectory(0.1)
for j in range(2):
    axarr[j].plot(times, trj[:, j], label="estimated")

plt.legend()
plt.show()
