from dtw import dtw
import numpy as np
import math


def trajectories_time_align(data):
    ls = np.argmax([d.shape[0] for d in data])

    data_warp = []

    for d in data:
        dist, cost, acc, path = dtw(
            data[ls], d, dist=lambda x, y: np.linalg.norm(x - y, ord=1)
        )

        data_warp += [d[path[1]][: data[ls].shape[0]]]

    return data_warp


def gaussian(x, mu, var):
    exponent = -((x - mu) ** 2) / (2 * var)
    return (1 / math.sqrt((2 * math.pi * var))) * math.exp(exponent)


def trajectories_space_align(data):
    data = np.array(data, dtype=np.float32)

    dist = [np.linalg.norm(d[-1][0:3] - d[0][0:3], ord=2, axis=-1) for d in data]
    ld_index = np.argmax(dist)
    ld = dist[ld_index]

    # 将距离最大曲线


if __name__ == "__main__":
    data = np.array(
        [
            [[0, 0, 0], [1, 2, 3], [4, 5, 6]],
            [[0, 0, 0], [1, 2, 3], [100, 9, 10]],
            [[0, 0, 0], [1, 2, 3], [8, 9, 10]],
        ]
    )
    trajectories_space_align(data=data)
