import torch


def generate_random_orientation(num_samples=1):
    # 生成随机欧拉角
    # roll (围绕X轴) 和 yaw (围绕Z轴) 可以是任意值
    roll = torch.rand(num_samples) * 2 * torch.pi
    yaw = torch.rand(num_samples) * 2 * torch.pi

    # pitch (围绕Y轴) 需要确保z轴在下半球
    # 我们选择 -π/2 到 0 的范围，因为当pitch在这个范围内时，z轴会指向下半球面
    pitch = torch.rand(num_samples) * (-torch.pi / 2)

    # 将roll, pitch, yaw合并成一个张量
    orientations = torch.stack((roll, pitch, yaw), dim=1)

    return orientations


print(generate_random_orientation(1))