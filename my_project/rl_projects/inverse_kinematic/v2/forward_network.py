import torch
import torch.nn as nn
from my_project.rl_projects.inverse_kinematic.buffer import Buffer


class FKNet(nn.Module):
    """Franka 神经网络逆解控制器"""

    def __init__(self) -> None:
        super().__init__()
        self.create_network()
        self.init_network()
        self.loss = nn.MSELoss()
        self.trainer = torch.optim.Adam(self.net.parameters(), lr=0.00005)

    def create_network(self):
        self.net = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
        ).to(device="cuda:0")

    def init_network(self):
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

        self.net.apply(init_normal)

    def forward(
        self,
        current_arm_pos: torch.tensor,
        target_position: torch.tensor,
        target_orientation: torch.tensor,
    ):
        obs = torch.cat((current_arm_pos, target_position, target_orientation), dim=-1)
        output = self.net(obs)
        return output

    def restore(self, chickpoint):
        self.net.load_state_dict(torch.load(chickpoint))

    def save(self, file_path):
        torch.save(self.net.state_dict(), file_path)

    def train(self, buffer: Buffer, train_epoch: int, mini_batch_size):
        for epoch in range(train_epoch):
            current_arm_pos, target_arm_pos, target_pos, target_rot = buffer.sample(
                mini_batch_size
            )
            net_out = self.forward(current_arm_pos, target_pos, target_rot)
            l = self.loss(net_out, target_arm_pos)
            self.trainer.zero_grad()
            l.backward()
            self.trainer.step()
        l = self.loss(
            self.forward(current_arm_pos, target_pos, target_rot), target_arm_pos
        )
        print("current lr: ", self.trainer.param_groups[0]["lr"])
        self.save("frankacontroller.params")
        print(f"epoch {epoch+1}, loss {l:f}")


if __name__ == "__main__":
    controller = FKNet()
    print(controller.parameters)
