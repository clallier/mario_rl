from torch import nn

from src.Common.conv_calc import debug_get_conv_out


class AgentNN(nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False, device='mps'):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=6, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
        )

        conv_out_size = debug_get_conv_out(self.conv, input_shape)
        # self.network = nn.Sequential(
        #     self.conv,
        #     nn.Flatten(),
        #     nn.Linear(conv_out_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, n_actions)
        # )
        self.network = nn.Sequential(
            self.conv,
            nn.Flatten(),
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.ReLU()
        )

        if freeze:
            self.freeze()

        self.device = device
        self.to(device)
        self.conv.to(device)
        self.network.to(device)

    def freeze(self):
        for p in self.network.parameters():
            p.requires_grad = False

    def forward(self, x):
        # print("### Hey from forward :)")
        # print("x size: ", x.size())
        # print("x type: ", x.dtype, "x device: ", x.device)
        # print("self device: ", self.device)
        return self.network(x)
