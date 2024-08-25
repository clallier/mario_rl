from torch import nn


class AgentNN(nn.Module):
    def __init__(self, nn_hidden_size, nn_output_size, freeze=False, device="cpu"):
        super().__init__()
        # v1
        # self.conv = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )

        # # v2
        # self.conv = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 16, kernel_size=6, stride=4, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),
        #     nn.ReLU(),
        # )

        # v3
        self.conv = nn.Sequential(
            # input: envs, channels, time, w,  h
            # input: 4,    1,        4,    30, 30
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3, 3),
                stride=(3, 2, 2),
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, nn_hidden_size),
            nn.ReLU(),
        )

        # conv_out_size = debug_get_conv_out(self.conv, input_shape)

        # 1
        # self.network = nn.Sequential(
        #     self.conv,
        #     nn.Flatten(),
        #     nn.Linear(conv_out_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, n_actions)
        # )

        # 2
        # self.network = nn.Sequential(
        #     self.conv,
        #     nn.Flatten(),
        #     nn.Linear(conv_out_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, n_actions),
        #     nn.ReLU()
        # )

        # 3
        hidden_size_2 = nn_hidden_size // 2
        hidden_size_4 = hidden_size_2 // 2
        self.network = nn.Sequential(
            self.conv,
            nn.Linear(nn_hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_4),
            nn.ReLU(),
            nn.Linear(hidden_size_4, nn_output_size),
            nn.ReLU(),
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
