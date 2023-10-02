import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOV3(nn.Module):
    def __init__(self, B=3, C=80):
        super(YOLOV3, self).__init__()
        in_channels = 3
        out_channels = 256
        yolo_channels = (5 + C) * B
        self.features = self.make_layers(in_channels=in_channels, out_channels=out_channels)

        # yolo2
        self.yolo_layer2_neck = nn.Sequential(
            nn.Conv2d(out_channels, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
        )
        self.yolo_layer2_head = nn.Sequential(
            nn.Conv2d(512, yolo_channels, kernel_size=1, stride=1, padding=0)
        )

        # yolo1
        self.yolo_layer1_neck1 = nn.Sequential(
            nn.Conv2d(out_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.yolo_layer1_neck2 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )
        self.yolo_layer1_head = nn.Sequential(
            nn.Conv2d(256, yolo_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        route = None
        for i, feature in enumerate(self.features):
            if isinstance(feature[0], nn.MaxPool2d) and feature[0].stride == 1:
                x = F.pad(x, [0, 1, 0, 1], mode='constant', value=0) # same as paper
            x = feature(x)
            if i == 8:
                route = x

        # yolo2
        yolo_layer2_neck = self.yolo_layer2_neck(x)
        yolo_layer2_head = self.yolo_layer2_head(yolo_layer2_neck)

        # yolo1
        yolo_layer1_neck = self.yolo_layer1_neck2(torch.cat([self.yolo_layer1_neck1(x), route], dim=1))
        yolo_layer1_head = self.yolo_layer1_head(yolo_layer1_neck)

        return [yolo_layer1_head, yolo_layer2_head]

    def make_layers(self, in_channels=3, out_channels=256):
        # conv: out_channels, kernel_size, stride, batchnorm, activate
        # maxpool: kernel_size stride
        params = [
            [16, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [32, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [64, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [128, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [256, 3, 1, True, 'leaky'], # 8
            ['M', 2, 2],
            [512, 3, 1, True, 'leaky'],
            ['M', 2, 1], # same as paper
            [1024, 3, 1, True, 'leaky'],
            # additinal
            [out_channels, 1, 1, True, 'leaky']
        ]

        module_list = nn.ModuleList()
        for i, v in enumerate(params):
            modules = nn.Sequential()
            if v[0] == 'M':
                modules.add_module(f'maxpool_{i}', nn.MaxPool2d(kernel_size=v[1], stride=v[2], padding=int((v[1] - 1) // 2)))
            else:
                modules.add_module(
                    f'conv_{i}',
                    nn.Conv2d(
                        in_channels,
                        v[0],
                        kernel_size=v[1],
                        stride=v[2],
                        padding=(v[1] - 1) // 2,
                        bias=not v[3]
                    )
                )
                if v[3]:
                    modules.add_module(f'bn_{i}', nn.BatchNorm2d(v[0]))
                modules.add_module(f'act_{i}', nn.LeakyReLU(0.1) if v[4] == 'leaky' else nn.ReLU())
                in_channels = v[0]
            module_list.append(modules)
        return module_list