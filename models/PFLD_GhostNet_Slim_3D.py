#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import torch
from torch.nn import Module, AvgPool2d, Linear
from models.base_module import Conv_Block, GhostBottleneck


class PFLD_GhostNet_Slim_3D(Module):
    def __init__(self, width_factor=1, input_size=112, landmark_number=98):
        super(PFLD_GhostNet_Slim_3D, self).__init__()

        self.conv1 = Conv_Block(3, int(64 * width_factor), 3, 2, 1)
        self.conv2 = Conv_Block(int(
            64 * width_factor), int(64 * width_factor), 3, 1, 1, group=int(64 * width_factor))

        self.conv3_1 = GhostBottleneck(
            int(64 * width_factor), int(96 * width_factor), int(80 * width_factor), stride=2)
        self.conv3_2 = GhostBottleneck(
            int(80 * width_factor), int(120 * width_factor), int(80 * width_factor), stride=1)
        self.conv3_3 = GhostBottleneck(
            int(80 * width_factor), int(120 * width_factor), int(80 * width_factor), stride=1)

        self.conv4_1 = GhostBottleneck(
            int(80 * width_factor), int(200 * width_factor), int(96 * width_factor), stride=2)
        self.conv4_2 = GhostBottleneck(
            int(96 * width_factor), int(240 * width_factor), int(96 * width_factor), stride=1)
        self.conv4_3 = GhostBottleneck(
            int(96 * width_factor), int(240 * width_factor), int(96 * width_factor), stride=1)

        self.conv5_1 = GhostBottleneck(int(
            96 * width_factor), int(336 * width_factor), int(144 * width_factor), stride=2)
        self.conv5_2 = GhostBottleneck(int(
            144 * width_factor), int(504 * width_factor), int(144 * width_factor), stride=1)
        self.conv5_3 = GhostBottleneck(int(
            144 * width_factor), int(504 * width_factor), int(144 * width_factor), stride=1)
        self.conv5_4 = GhostBottleneck(int(
            144 * width_factor), int(504 * width_factor), int(144 * width_factor), stride=1)

        self.conv6 = GhostBottleneck(int(
            144 * width_factor), int(216 * width_factor), int(16 * width_factor), stride=1)
        self.conv7 = Conv_Block(int(16 * width_factor),
                                int(32 * width_factor), 3, 1, 1)
        self.conv8 = Conv_Block(int(
            32 * width_factor), int(128 * width_factor), input_size // 16, 1, 0, has_bn=False)

        self.avg_pool1 = AvgPool2d(input_size // 2)
        self.avg_pool2 = AvgPool2d(input_size // 4)
        self.avg_pool3 = AvgPool2d(input_size // 8)
        self.avg_pool4 = AvgPool2d(input_size // 16)

        x = torch.ones([1, 3, input_size, input_size])
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x3 = self.avg_pool3(x)
        x3 = x3.view(x3.size(0), -1)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x4 = self.avg_pool4(x)
        x4 = x4.view(x4.size(0), -1)

        x = self.conv6(x)
        x = self.conv7(x)
        x5 = self.conv8(x)
        x5 = x5.view(x5.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3, x4, x5], 1)

        self.fc = Linear(int(multi_scale.shape[1]), landmark_number * 2)
        self.fc_depth = Linear(int(multi_scale.shape[1]), landmark_number)
        self.fc_angle = Linear(int(multi_scale.shape[1]), 3)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            x -= 127.5
            x *= 0.00784313725490196
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x3 = self.avg_pool3(x)
        x3 = x3.view(x3.size(0), -1)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x4 = self.avg_pool4(x)
        x4 = x4.view(x4.size(0), -1)

        x = self.conv6(x)
        x = self.conv7(x)
        x5 = self.conv8(x)
        x5 = x5.view(x5.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3, x4, x5], 1)
        landmarks = self.fc(multi_scale)
        depths = self.fc_depth(multi_scale)
        angles = self.fc_angle(multi_scale)

        if torch.onnx.is_in_onnx_export():
            return landmarks*0.5+0.5, depths*0.5+0.5, angles*57.29578
        return landmarks, depths, angles
