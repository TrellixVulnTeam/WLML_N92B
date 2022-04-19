import torch
from torch import nn

class FrameAvgPool(nn.Module):

    def __init__(self, cfg):
        super(FrameAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_size = cfg.HIDDEN_SIZE
        kernel_size = cfg.KERNEL_SIZE
        stride = cfg.STRIDE
        conv_num = cfg.CONV_NUM
        modules = []
        for i in range(conv_num):
            modules.append(torch.nn.Conv1d(input_size, hidden_size, 3, stride=1, padding=1))
            modules.append(torch.nn.ReLU(inplace=True))
            input_size = hidden_size
            if i < conv_num-1:
                modules.append(nn.AvgPool1d(2, 2))
        if conv_num > 0:
            modules.append(nn.AvgPool1d(kernel_size//(2**(conv_num-1)), stride//(2**(conv_num-1)), kernel_size//(2**conv_num)))
        else:
            modules.append(nn.AvgPool1d(kernel_size, stride, kernel_size//2))

        self.final_mlp_conv = nn.Sequential(*modules)

    def forward(self, visual_input):
        vis_h = self.final_mlp_conv(visual_input)
        return vis_h

class FrameMaxPool(nn.Module):

    def __init__(self, input_size, hidden_size, stride):
        super(FrameMaxPool, self).__init__()
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.max_pool = nn.MaxPool1d(stride)

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.max_pool(visual_input)
        return vis_h
