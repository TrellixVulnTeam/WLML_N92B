import torch
from torch import nn
import torch.nn.functional as F
from models.map_modules import get_padded_mask_and_weight

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_sizes = cfg.HIDDEN_SIZES
        kernel_sizes = cfg.KERNEL_SIZES
        strides = cfg.STRIDES
        paddings = cfg.PADDINGS
        dilations = cfg.DILATIONS
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size]+hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv1d(channel_sizes[i], channel_sizes[i+1], k, s, p, d))

        self.rnn = nn.LSTM(channel_sizes[-1], channel_sizes[-1], 1, bidirectional=True)

    def forward(self, textual_input, textual_mask, vis_h):
        print(textual_input, textual_mask)
        for i, pred in enumerate(self.convs):
            x = F.relu(pred(x))
        x = x.permute(2,0,1)
        h0 = x.new_zeros(2, x.size(1), x.size(2))
        c0 = x.new_zeros(2, x.size(1), x.size(2))
        self.rnn.flatten_parameters() 
        output, (hn, cn) = self.rnn(x, (h0, c0))
        output = output.permute(1,2,0)
        out_forword, out_backword = torch.split(output, x.size(2), dim=1)

        return out_forword, out_backword



