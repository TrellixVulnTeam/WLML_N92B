import torch
from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.bert_modules as bert_modules

class WLML(nn.Module):
    def __init__(self):
        super(WLML, self).__init__()

        self.frame_layer = getattr(frame_modules, config.WLML.FRAME_MODULE.NAME)(config.WLML.FRAME_MODULE.PARAMS)
        self.bert_layer = getattr(bert_modules, config.WLML.VLBERT_MODULE.NAME)(config.WLML.VLBERT_MODULE.PARAMS)

    def forward(self, textual_input, textual_mask, word_mask, visual_input):

        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        vis_h = vis_h.transpose(1, 2)
        logits_text, logits_iou, iou_mask_map, epis_concepts, epis_rewards, positive_video_att, video_score = self.bert_layer(textual_input, textual_mask, word_mask, vis_h)

        return logits_text, logits_iou, iou_mask_map, epis_concepts, epis_rewards, positive_video_att, video_score

    def extract_features(self, textual_input, textual_mask, visual_input):
        pass
