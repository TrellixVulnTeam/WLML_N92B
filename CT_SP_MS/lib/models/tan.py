import torch
from torch import nn
import torch.nn.functional as F
from core.config import config
import models.frame_modules as frame_modules
import models.bert_modules as bert_modules
import numpy as np
from scipy.signal import savgol_filter

class WLML(nn.Module):
    def __init__(self):
        super(WLML, self).__init__()

        self.pre_smooth = config.WLML.PRE_SMOOTH

        self.frame_layer_2 = getattr(frame_modules, config.WLML.FRAME_MODULE.NAME)(config.WLML.FRAME_MODULE_2.PARAMS)
        self.bert_layer_2 = getattr(bert_modules, config.WLML.VLBERT_MODULE_2.NAME)(config.WLML.VLBERT_MODULE_2.PARAMS)

        self.frame_layer_3 = getattr(frame_modules, config.WLML.FRAME_MODULE.NAME)(config.WLML.FRAME_MODULE_3.PARAMS)
        self.bert_layer_3 = getattr(bert_modules, config.WLML.VLBERT_MODULE_3.NAME)(config.WLML.VLBERT_MODULE_3.PARAMS)

    def forward(self, textual_input, textual_mask, word_mask, visual_input):

        vis_h_2 = self.frame_layer_2(visual_input.transpose(1, 2))
        vis_h_2 = vis_h_2.transpose(1, 2)
        logits_text_2, _, epis_concepts_2, epis_rewards_2, video_att_2, video_score_2 = self.bert_layer_2(textual_input, textual_mask, word_mask, vis_h_2)
        score_2 = video_att_2 * video_score_2

        vis_h_3 = self.frame_layer_3(visual_input.transpose(1, 2))
        vis_h_3 = vis_h_3.transpose(1, 2)
        logits_text_3, _, epis_concepts_3, epis_rewards_3, video_att_3, video_score_3 = self.bert_layer_3(textual_input, textual_mask, word_mask, vis_h_3)
        score_3 = video_att_3 * video_score_3
        
        xp = list(range(0,33,2))
        x_new = list(range(0,33))
        score_2_iterp = np.array([np.interp(x_new, xp, s) for s in score_2.cpu().detach().numpy()])
        score_2_iterp = savgol_filter(score_2_iterp, 9, 1, mode= 'nearest') if self.pre_smooth else score_2_iterp * score_2_iterp
        score_3_iterp = score_3.cpu().detach().numpy()
        score_3_iterp = savgol_filter(score_3_iterp, 9, 1, mode= 'nearest') if self.pre_smooth else score_3_iterp * score_3_iterp

        merge_scores = savgol_filter(np.sqrt(score_2_iterp*score_3_iterp), 9, 1, mode= 'nearest')
        merge_scores = torch.from_numpy(merge_scores).float().cuda()
        merge_scores.requires_grad = False

        return merge_scores, logits_text_2, epis_concepts_2, epis_rewards_2, score_2, logits_text_3, epis_concepts_3, epis_rewards_3, score_3



    def extract_features(self, textual_input, textual_mask, visual_input):
        pass
