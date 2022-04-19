import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from .modeling import BertPredictionHeadTransform
from .visual_linguistic_bert import VisualLinguisticBert
import numpy as np
from scipy.signal import savgol_filter

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class TLocVLBERT(nn.Module):
    def __init__(self, config):

        super(TLocVLBERT, self).__init__()

        self.config = config

        language_pretrained_model_path = None
        if config.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.BERT_PRETRAINED,
                                                                      config.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.BERT_MODEL_NAME):
            weight_path = os.path.join(config.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        iou_mask_map = torch.zeros(32,33).float()
        for i in range(0,32,1):
            iou_mask_map[i,i+1:min(i+17,33)] = 1.
        for i in range(0,32-16,2):
            iou_mask_map[i,range(18+i,33,2)] = 1.
        torch.set_printoptions(threshold=10000)
        # print(iou_mask_map)
        self.register_buffer('iou_mask_map', iou_mask_map)

        self.vlbert = VisualLinguisticBert(config,
                                         language_pretrained_model_path=language_pretrained_model_path)

        dim = config.hidden_size
        if config.CLASSIFIER_TYPE == "2fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, config.vocab_size)
            )
            self.final_mlp_2 = torch.nn.Sequential(
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, dim*3),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
            )
            self.final_mlp_3 = torch.nn.Sequential(
                torch.nn.Linear(dim*3, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 3)
            )

        else:
            raise ValueError("Not support classifier type: {}!".format(config.CLASSIFIER_TYPE))

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_2.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_3.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def fix_params(self):
        pass

    def get_moments(self, video_att):
        out_moments = []
        is_single = []

        scores = video_att.cpu().detach().numpy()
        # scores = savgol_filter(video_att, 9, 1, mode= 'nearest')
        cumsum = np.cumsum(scores, axis=1)
        cumsum[:,-1] = 0
        scores[:,-1] = 0

        max_n = 0
        for score, cs in zip(scores, cumsum):
            sorted_indexs = []
            threshold = score.mean()
            temp = score > threshold
            s = 0
            e = 0
            pre = False
            for i in range(temp.shape[0]):
                if temp[i]:
                    if pre == False:
                        s = i
                else:
                    if pre == True:
                        e = i
                        sorted_indexs.append([s, e, (cs[e-1]-cs[s-1])/(e-s)])
                pre = temp[i]

            # sorted_indexs = sorted(sorted_indexs, key=lambda x: -x[2])
            out_moments.append(sorted_indexs)
            if len(sorted_indexs) == 1:
                is_single.append(1.)
            elif len(sorted_indexs) > 1:
                is_single.append(0.)
            else:
                print('error: peudo moment')
                exit()
            if len(sorted_indexs) > max_n:
                max_n = len(sorted_indexs)

        for i in range(len(out_moments)):
            if len(out_moments[i]) < max_n:
                out_moments[i].extend([[0, 0, 0]]*(max_n-len(out_moments[i])))

        return torch.from_numpy(np.array(out_moments,dtype=np.float32)).cuda(), torch.from_numpy(np.array(is_single,dtype=np.float32)).cuda()


    def forward(self, text_input_feats, text_mask, word_mask, object_visual_feats, video_score, video_label):
        ###########################################

        if self.config.NO_GROUNDING:
            text_visual_embeddings = object_visual_feats.new_zeros([text_input_feats.shape[0],text_input_feats.shape[1]+1,object_visual_feats.shape[2]])
        else:
            text_visual_embeddings = object_visual_feats.mean(1, keepdim=True).repeat(1,text_input_feats.shape[1]+1,1)

        object_linguistic_embeddings = None
        feat_tmp = object_visual_feats.mean(1, keepdim=True)
        object_visual_feats = torch.cat((feat_tmp, object_visual_feats, feat_tmp), 1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_object, hc = self.vlbert(text_input_feats,
                                                                  text_visual_embeddings,
                                                                  text_mask,
                                                                  word_mask,
                                                                  object_visual_feats,
                                                                  object_linguistic_embeddings,
                                                                  output_all_encoded_layers=False)

        logits_text = self.final_mlp(hidden_states_text)

        iou_mask_map = self.iou_mask_map.clone().detach()

        hidden_s, hidden_e, hidden_c = hidden_states_object, hidden_states_object, hidden_states_object


        T = hidden_states_object.size(1) - 1
        s_idx = torch.arange(T, device=hidden_states_object.device)
        e_idx = torch.arange(T+1, device=hidden_states_object.device)
        c_point = hidden_c[:,(0.5*(s_idx[:,None] + e_idx[None,:])).long().flatten(),:].view(hidden_c.size(0),T,T+1,hidden_c.size(-1))
        s_c_e_points = torch.cat((hidden_s[:,:-1,None,:].repeat(1,1,T+1,1), c_point, hidden_e[:,None,:,:].repeat(1,T,1,1)), -1)
        logits_iou = self.final_mlp_3(s_c_e_points).permute(0,3,1,2).contiguous()

        if self.training:
            M = int(video_label.shape[0] / video_label.sum().item())
            video_score = video_score.view(-1,M,video_score.size(1))
            video_score = video_score[:,0,:]
            logits_text = logits_text.view(-1,M,*logits_text.shape[1:])
            logits_text = logits_text[:,0,:,:]

        gt_times, is_single = self.get_moments(video_score)

        return logits_text, logits_iou, iou_mask_map, gt_times, is_single
