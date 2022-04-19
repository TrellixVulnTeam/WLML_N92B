import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from .modeling import BertPredictionHeadTransform
from .visual_linguistic_bert import VisualLinguisticBert

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

        iou_mask_map = torch.zeros(16,17).float()
        for i in range(0,16,1):
            iou_mask_map[i,i+1:17] = 1.
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
                # torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                # torch.nn.Linear(config.hidden_size, config.CLASSIFIER_HIDDEN_SIZE),
                # torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, config.concept_size)
            )
            self.final_mlp_3 = torch.nn.Sequential(
                torch.nn.Linear(dim*3, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 3)
            )
            self.final_mlp_binary = torch.nn.Sequential(
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 1)
            )
            self.final_mlp_binary_2 = torch.nn.Sequential(
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 1)
            )
        else:
            raise ValueError("Not support classifier type: {}!".format(config.CLASSIFIER_TYPE))

        if not config.PRE_CONV:
            self.final_mlp_conv = torch.nn.Sequential(
                torch.nn.Conv1d(config.visual_size, config.hidden_size, config.kernel_size, stride=1, padding=config.kernel_size//2),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Conv1d(config.hidden_size, config.hidden_size, config.kernel_size, stride=1, padding=config.kernel_size//2),
                torch.nn.ReLU(inplace=True)
            )


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
        for m in self.final_mlp_binary.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_binary_2.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def fix_params(self):
        pass


    def forward(self, text_input_feats, text_mask, word_mask, object_visual_feats):
        ###########################################

        if self.config.NO_GROUNDING:
            text_visual_embeddings = object_visual_feats.new_zeros([text_input_feats.shape[0],text_input_feats.shape[1]+1,object_visual_feats.shape[2]])
        else:
            text_visual_embeddings = object_visual_feats.mean(1, keepdim=True).repeat(1,text_input_feats.shape[1]+1,1)

        object_linguistic_embeddings = None
        feat_tmp = object_visual_feats.mean(1, keepdim=True)
        object_visual_feats_extend = torch.cat((feat_tmp, object_visual_feats, feat_tmp), 1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_object, hc = self.vlbert(text_input_feats,
                                                                  text_visual_embeddings,
                                                                  text_mask,
                                                                  word_mask,
                                                                  object_visual_feats_extend,
                                                                  object_linguistic_embeddings,
                                                                  output_all_encoded_layers=False)

        logits_text = self.final_mlp(hidden_states_text)

        iou_mask_map = self.iou_mask_map.clone().detach()

        video_score = torch.sigmoid(self.final_mlp_binary_2(hidden_states_object))
        video_score = video_score.view(video_score.size(0), video_score.size(1))

        video_att = self.final_mlp_binary(hidden_states_object).view(hidden_states_object.size(0), hidden_states_object.size(1))
        video_att = torch.softmax(video_att, dim=1)
        epis_rewards = torch.sum(video_score * video_att, dim=1)

        if not self.config.PRE_CONV:
            object_visual_feats = object_visual_feats.permute(0,2,1)
            object_visual_feats = self.final_mlp_conv(object_visual_feats)
            object_visual_feats = object_visual_feats.permute(0,2,1)
        video_feat = torch.sum(object_visual_feats * video_score[:,:,None] * video_att[:,:,None], dim=1)
        epis_concepts = torch.sigmoid(self.final_mlp_2(F.normalize(video_feat, dim=-1)))

        return logits_text, None, iou_mask_map, epis_concepts, epis_rewards, video_att, video_score
