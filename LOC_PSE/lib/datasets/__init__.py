import torch
import torch.nn as nn
from core.config import config

def collate_fn_train(batch):
    batch_word_vectors = [b['word_vectors'].transpose(0,1) for b in batch]
    batch_txt_mask = [b['txt_mask'].transpose(0,1) for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_word_label = [b['word_label'].transpose(0,1) for b in batch]
    batch_word_mask = [b['word_mask'].transpose(0,1) for b in batch]
    batch_scores = [b['scores'] for b in batch]
    batch_gt_pairs = [b['gt_pairs'].unsqueeze(0) for b in batch]

    batch_anno_idxs, batch_duration = [], []
    for b in batch:
        batch_anno_idxs.extend(b['anno_idx'])
        batch_duration.extend(b['duration'])

    batch_word_vectors = nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True).transpose(1,2)
    batch_word_vectors = batch_word_vectors.reshape(-1, *batch_word_vectors.shape[2:])
    batch_txt_mask = nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True).transpose(1,2)
    batch_txt_mask = batch_txt_mask.reshape(-1, *batch_txt_mask.shape[2:])
    batch_vis_input = nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float()
    batch_vis_input = batch_vis_input.reshape(-1, *batch_vis_input.shape[2:])
    batch_word_label = nn.utils.rnn.pad_sequence(batch_word_label, batch_first=True).long().transpose(1,2)
    batch_word_label = batch_word_label.reshape(-1, *batch_word_label.shape[2:])
    batch_word_mask = nn.utils.rnn.pad_sequence(batch_word_mask, batch_first=True).float().transpose(1,2)
    batch_word_mask = batch_word_mask.reshape(-1, *batch_word_mask.shape[2:])
    batch_scores = nn.utils.rnn.pad_sequence(batch_scores, batch_first=True).float()
    batch_scores = batch_scores.reshape(-1, *batch_scores.shape[2:])
    batch_gt_pairs = torch.cat(batch_gt_pairs, 0).view(-1)

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_word_vectors': batch_word_vectors,
        'batch_txt_mask': batch_txt_mask,
        'batch_vis_input': batch_vis_input,
        'batch_duration': batch_duration,
        'batch_word_label': batch_word_label,
        'batch_word_mask': batch_word_mask,
        'batch_scores': batch_scores,
        'batch_gt_pairs': batch_gt_pairs
    }

    return batch_data

def collate_fn_test(batch):
    batch_word_vectors = [b['word_vectors'].transpose(0,1) for b in batch]
    batch_txt_mask = [b['txt_mask'].transpose(0,1) for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_word_label = [b['word_label'].transpose(0,1) for b in batch]
    batch_word_mask = [b['word_mask'].transpose(0,1) for b in batch]
    batch_scores = [b['scores'] for b in batch]
    batch_gt_pairs = [b['gt_pairs'].unsqueeze(0) for b in batch]

    batch_anno_idxs, batch_duration = [], []
    for b in batch:
        batch_anno_idxs.append(b['anno_idx'][0])
        batch_duration.append(b['duration'][0])

    batch_word_vectors = nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True).transpose(1,2)
    batch_word_vectors = batch_word_vectors[:,0,:,:]
    batch_txt_mask = nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True).transpose(1,2)
    batch_txt_mask = batch_txt_mask[:,0,:]
    batch_vis_input = nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float()
    batch_vis_input = batch_vis_input[:,0,:,:]
    batch_word_label = nn.utils.rnn.pad_sequence(batch_word_label, batch_first=True).long().transpose(1,2)
    batch_word_label = batch_word_label[:,0,:]
    batch_word_mask = nn.utils.rnn.pad_sequence(batch_word_mask, batch_first=True).float().transpose(1,2)
    batch_word_mask = batch_word_mask[:,0,:]
    batch_scores = nn.utils.rnn.pad_sequence(batch_scores, batch_first=True).float()
    batch_scores = batch_scores[:,0,:]
    batch_gt_pairs = torch.cat(batch_gt_pairs, 0)[:,0]

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_word_vectors': batch_word_vectors,
        'batch_txt_mask': batch_txt_mask,
        'batch_vis_input': batch_vis_input,
        'batch_duration': batch_duration,
        'batch_word_label': batch_word_label,
        'batch_word_mask': batch_word_mask,
        'batch_scores': batch_scores,
        'batch_gt_pairs': batch_gt_pairs
    }

    return batch_data

def average_to_fixed_length(visual_input):
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input

from datasets.activitynet import ActivityNet
from datasets.charades import Charades
