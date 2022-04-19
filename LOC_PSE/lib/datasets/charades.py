from __future__ import division
import os
import csv
from collections import OrderedDict
import json
import random
import _pickle as pickle

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from random import choice

from . import average_to_fixed_length
from core.eval import iou
from core.config import config

from pattern.en import lemma
try:
    lemma('a')
except:
    pass

class Charades(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(Charades, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split
        self.videos = []
        self.video_sentence = {}
        
        with open(os.path.join(self.data_dir, './concepts_charades.json'),'r') as f:
            temp = json.load(f)
            self.concepts = temp['concepts']

        self.concepts = list(set([lemma(w) for w in self.concepts]))
        print('number of concepts:', len(self.concepts))

        self.concept_videos = {}
        self.concept_annos = {}

        with open(os.path.join(self.data_dir,'words_vocab_charades.json'), 'r') as f:
            tmp = json.load(f)
            self.itos = tmp['words']
        self.stoi = OrderedDict()
        for i, w in enumerate(self.itos):
            self.stoi[w] = i
        print(len(self.stoi))

        self.durations = {}
        with open(os.path.join(self.data_dir, 'Charades_v1_{}.csv'.format(split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])

        anno_file = open(os.path.join(self.data_dir, "charades_sta_{}.txt".format(self.split)),'r')
        annotations = []
        max_sent_len = 0
        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            s_time = float(s_time)
            e_time = min(float(e_time), self.durations[vid])
            if s_time < e_time:
                sent = sent.replace(',',' ').replace('/',' ').replace('-',' ').replace(';',' ').replace('.',' ').replace('#',' ').replace('!',' ').replace("'s",' ').replace("'re",' are').replace("'ve",' have').replace('(',' ').replace(')',' ').replace("'",' ')

                word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sent.split()], dtype=torch.long)
                word_vectors = self.word_embedding(word_idxs)

                concepts = []
                for w in sent.split():
                    if lemma(w) == 'tv':
                        w = 'television'
                        # print(w)
                    if lemma(w) in self.concepts:
                        concepts.append(lemma(w))
                        if lemma(w) not in self.concept_videos.keys():
                            self.concept_videos[lemma(w)] = [vid]
                        else:
                            self.concept_videos[lemma(w)].append(vid)
                        if lemma(w) not in self.concept_annos.keys():
                            self.concept_annos[lemma(w)] = [len(annotations)]
                        else:
                            self.concept_annos[lemma(w)].append(len(annotations))
                concepts = list(set(concepts))

                annotations.append({'video':vid, 'times':[s_time, e_time], 'description': sent, 'duration': self.durations[vid], 'word_vectors': word_vectors, 'concepts':concepts})

                if vid not in self.videos:
                    self.videos.append(vid)
                    
                if len(sent.split()) > max_sent_len:
                    max_sent_len = len(sent.split())

        anno_file.close()
        self.annotations = annotations
        print('max_sent_len', max_sent_len)
        print('len of annotations', len(annotations))

        for c in self.concept_videos.keys():
            self.concept_videos[c] = list(set(self.concept_videos[c]))

        self.scores = {}
        with h5py.File('../CT_SP_MS/%s_scores_32L_%s_fusion.h5'%(config.DATASET.NAME, self.split), 'r') as f:
            for i in f.keys():
                self.scores[i] = torch.from_numpy(f[i][:]).float()

        hf = h5py.File(os.path.join(self.data_dir,'%s_video_sentences.h5'%self.split), 'r')
        for key in hf.keys():
            self.video_sentence[key] = torch.from_numpy(hf[key][:]).float()
        hf.close()

        hf = h5py.File(os.path.join(self.data_dir,'%s_annotation_sentences.h5'%self.split), 'r')
        self.annotation_sentences = torch.from_numpy(hf['data'][:]).float()
        print('annotation_sentences shape', self.annotation_sentences.shape)
        hf.close()
    

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        duration, word_label, word_mask, word_vectors, visual_input = self.get_visual_sentence_pair(index)

        visual_input_list = [visual_input]
        anno_idx_list = [index]
        word_vectors_list = [word_vectors]
        txt_mask_list = [torch.ones(word_vectors.shape[0],)]
        word_label_list = [word_label]
        word_mask_list = [word_mask]
        duration_list = [duration]
        score_list = [self.scores['anno-'+str(index)]]

        concepts = self.annotations[index]['concepts']
        concept = choice(concepts)

        select_videos = [video_id]
        sents = self.annotation_sentences[index:index+1]
        k = 0
        flag = random.random()<0.5
        while True:
            if flag and len(self.concept_videos[concept]) > 3 and k < 3:
                index_neg = choice(self.concept_annos[concept])
            else:
                index_neg = (index + random.randint(1, len(self.annotations)-1)) % len(self.annotations)
            video_id_neg = self.annotations[index_neg]['video']
            if video_id_neg not in select_videos:
                sents_neg = self.video_sentence[video_id_neg]
                sim = F.cosine_similarity(sents[:,None,:], sents_neg[None,:,:], dim=2).view(-1).max()
                if sim < 0.9:
                    duration_neg, word_label_neg, word_mask_neg, word_vectors_neg, visual_input_neg = self.get_visual_sentence_pair(index_neg)

                    visual_input_list.extend([visual_input_neg])
                    anno_idx_list.extend([index_neg])
                    word_vectors_list.extend([word_vectors])
                    txt_mask_list.extend([torch.ones(word_vectors.shape[0],)])
                    word_label_list.extend([word_label])
                    word_mask_list.extend([word_mask])
                    duration_list.extend([duration_neg])
                    score_list.extend([self.scores['anno-'+str(index_neg)]])

                    select_videos.append(video_id_neg)

                    k = 0

                    if len(select_videos) < 3:
                        if len(concepts) > 1:
                            concepts.remove(concept)
                            concept = choice(concepts)
                        else:
                            flag = False
                    elif len(select_videos) >= 3 and len(select_videos) < 5:
                    	flag = False
                    else:
                    	break
                else:
                	k += 1
            else:
            	k += 1

        item = {
            'visual_input': torch.stack(visual_input_list),
            'anno_idx': anno_idx_list,
            'word_vectors': nn.utils.rnn.pad_sequence(word_vectors_list, batch_first=True),
            'txt_mask': nn.utils.rnn.pad_sequence(txt_mask_list, batch_first=True),
            'word_label': nn.utils.rnn.pad_sequence(word_label_list, batch_first=True),
            'word_mask': nn.utils.rnn.pad_sequence(word_mask_list, batch_first=True),
            'duration': duration_list,
            'gt_pairs': torch.from_numpy(np.array([1., 0., 0., 0., 0.], dtype=np.float32)),
            'scores': torch.stack(score_list)
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_visual_sentence_pair(self, index):
        video_id = self.annotations[index]['video']
        description = self.annotations[index]['description']
        duration = self.durations[video_id]

        word_label = [self.stoi.get(w.lower(), 1065) for w in description.split()]
        range_i = range(len(word_label))
        word_mask = [1. if np.random.uniform(0,1)<0.15 else 0. for _ in range_i]
        if np.sum(word_mask) == 0.:
            mask_i = np.random.choice(range_i)
            word_mask[mask_i] = 1.
        if np.sum(word_mask) == len(word_mask):
            unmask_i = np.random.choice(range_i)
            word_mask[unmask_i] = 0.
        word_label = torch.tensor(word_label, dtype=torch.long)
        word_mask = torch.tensor(word_mask, dtype=torch.float)

        word_vectors = self.annotations[index]['word_vectors']

        visual_input, _ = self.get_video_features(video_id)
        visual_input = average_to_fixed_length(visual_input)

        return duration, word_label, word_mask, word_vectors, visual_input


    def get_video_features(self, vid):
        if config.DATASET.VIS_INPUT_TYPE == 'c3d':
            features = torch.load(os.path.join(self.data_dir, 'C3D_unit16_overlap0.5_merged/{}.pt'.format(vid))).float()
        elif config.DATASET.VIS_INPUT_TYPE == 'vgg':
            hdf5_file = h5py.File(os.path.join(self.data_dir, 'vgg_rgb_features.hdf5'), 'r')
            features = torch.from_numpy(hdf5_file[vid][:]).float()
        else:
            print('feature type error')
            exit()
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

    def get_target_weights(self):
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        pos_count = [0 for _ in range(num_clips)]
        total_count = [0 for _ in range(num_clips)]
        pos_weight = torch.zeros(num_clips, num_clips)
        for anno in self.annotations:
            video_id = anno['video']
            gt_s_time, gt_e_time = anno['times']
            duration = self.durations[video_id]
            s_times = torch.arange(0, num_clips).float() * duration / num_clips
            e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
            overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                        e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                           torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)
            overlaps[overlaps >= 0.5] = 1
            overlaps[overlaps < 0.5] = 0
            for i in range(num_clips):
                s_idxs = list(range(0, num_clips - i))
                e_idxs = [s_idx + i for s_idx in s_idxs]
                pos_count[i] += sum(overlaps[s_idxs, e_idxs])
                total_count[i] += len(s_idxs)

        for i in range(num_clips):
            s_idxs = list(range(0, num_clips - i))
            e_idxs = [s_idx + i for s_idx in s_idxs]
            # anchor weights
            # pos_weight[s_idxs,e_idxs] = pos_count[i]/total_count[i]
            # global weights
            pos_weight[s_idxs, e_idxs] = sum(pos_count) / sum(total_count)


        return pos_weight
