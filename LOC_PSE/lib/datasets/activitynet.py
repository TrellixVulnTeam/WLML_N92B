""" Dataset loader for the ActivityNet Captions dataset """
import os
import json
from collections import OrderedDict
import numpy as np
import random
from random import choice

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from core.eval import iou
from core.config import config

from pattern.en import lemma
try:
    lemma('a')
except:
    pass

class ActivityNet(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(ActivityNet, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        # self.itos = []
        # self.ston = OrderedDict()

        with open(os.path.join(self.data_dir, 'words_vocab_activitynet.json'), 'r') as f:
            tmp = json.load(f)
            self.itos = tmp['words']
        self.stoi = OrderedDict()
        for i, w in enumerate(self.itos):
            self.stoi[w] = i
        print(len(self.stoi))

        with open(os.path.join(self.data_dir, 'concepts_activitynet.json'),'r') as f:
            temp = json.load(f)
            self.concepts = temp['concepts']
        self.concepts = list(set([lemma(w) for w in self.concepts]))
        print('number of concepts:', len(self.concepts))

        self.concept_videos = {}
        self.concept_annos = {}

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        with open(os.path.join(self.data_dir, '{}.json'.format(split)),'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        max_sent_len = 0
        for vid, video_anno in annotations.items():
            duration = video_anno['duration']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    sentence = sentence.replace(',',' ').replace('/',' ').replace('\"',' ').replace('-',' ').replace(';',' ').replace('.',' ').replace('&',' ').replace('?',' ').replace('!',' ').replace('(',' ').replace(')',' ').replace("'s",' is').replace("'re",' are').replace("'ve",' have').replace("'m",' am').replace("n't",' not').replace("'",' ').replace(':',' ').replace('[',' ').replace(']',' ').replace('@',' ').lower()
                    
                    concepts = []
                    for w in sentence.split():
                        if lemma(w) in ['he', 'him', 'his', 'himself', 'men', 'gentleman']:
                            w = 'man'
                        elif lemma(w) in ['she', 'her', 'herself', 'women', 'lady']:
                            w = 'woman'
                        if lemma(w) in self.concepts:
                            concepts.append(lemma(w))
                            if lemma(w) not in self.concept_videos.keys():
                                self.concept_videos[lemma(w)] = [vid]
                            else:
                                self.concept_videos[lemma(w)].append(vid)
                            if lemma(w) not in self.concept_annos.keys():
                                self.concept_annos[lemma(w)] = [len(anno_pairs)]
                            else:
                                self.concept_annos[lemma(w)].append(len(anno_pairs))
                    concepts = list(set(concepts))

                    anno_pairs.append(
                        {
                            'video': vid,
                            'duration': duration,
                            'times':[max(timestamp[0],0),min(timestamp[1],duration)],
                            'description':sentence,
                            'concepts':concepts,
                        }
                    )
                    if len(sentence.split()) > max_sent_len:
                        max_sent_len = len(sentence.split())

                    # for w in sentence.split():
                    #     if w.lower() not in self.ston.keys():
                    #         self.itos.append(w.lower())
                    #         self.ston[w.lower()] = 1
                    #     else:
                    #         self.ston[w.lower()] += 1

        self.annotations = anno_pairs
        print('max_sent_len:', max_sent_len)

        # self.itos.extend(['UNK'])
        # self.ston['UNK'] = 0
        # print('total words:', len(self.itos))
        # # ston = sorted(self.ston.items(),key = lambda x:x[1],reverse = True)
        # if len(self.itos) > 10000:
        #     with open('words_vocab.json', 'w') as f:
        #         json.dump({'words':self.itos}, f)
        #         # print(self.itos)

        for c in self.concept_videos.keys():
            self.concept_videos[c] = list(set(self.concept_videos[c]))

        self.video_sents = {}
        for anno in self.annotations:
            vid = anno['video']
            if len(anno['concepts']) > 0:
                if vid not in self.video_sents.keys():
                    self.video_sents[vid] = [anno['concepts']]
                else:
                    self.video_sents[vid].append(anno['concepts'])

        self.scores = {}
        with h5py.File('../CT_SP_MS/%s_scores_32L_%s_fusion.h5'%(config.DATASET.NAME, self.split), 'r') as f:
            for i in f.keys():
                self.scores[i] = torch.from_numpy(f[i][:]).float()


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
        if len(concepts) > 0:
            concept = choice(concepts)
        else:
            concept = None

        select_videos = [video_id]
        k = 0
        flag = random.random()<0.5
        while True:
            if flag and concept is not None and len(self.concept_videos[concept]) > 1 and k < 5:
                concept_v = self.word_embedding(torch.tensor([self.vocab.stoi.get(c, 400000) for c in concepts], dtype=torch.long))
                index_neg = choice(self.concept_annos[concept])
                video_neg = self.annotations[index_neg]['video']
                sents_neg = self.video_sents[video_neg]
                sims = [F.cosine_similarity(concept_v[:,None,:], self.word_embedding(torch.tensor([self.vocab.stoi.get(c, 400000) for c in sent], dtype=torch.long))[None,:,:], dim=2).max(axis=1)[0].mean(axis=0).item() for sent in sents_neg]
                if np.max(sims) > 0.8:
                    k += 1
                    continue
            else:
                index_neg = (index + random.randint(1, len(self.annotations)-1)) % len(self.annotations)
                if concept is not None:
                    concept_v = self.word_embedding(torch.tensor([self.vocab.stoi.get(c, 400000) for c in concepts], dtype=torch.long))
                    video_neg = self.annotations[index_neg]['video']
                    sents_neg = self.video_sents[video_neg]
                    sims = [F.cosine_similarity(concept_v[:,None,:], self.word_embedding(torch.tensor([self.vocab.stoi.get(c, 400000) for c in sent], dtype=torch.long))[None,:,:], dim=2).max(axis=1)[0].mean(axis=0).item() for sent in sents_neg]
                    if np.max(sims) > 0.8:
                        k += 1
                        continue
            video_id_neg = self.annotations[index_neg]['video']
            if video_id_neg not in select_videos:
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

                if len(select_videos) == 4:
                    break
                else:
                    if len(concepts) > 1:
                        concepts.remove(concept)
                        concept = choice(concepts)
                    else:
                        concept = None
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
            'gt_pairs': torch.from_numpy(np.array([1., 0., 0., 0.], dtype=np.float32)),
            'scores': torch.stack(score_list)
        }

        return item


    def get_visual_sentence_pair(self, index):

        video_id = self.annotations[index]['video']
        sentence = self.annotations[index]['description']
        duration = self.annotations[index]['duration']

        word_label = [self.stoi.get(w.lower(), 10532) for w in sentence.split()]
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

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        visual_input, visual_mask = self.get_video_features(video_id)
        if config.DATASET.VIS_INPUT_TYPE == 'c3d' or (config.DATASET.VIS_INPUT_TYPE == 'resnet152' and visual_input.shape[0] < 256):
            visual_input = average_to_fixed_length(visual_input)

        return duration, word_label, word_mask, word_vectors, visual_input

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        if config.DATASET.VIS_INPUT_TYPE == 'c3d':
            with h5py.File(os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
                features = torch.from_numpy(f[vid]['c3d_features'][:])
        elif config.DATASET.VIS_INPUT_TYPE == 'resnet152':
            with h5py.File(os.path.join(self.data_dir, 'resnet152_features_activitynet_L256.hdf5'), 'r') as f:
                features = torch.from_numpy(f[vid][:])
        else:
            print('feature type error')
            exit()
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask