import os
import math
import argparse
import pickle as pkl

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import _init_paths
from core.engine import Engine
import datasets
import models
from core.utils import AverageMeter
from core.config import config, update_config
from core.eval import eval_predictions, display_results, iou
import models.loss as loss

# import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import h5py

torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.set_printoptions(precision=2, sci_mode=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Test localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # testing
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--split', default='val', required=True, choices=['train', 'val', 'test'], type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.OUTPUT_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose

def save_scores(scores, data, dataset_name, split):
    results = {}
    for i, d in enumerate(data):
        results[d['video']] = scores[i]
    pkl.dump(results,open(os.path.join(config.RESULT_DIR, dataset_name, '{}_{}_{}.pkl'.format(config.MODEL.NAME,config.DATASET.VIS_INPUT_TYPE,
        split)),'wb'))

if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)

    test_dataset = getattr(datasets, config.DATASET.NAME)(args.split)
    dataloader = DataLoader(test_dataset,
                            batch_size=config.TRAIN.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.WORKERS,
                            pin_memory=False,
                            collate_fn=datasets.collate_fn_test)

    def network(sample):
        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        duration = sample['batch_duration']
        word_label = sample['batch_word_label'].cuda()
        word_mask = sample['batch_word_mask'].cuda()
        video_label = sample['batch_gt_times'].cuda()
        concept_label = sample['batch_concept_label'].cuda()

        merge_scores, logits_text_2, epis_concepts_2, epis_rewards_2, score_2, logits_text_3, epis_concepts_3, epis_rewards_3, score_3 = model(textual_input, textual_mask, word_mask, visual_input)
        loss_value = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, model.training, logits_text_2, word_label, word_mask, epis_concepts_2, epis_rewards_2, score_2, video_label, concept_label)
        loss_value += getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, model.training, logits_text_3, word_label, word_mask, epis_concepts_3, epis_rewards_3, score_3, video_label, concept_label)
        loss_value = loss_value / 2.

        sorted_times = None if model.training else get_proposal_results(anno_idxs, merge_scores, duration)

        if not model.training:
            torch.cuda.empty_cache()

        return loss_value, sorted_times

    def inter_outer(cs, s, e):
        l = max((e - s) // 2, 1)
        outer = cs[min(len(cs)-1, e+l) - 1] - cs[max(0, s-l) - 1]
        inter = cs[e-1] - cs[s-1]
        inter_outer = inter/(e-s) - (outer-inter)/(min(len(cs)-1, e+l) - max(0, s-l) - (e-s))
        return inter_outer

    def get_proposal_results(anno_idxs, video_att, durations):
        out_sorted_times = []

        scores = video_att.cpu().detach().numpy()
        # scores = savgol_filter(video_att, 9, 1, mode= 'nearest')

        file_name = '%s_scores_32L_%s_fusion.h5'%(config.DATASET.NAME, args.split)
        if not os.path.exists(file_name):
            f = h5py.File(file_name, 'w')
        else:
            f = h5py.File(file_name, 'a')
        for anno_idx, score in zip(anno_idxs, scores):
            f.create_dataset('anno-'+str(anno_idx), data=score)
        f.close()

        cumsum = np.cumsum(scores, axis=1)
        cumsum[:,-1] = 0
        scores[:,-1] = 0

        for score, cs, duration in zip(scores, cumsum, durations):
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
                        sorted_indexs.append([s, e, inter_outer(cs, s, e)])
                pre = temp[i]

            # sorted_indexs = sorted(sorted_indexs, key=lambda x: (cs[x[1]-1]-cs[x[0]-1])/(x[0]-x[1]))
            sorted_indexs = sorted(sorted_indexs, key=lambda x: -x[2])
            sorted_indexs = np.array(sorted_indexs).astype(float)
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())

        return out_sorted_times

    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        state['output'] = []
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TRAIN.BATCH_SIZE), ncols=80)

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()
            print()

        annotations = test_dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = eval_predictions(state['sorted_segments_list'], annotations, verbose=True)

        loss_message = '\ntest loss {:.4f}'.format(state['loss_meter'].avg)
        print(loss_message)
        state['loss_meter'].reset()
        test_table = display_results(state['Rank@N,mIoU@M'], state['miou'],
                                          'performance on testing set')
        table_message = '\n'+test_table
        print(table_message)

        # save_scores(state['sorted_segments_list'], annotations, config.DATASET.NAME, args.split)
 
    file_name = '%s_scores_32L_%s_fusion.h5'%(config.DATASET.NAME, args.split)
    if os.path.exists(file_name):  
        os.remove(file_name)       #remove the old score file and generate new score file.

    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = getattr(models, config.MODEL.NAME)()
    model_dict = model.state_dict()

    print(config.MODEL.CHECKPOINT_2)
    model_checkpoint = torch.load(config.MODEL.CHECKPOINT_2)
    diff = {k.replace('bert_layer', 'bert_layer_2').replace('frame_layer', 'frame_layer_2'):v for k,v in model_checkpoint.items() if 'iou_mask_map' not in k}
    model_dict.update(diff)

    print(config.MODEL.CHECKPOINT_3)
    model_checkpoint = torch.load(config.MODEL.CHECKPOINT_3)
    diff = {k.replace('bert_layer', 'bert_layer_3').replace('frame_layer', 'frame_layer_3'):v for k,v in model_checkpoint.items() if 'iou_mask_map' not in k}
    model_dict.update(diff)

    model.load_state_dict(model_dict)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model  = model.to(device)
    model.eval()

    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.test(network,dataloader, args.split)
