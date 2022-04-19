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

import matplotlib.pyplot as plt
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
        # map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']
        word_label = sample['batch_word_label'].cuda()
        word_mask = sample['batch_word_mask'].cuda()
        video_label = sample['batch_gt_times'].cuda()
        concept_label = sample['batch_concept_label'].cuda()

        logits_text, logits_iou, iou_mask_map, epis_concepts, epis_rewards, video_att, video_score = model(textual_input, textual_mask, word_mask, visual_input)
        loss_value, video_att, regress = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, model.training, logits_text, logits_iou, iou_mask_map, word_label, word_mask, epis_concepts, epis_rewards, video_att, video_label, concept_label)

        sorted_times = None if model.training else get_proposal_results(anno_idxs, video_att*video_score, duration)

        if not model.training:
            torch.cuda.empty_cache()

        return loss_value, sorted_times

    def get_proposal_results(anno_idxs, video_att, durations):
        out_sorted_times = []
        N = video_att.shape[0]

        video_att = video_att.cpu().detach().numpy()
        xp = list(range(0,33,2))
        x_new = list(range(0,33))

        scores = savgol_filter(video_att, config.TEST.SAVGOL_K, 1, mode= 'nearest')
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
                        sorted_indexs.append([s, e])
                pre = temp[i]

            sorted_indexs = sorted(sorted_indexs, key=lambda x: -(cs[x[1]-1]-cs[x[0]-1])/(x[1]-x[0]))
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

    CHECKPOINT = config.MODEL.CHECKPOINT
    print(CHECKPOINT)
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = getattr(models, config.MODEL.NAME)()
    model_checkpoint = torch.load(CHECKPOINT)
    
    model.load_state_dict(model_checkpoint)
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
