from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import datasets
import models
from core.config import config, update_config
from core.engine import Engine
from core.utils import AverageMeter
from core import eval
from core.utils import create_logger
import models.loss as loss
import math
from scipy.signal import savgol_filter

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
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
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n'+pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    train_dataset = getattr(datasets, dataset_name)('train')
    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name)('val')
    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.MODEL.CHECKPOINT_2 and config.MODEL.CHECKPOINT_3 and config.TRAIN.CONTINUE:
        model_dict = model.state_dict()

        model_checkpoint = torch.load(config.MODEL.CHECKPOINT_2)
        diff = {k.replace('bert_layer', 'bert_layer_2'):v for k,v in model_checkpoint.items() if 'iou_mask_map' not in k}
        model_dict.update(diff)

        model_checkpoint = torch.load(config.MODEL.CHECKPOINT_3)
        diff = {k.replace('bert_layer', 'bert_layer_3'):v for k,v in model_checkpoint.items() if 'iou_mask_map' not in k}
        model_dict.update(diff)

        model.load_state_dict(model_dict)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(),lr=config.TRAIN.LR, betas=(0.9, 0.999), weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR, patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)


    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn_train)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn_test)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn_test)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn_test)
        else:
            raise NotImplementedError

        return dataloader

    def network(sample):
        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        duration = sample['batch_duration']
        word_label = sample['batch_word_label'].cuda()
        word_mask = sample['batch_word_mask'].cuda()
        video_label = sample['batch_gt_times'].cuda()

        merge_scores, logits_text_2, epis_concepts_2, epis_rewards_2, score_2, logits_text_3, epis_concepts_3, epis_rewards_3, score_3 = model(textual_input, textual_mask, word_mask, visual_input)
        loss_value = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, model.training, logits_text_2, word_label, word_mask, epis_concepts_2, epis_rewards_2, score_2, video_label, concept_label)
        loss_value += getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, model.training, logits_text_3, word_label, word_mask, epis_concepts_3, epis_rewards_3, score_3, video_label, concept_label)
        loss_value = loss_value / 2.

        sorted_times = None if model.training else get_proposal_results(merge_scores, duration)

        if not model.training:
            torch.cuda.empty_cache()

        return loss_value, sorted_times

    def get_proposal_results(video_att, durations):
        out_sorted_times = []

        scores = video_att.cpu().detach().numpy()
        # scores = savgol_filter(video_att, 9, 1, mode= 'nearest')
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

            sorted_indexs = sorted(sorted_indexs, key=lambda x: (cs[x[1]-1]-cs[x[0]-1])/(x[0]-x[1]))
            sorted_indexs = np.array(sorted_indexs).astype(float)
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())

        return out_sorted_times

    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset)/config.TRAIN.BATCH_SIZE*config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'], ncols=100)

    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)

    def on_update(state):# Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg)
            table_message = ''
            if config.TEST.EVAL_TRAIN:
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train')
                train_table = eval.display_results(train_state['Rank@N,mIoU@M'], train_state['miou'],
                                                   'performance on training set')
                table_message += '\n'+ train_table
            if not config.DATASET.NO_VAL:
                val_state = engine.test(network, iterator('val'), 'val')
                state['scheduler'].step(-val_state['loss_meter'].avg)
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                 'performance on validation set')
                table_message += '\n'+ val_table

            test_state = engine.test(network, iterator('test'), 'test')
            loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()
            test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                              'performance on testing set')
            table_message += '\n' + test_table

            message = loss_message+table_message+'\n'
            logger.info(message)

            saved_model_filename = os.path.join(config.MODEL_DIR,'{}/{}/iter{:06d}-{:.4f}-{:.4f}.pkl'.format(
                dataset_name, model_name+'_'+config.DATASET.VIS_INPUT_TYPE,
                state['t'], test_state['Rank@N,mIoU@M'][0,0], test_state['Rank@N,mIoU@M'][0,1]))

            rootfolder1 = os.path.dirname(saved_model_filename)
            rootfolder2 = os.path.dirname(rootfolder1)
            rootfolder3 = os.path.dirname(rootfolder2)
            if not os.path.exists(rootfolder3):
                print('Make directory %s ...' % rootfolder3)
                os.mkdir(rootfolder3)
            if not os.path.exists(rootfolder2):
                print('Make directory %s ...' % rootfolder2)
                os.mkdir(rootfolder2)
            if not os.path.exists(rootfolder1):
                print('Make directory %s ...' % rootfolder1)
                os.mkdir(rootfolder1)

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)


            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'], ncols=100)
            model.train()
            state['loss_meter'].reset()

    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset)/config.TEST.BATCH_SIZE), ncols=100)
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset)/config.TEST.BATCH_SIZE), ncols=100)
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TEST.BATCH_SIZE), ncols=100)
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations, verbose=False)
        if config.VERBOSE:
            state['progress_bar'].close()

    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)
