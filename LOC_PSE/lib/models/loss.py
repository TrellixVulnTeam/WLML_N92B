import torch
import torch.nn.functional as F

def bce_rescale_loss(config, training, logits_text, logits_iou, iou_mask_map, word_label, word_mask, gt_times, is_single, video_label):
    N = iou_mask_map.shape[0]
    T = iou_mask_map.shape[-1]

    idxs = torch.arange(T, device=logits_iou.device)
    s_e_idx = torch.cat((idxs[None,None,:T-1,None].repeat(1,1,1,T), idxs[None,None,None,:].repeat(1,1,T-1,1)), 1)
    s_e_time = s_e_idx.repeat(gt_times.size(0),1,1,1).float().clone().detach()

    iou = torch.clamp(torch.min(gt_times[:,:,1][:,:,None,None], s_e_time[:,1,:,:][:,None,:,:]) - torch.max(gt_times[:,:,0][:,:,None,None], s_e_time[:,0,:,:][:,None,:,:]), min=0.) / torch.clamp(torch.max(gt_times[:,:,1][:,:,None,None], s_e_time[:,1,:,:][:,None,:,:]) - torch.min(gt_times[:,:,0][:,:,None,None], s_e_time[:,0,:,:][:,None,:,:]), min=0.0000001)
    iou = torch.max(iou, dim=1)[0]
    
    iou[iou > 0.9] = 1.
    iou[iou < 0.7] = 0.
    thred = torch.topk(iou.view(iou.size(0),-1), 3, dim=1)[0][:,2]
    iou[iou >= thred[:,None,None]] = 1.
    temp = (s_e_time[:,0,:,:] < s_e_time[:,1,:,:]) * iou_mask_map[None,:,:]
    temp_pos = temp * ((iou > 0.9)|(iou >= thred[:,None,None])) * is_single[:,None,None]
    temp_neg = temp * (iou < 0.7)
    
    if training:
        M = int(video_label.shape[0] / video_label.sum().item())
        logits_iou_ = logits_iou.reshape(-1,M,*logits_iou.shape[1:])
        sigmoid_iou_pos = torch.sigmoid(logits_iou_[:,0,2,:,:])
        sigmoid_iou_neg = torch.sigmoid(logits_iou_[:,1:,:,:,:].reshape(-1,*logits_iou.shape[1:]))
        top_sigmoid_iou_neg = torch.topk(sigmoid_iou_neg[:,2,:,:].view(sigmoid_iou_neg.size(0),-1), k=6, dim=1)[0]
        loss_iou_neg = ( - torch.log((torch.clamp(1.-top_sigmoid_iou_neg, min=0.000001))) ).mean()
        word_mask = word_mask.view(-1,M,*word_mask.shape[1:])
        word_mask = word_mask[:,0,:]
        word_label = word_label.view(-1,M,*word_label.shape[1:])
        word_label = word_label[:,0,:]
    else:
        sigmoid_iou_pos = torch.sigmoid(logits_iou[:,2,:,:])
        loss_iou_neg = 0.

    loss_pos = - ( torch.log(torch.clamp(sigmoid_iou_pos, min=0.000001)) * torch.pow(sigmoid_iou_pos-1., 2) * temp_pos ).sum() / torch.clamp(is_single.sum(), min=1.)
    loss_neg = - ( torch.log(torch.clamp(1.-sigmoid_iou_pos, min=0.000001)) * torch.pow(sigmoid_iou_pos, 2) * temp_neg ).sum() / temp_neg.size(0)

    sigmoid_iou_pos_max = torch.topk(sigmoid_iou_pos.view(sigmoid_iou_pos.size(0),-1), config.TOPK, dim=1)[0]
    loss_mil = - ( torch.log(torch.clamp(sigmoid_iou_pos_max, min=0.000001)) * torch.pow(sigmoid_iou_pos_max-1., 2) * (1. - is_single[:,None]) ).sum() / torch.clamp((1. - is_single).sum(), min=1.)

    log_p = F.log_softmax(logits_text, -1) * word_mask.unsqueeze(2)
    grid = torch.arange(log_p.shape[-1], device=log_p.device).repeat(log_p.shape[0], log_p.shape[1], 1)
    text_loss = torch.sum(-log_p[(grid==word_label.unsqueeze(2))]) / logits_text.size(0)

    loss_value = config.W1*loss_pos + loss_neg + loss_iou_neg + config.W2*text_loss + config.W3*loss_mil

    return loss_value, sigmoid_iou_pos*temp, s_e_time
