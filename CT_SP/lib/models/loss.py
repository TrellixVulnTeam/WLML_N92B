import torch
import torch.nn.functional as F

def bce_rescale_loss(config, is_training, logits_text, logits_iou, iou_mask_map, word_label, word_mask, epis_concepts, epis_rewards, video_att, video_label, concept_label):
    N = video_label.shape[0]
    T = video_att.shape[-1]

    loss_pos = - ( video_label * torch.log(torch.clamp(epis_rewards, min=0.000001)) * torch.pow(video_label-epis_rewards, 2) ).sum() / video_label.sum()
    loss_neg = - ( (1.-video_label) * torch.log(torch.clamp(1.-epis_rewards, min=0.000001)) * torch.pow(video_label-epis_rewards, 2) ).sum() / torch.clamp((1.-video_label).sum(), min=1.)

    if is_training:
        M = int(video_label.shape[0] / video_label.sum().item())
        epis_concepts = epis_concepts.reshape(-1,M,*epis_concepts.shape[1:])
        epis_concepts = epis_concepts[:,0,:]
        concept_label = concept_label.reshape(-1,M,*concept_label.shape[1:])
        concept_label = concept_label[:,0,:]

    concept_pos = - (concept_label * torch.log(torch.clamp(epis_concepts, min=0.000001)) * torch.pow(concept_label - epis_concepts, 2) ).sum() / concept_label.sum()
    concept_neg = - ( (1. - concept_label) * torch.log(torch.clamp(1. - epis_concepts, min=0.000001)) * torch.pow(concept_label - epis_concepts, 2) ).sum() / torch.clamp((1. - concept_label).sum(), min=1.)

    idxs = torch.arange(T, device=epis_rewards.device)
    s_e_idx = torch.cat((idxs[None,None,:T-1,None].repeat(1,1,1,T), idxs[None,None,None,:].repeat(1,1,T-1,1)), 1)
    s_e_time = s_e_idx.repeat(N,1,1,1).float().clone().detach()

    log_p = F.log_softmax(logits_text, -1) * word_mask.unsqueeze(2)
    grid = torch.arange(log_p.shape[-1], device=log_p.device).repeat(log_p.shape[0], log_p.shape[1], 1)
    text_loss = torch.sum(-log_p[(grid==word_label.unsqueeze(2))*(video_label[:,None,None]==1.)]) / torch.sum(video_label)

    loss_value = loss_pos + config.W1*loss_neg + config.W2*text_loss + concept_pos + concept_neg

    return loss_value, video_att, s_e_time
