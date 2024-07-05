import math
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
EPS = sys.float_info.epsilon
import  sys

class InstanceLoss(nn.Module):
    """实例级别的对比损失"""
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    """类簇级别的对比损失"""
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + alpha * ne_loss



def JS_divergence(p, q, ):
    """计算两个分布之间的JS散度"""
    mean = 0.5 * (p + q)
    mean = torch.where(mean < EPS, torch.tensor([EPS], device=mean.device), mean)
    js_div = 0.5 * (F.kl_div(mean.log(), p, reduction='batchmean') + F.kl_div(mean.log(), q, reduction='batchmean'))
    return js_div


def SimCLRLoss(ps, temp=0.1, large_num=1e9):
    """SimClr对比损失"""
    n = ps.size(0) // 2
    h1, h2 = ps[:n], ps[n:]
    h2 = torch.nn.functional.normalize(h2, p=2, dim=1)
    h1 = torch.nn.functional.normalize(h1, p=2, dim=1)

    labels = torch.arange(0, n, device=ps.device, dtype=torch.long)
    masks = torch.eye(n, device=ps.device)

    logits_aa = ((h1 @ h1.t()) / temp) - masks * large_num
    logits_bb = ((h2 @ h2.t()) / temp) - masks * large_num

    logits_ab = (h1 @ h2.t()) / temp
    logits_ba = (h2 @ h1.t()) / temp

    loss_a = torch.nn.functional.cross_entropy(torch.cat((logits_ab, logits_aa), dim=1), labels)
    loss_b = torch.nn.functional.cross_entropy(torch.cat((logits_ba, logits_bb), dim=1), labels)
    loss = (loss_a + loss_b)
    return loss


def SimSiamLoss(h, hs, p, ps):
    """h为project后的特征，p为predict特征"""
    loss1 = - F.cosine_similarity(p, hs.detach(), dim=-1).mean()
    loss2 = - F.cosine_similarity(ps, h.detach(), dim=-1).mean()
    return 0.5 * (loss1 + loss2)

class InstanceLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(
        self,
        tau=0.5,
        multiplier=2,
        distributed=False,
        alpha=0.99,
        gamma=0.5,
        cluster_num=10,
    ):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.alpha = alpha
        self.gamma = gamma
        self.cluster_num = cluster_num

    @torch.no_grad()
    def generate_pseudo_labels(self, c):#传入概率值，伪标签，下标
        pseudo_label_cur=c.shape[0]
        batch_size = c.shape[0]#数据长度
        device = c.device
        pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)#建一个值为-1的tensor
        tmp = torch.arange(0, batch_size).to(device)#建一个值为1的数据

        prediction = c.argmax(dim=1)#取出预测概率最大的的下标
        confidence = c.max(dim=1).values#取出概率最大的值
        unconfident_pred_index = confidence < self.alpha#判断置信度是否小于预设的值
        pseudo_per_class = np.ceil(batch_size / self.cluster_num * self.gamma).astype(
            int
        )#计算大于等于改值的最小整数
        for i in range(self.cluster_num):
            class_idx = prediction == i
            if class_idx.sum() == 0:
                continue
            confidence_class = confidence[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class)
            confident_idx = torch.argsort(-confidence_class)
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]]
                pseudo_label_nxt[idx] = i

        todo_index = pseudo_label_cur == -1
        pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
        pseudo_label_nxt = pseudo_label_cur
        pseudo_label_nxt[unconfident_pred_index] = -1
        return pseudo_label_nxt.cpu(), index

    def forward(self, z, pseudo_label):
        n = z.shape[0]
        assert n % self.multiplier == 0

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            pseudo_label_list = [
                torch.zeros_like(pseudo_label) for _ in range(dist.get_world_size())
            ]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            z_list = diffdist.functional.all_gather(z_list, z)
            pseudo_label_list = diffdist.functional.all_gather(
                pseudo_label_list, pseudo_label
            )
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            pseudo_label_list = [
                chunk for x in pseudo_label_list for chunk in x.chunk(self.multiplier)
            ]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            pesudo_label_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
                    pesudo_label_sorted.append(
                        pseudo_label_list[i * self.multiplier + m]
                    )
            z_i = torch.cat(
                z_sorted[: int(self.multiplier * dist.get_world_size() / 2)], dim=0
            )
            z_j = torch.cat(
                z_sorted[int(self.multiplier * dist.get_world_size() / 2) :], dim=0
            )
            pseudo_label = torch.cat(pesudo_label_sorted, dim=0,)
            n = z_i.shape[0]

        invalid_index = pseudo_label == -1
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(
            z_i.device
        )
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(n).float().to(z_i.device)
        mask &= ~(mask_eye.bool())
        mask = mask.float()

        contrast_count = self.multiplier
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.tau
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask_with_eye = mask | mask_eye.bool()
        # mask = torch.cat(mask)
        mask = mask.repeat(anchor_count, contrast_count)
        mask_eye = mask_eye.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(n * anchor_count).view(-1, 1).to(z_i.device),
            0,
        )
        logits_mask *= 1 - mask
        mask_eye = mask_eye * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_eye * log_prob).sum(1) / mask_eye.sum(1)

        # loss
        instance_loss = -mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, n).mean()

        return instance_loss


class ClusterLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, multiplier=1, distributed=False, cluster_num=10):
        super().__init__()
        self.multiplier = multiplier
        self.distributed = distributed
        self.cluster_num = cluster_num

    def forward(self, c, pseudo_label):
        if self.distributed:
            # c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
            pesudo_label_list = [
                torch.zeros_like(pseudo_label) for _ in range(dist.get_world_size())
            ]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # c_list = diffdist.functional.all_gather(c_list, c)
            pesudo_label_list = diffdist.functional.all_gather(
                pesudo_label_list, pseudo_label
            )
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            # c_list = [chunk for x in c_list for chunk in x.chunk(self.multiplier)]
            pesudo_label_list = [
                chunk for x in pesudo_label_list for chunk in x.chunk(self.multiplier)
            ]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            # c_sorted = []
            pesudo_label_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    # c_sorted.append(c_list[i * self.multiplier + m])
                    pesudo_label_sorted.append(
                        pesudo_label_list[i * self.multiplier + m]
                    )
            # c = torch.cat(c_sorted, dim=0)
            pesudo_label_all = torch.cat(pesudo_label_sorted, dim=0)
        pseudo_index = pesudo_label_all != -1
        pesudo_label_all = pesudo_label_all[pseudo_index]
        idx, counts = torch.unique(pesudo_label_all, return_counts=True)
        freq = pesudo_label_all.shape[0] / counts.float()
        weight = torch.ones(self.cluster_num).to(c.device)
        weight[idx] = freq
        pseudo_index = pseudo_label != -1
        if pseudo_index.sum() > 0:
            criterion = nn.CrossEntropyLoss(weight=weight).to(c.device)
            loss_ce = criterion(
                c[pseudo_index], pseudo_label[pseudo_index].to(c.device)
            )
        else:
            loss_ce = torch.tensor(0.0, requires_grad=True).to(c.device)
        return loss_ce
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        bsz = features.shape[0] // 2#加

        # m.append(features)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # m= (  anchor_dot_contrast[:2]-  anchor_dot_contrast[2:]).norm(p=2, dim=1).pow(2).mean()#
        # n=torch.pdist(  anchor_dot_contrast[:2] , p=2).pow(2).mul(-2).exp().mean().log()
        # n1 = torch.pdist(anchor_dot_contrast[2:], p=2).pow(2).mul(-2).exp().mean().log()
        # n2=n+n1
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def instance_contrastive_Loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss