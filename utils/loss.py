# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device
    #print(device)
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if no == 5 else balance
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no >= 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


# def build_targets(p, targets, model):
#     nt = targets.shape[0]  # number of anchors, targets
#     tcls, tbox, indices, anch = [], [], [], []
#     gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
#     off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

#     g = 0.5  # offset
#     multi_gpu = is_parallel(model)
#     for i, jj in enumerate(model.module.yolo_layers if multi_gpu else model.yolo_layers):
#         # get number of grid points and anchor vec for this yolo layer
#         anchors = model.module.module_list[jj].anchor_vec if multi_gpu else model.module_list[jj].anchor_vec
#         gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

#         # Match targets to anchors
#         a, t, offsets = [], targets * gain, 0
#         if nt:
#             na = anchors.shape[0]  # number of anchors
#             at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)
#             r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
#             j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
#             # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            
#             # a, t = at[j], t.repeat(na, 1, 1)[j]  # old filter
#             # ensure 'j' (index tensor) is on same device as 'at' before indexing
#             if isinstance(j, torch.Tensor) and j.device != at.device:
#                 # move index tensor to device of the tensor being indexed (safe & cheap for 1D indices)
#                 j = j.to(at.device)
#             a = at[j]
#             t = t.repeat(na, 1, 1)[j]


#             # overlaps
#             gxy = t[:, 2:4]  # grid xy
#             z = torch.zeros_like(gxy)
#             j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#             l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
#             a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
#             offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

#         # Define
#         b, c = t[:, :2].long().T  # image, class
#         gxy = t[:, 2:4]  # grid xy
#         gwh = t[:, 4:6]  # grid wh
#         gij = (gxy - offsets).long()
#         gi, gj = gij.T  # grid xy indices

#         # Append
#         #indices.append((b, a, gj, gi))  # image, anchor, grid indices
#         indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
#         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#         anch.append(anchors[a])  # anchors
#         tcls.append(c)  # class

#     return tcls, tbox, indices, anch
def build_targets(p, targets, model):
    nt = targets.shape[0]  # number of targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

    g = 0.5  # offset
    multi_gpu = is_parallel(model)
    for i, jj in enumerate(model.module.yolo_layers if multi_gpu else model.yolo_layers):
        # get number of grid points and anchor vec for this yolo layer
        anchors = model.module.module_list[jj].anchor_vec if multi_gpu else model.module_list[jj].anchor_vec

        # ensure anchors live on same device as targets for safe ops/indexing
        if anchors.device != targets.device:
            anchors = anchors.to(targets.device)

        gain[2:] = torch.tensor(p[i].shape, device=targets.device)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            na = anchors.shape[0]  # number of anchors

            # make anchor tensor on same device as targets (and anchors)
            at = torch.arange(na, device=targets.device).view(na, 1).repeat(1, nt)  # anchor tensor

            # wh ratio (both t and anchors are on targets.device)
            r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # mask (na x nt)

            # if nothing matched, skip this layer
            if not j.any():
                continue

            # filter: keep matched anchors/targets, ensure index tensors are on same device as the indexed tensors
            # 'j' is boolean mask on targets.device; use it to index at and repeated t
            a = at[j]  # anchor indices (on targets.device)
            t = t.repeat(na, 1, 1)[j]  # matched targets (on targets.device)

            # overlaps (compute grid offsets)
            gxy = t[:, 2:4]  # grid xy (on targets.device)
            z = torch.zeros_like(gxy)
            # boolean masks for 4 neighbor offsets
            jx, kx = ((gxy % 1. < g) & (gxy > 1.)).T
            lx, mx = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T

            # concatenate original + 4 shifted copies
            a = torch.cat((a, a[jx], a[kx], a[lx], a[mx]), 0)
            t = torch.cat((t, t[jx], t[kx], t[lx], t[mx]), 0)
            offsets = torch.cat((z, z[jx] + off[0], z[kx] + off[1], z[lx] + off[2], z[mx] + off[3]), 0) * g

        # Define (these assume there is at least one matched t for this layer)
        if t.shape[0] == 0:
            continue

        b, c = t[:, :2].long().T  # image indices, class (both on targets.device)
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Clamp grid indices to valid range (gain holds grid dims)
        gi = gi.clamp_(0, int(gain[2].item() - 1))
        gj = gj.clamp_(0, int(gain[3].item() - 1))

        # Append to lists (all tensors live on targets.device)
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box (dx,dy,w,h)
        anch.append(anchors[a])  # anchors for matched indices
        tcls.append(c)  # class

    return tcls, tbox, indices, anch

