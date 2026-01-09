import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher_crowd

import numpy as np
import time

# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)
    # sub-branch forward
    def forward(self, x):  # 输入：FPN 输出的单层级特征（如 P4_x），维度 (B, 256, H/8, W/8)
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)  # 输出：偏移量预测，维度 (B, N_anchor, 2) N_anchor：当前层级的参考点总数（row×line×(H/8 × W/8)） 2 表示 x、y 方向的偏移量

# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()


    # sub-branch forward
    def forward(self, x): # 输入：同回归分支，(B, 256, H/8, W/8)
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes) # 输出：类别概率预测，维度 (B, N_anchor, num_classes)

# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points

# 空间复制：通过 shift 函数将单个网格的基础参考点复制到全图所有网格，形成覆盖图像的参考点集合
# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):

    # *  stride：从 “特征图网格坐标” 映射到 “输入图像像素坐标”
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride # 生成所有网格的中心坐标（x方向） # shape[1]是特征图宽度（网格数量）
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride  # 生成所有网格的中心坐标（y方向） shape[0]是特征图高度（网格数量）

    shift_x, shift_y = np.meshgrid(shift_x, shift_y) # 生成网格中心坐标的网格矩阵（x和y的所有组合）


    #  # 将网格中心坐标转换为 (K, 2) 格式（K是网格总数）
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()  #  # 形状：(K, 2)，K = 特征图高 × 特征图宽

    A = anchor_points.shape[0]   # 单个网格内的参考点数量（row×line）
    K = shifts.shape[0] # 网格总数

    # 复制基础参考点到每个网格：
    # 1. 将基础参考点从 (A, 2) 扩展为 (1, A, 2)
    # 2. 将网格中心从 (K, 2) 扩展为 (K, 1, 2)
    # 3. 相加后得到 (K, A, 2)，即每个网格的A个参考点
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))

    # 展平为 (K×A, 2)，即所有参考点的坐标
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        #  金字塔层级（默认 [3,4,5,6,7]，对应不同尺度的特征图）
        # 如层级 3 对应输入图像下采样 8 倍的特征图，层级 4 对应下采样 16 倍，以此类推
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels


        #  # 每个层级的步长（stride），默认 2^p（p 为层级）；
        # 特征图上 1 个像素对应输入图像的像素数（如层级 3 的 stride=8，即特征图 1 像素 = 输入图像 8×8 像素；
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        #  # 每个网格内沿高度（row）和宽度（line）方向的参考点数量
        self.row = row  #  # 高度方向的点数
        self.line = line  # 宽度方向的点数

    def forward(self, image):  # (B, 3, H, W)
        image_shape = image.shape[2:]  # 输入图像的高和宽 (H, W)
        image_shape = np.array(image_shape)
        # 例如：输入图像 H=640，层级 3（2^3=8）的特征图高度为 (640 + 8 - 1) // 8 = 80
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels] # 计算每个金字塔层级对应的特征图尺寸（向上取整）确保特征图尺寸不会因整除问题偏小

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)  # 存储所有层级的参考点
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)  # 步骤2.1：生成单个网格内的基础参考点（meta-anchor）
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)  # 步骤2.2：将基础参考点复制到特征图的每个网格，生成全图参考点
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0) # 步骤2.3：拼接当前层级的参考点到总列表


        # N_anchor：所有层级的参考点总数，由 row、line 和金字塔层级决定（默认层级为 [3]）；
        # 若 row=2、line=2，单层级参考点数量为 2×2×(H/8 × W/8)（因层级 3 对应 stride=8），最终 N_anchor 为该值；
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)  # (B, N_anchor, 2)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))

class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)


    def forward(self, inputs):


        # C3：(B, 256, H/4, W/4)；
        # C4：(B, 512, H/8, W/8)；
        # C5：(B, 512, H/16, W/16)
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)   # P5_x：(B, 256, H/16, W/16)

        P4_x = self.P4_1(C4)  # P4_x--->torch.Size([1, 256, 189, 240])
        P4_x = P5_upsampled_x + P4_x  # P5_upsampled_x--->torch.Size([1, 256, 188, 240])
        P4_upsampled_x = self.P4_upsampled(P4_x)   
        P4_x = self.P4_2(P4_x)  # P4_x：(B, 256, H/8, W/8)

        P3_x = self.P3_1(C3)  # P3_x：(B, 256, H/4, W/4)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]

# the defenition of the P2PNet model
class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256, \
                                            num_classes=self.num_classes, \
                                            num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[3,], row=row, line=line)

        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid
        features_fpn = self.fpn([features[1], features[2], features[3]])

        batch_size = features[0].shape[0]
        # run the regression and classification branch
        regression = self.regression(features_fpn[1]) * 100 # 8x
        classification = self.classification(features_fpn[1])

        # 生成参考点（初始形状：(1, N_anchor, 2)，N_anchor是所有参考点总数）
        # 复制到batch中的每个样本：repeat(batch_size, 1, 1) 保持参考点坐标不变，仅扩展batch维度；
        # 批次复制：通过 repeat 方法将参考点集合复制到批次中的每个样本，确保批量处理时参考点位置一致；
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)    # 形状变为 (B, N_anchor, 2)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification
        out = {'pred_logits': output_class, # pred_logits：分类预测，维度 (B, N_anchor, 2)
                'pred_points': output_coord} # pred_points：最终点坐标（参考点 + 偏移量），维度 (B, N_anchor, 2)
       
        return out
    



class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_bbox.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        return losses

# create the P2PNet model
def build(args, training):
    # treats persons as a single class
    num_classes = 1

    backbone = build_backbone(args)
    model = P2PNet(backbone, args.row, args.line)
    if not training: 
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes, \
                                matcher=matcher, weight_dict=weight_dict, \
                                eos_coef=args.eos_coef, losses=losses)

    return model, criterion