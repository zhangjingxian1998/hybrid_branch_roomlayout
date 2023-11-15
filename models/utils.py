import cv2
import numba
import numpy as np
import torch
from torch import nn

from models.loss import _gather_feat, _sigmoid, _transpose_and_gather_feat
from models.visualize import _validate_colormap
from scipy.optimize import linear_sum_assignment
import copy

def get_optimizer(parameters, cfg):
    if cfg.method == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                                    lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.method == 'adam':
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters),
        #                              lr=cfg.lr, weight_decay=cfg.weight_decay)
        optimizer = torch.optim.Adam([{'params': parameters,
                                       'initial_lr': cfg.lr}],
                                     lr=cfg.lr, weight_decay=cfg.weight_decay)
        # optimizer = torch.optim.Adam([{'params': parameters,
        #                                'initial_lr': cfg.lr}],
        #                              lr=cfg.lr)
    elif cfg.method == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters),
                                        lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.method == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, parameters),
                                         lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.name = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = ''

    def update(self, name, val, n=1):
        self.name = name
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# gt produce function
def gaussian_radius(det_size, min_overlap=0.7): # 来自Cornernet的理论代码
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def line_gaussian_horizen(heatmap, points,img, radius=1): # point 先y后x了,交换一下，先x后y
    # points[:,0], points[:,1] =  points[:,1], points[:,0]
    tmp = copy.deepcopy(points[:,0])
    points[:,0] = points[:,1]
    points[:,1] = tmp

    img = img
    diameter = 2 * radius + 1
    h, w = heatmap.shape[1], heatmap.shape[2]
    # line function y = ax + c
    a = (points[0, 1] - points[1, 1])/(points[0, 0] - points[1, 0] + 1e-5)
    c = points[0, 1] - a * points[0, 0]

    #################需要删除掉近乎垂直线#######################
    # if a>1e5:
    #     return
    ########################################################

    # alpha = np.pi / 2 if m == 0 else np.arctan(1 / m)
    # if a<0:
    #     a = -a
    # a = -a if a<0 else pass
    alpha = np.arctan(a)
    alpha = alpha + np.pi if alpha < 0 else alpha
    assert alpha >= 0 and alpha <= np.pi
    alpha_norm = alpha / np.pi

    line_hm = np.zeros((h, w), dtype=np.float32)
    line_alpha = np.zeros((h, w), dtype=np.float32)
    line_offset = np.zeros((h, w), dtype=np.float32)

    points_int = points.astype(np.int32)

    cv2.line(line_hm, tuple(points_int[0]), tuple(points_int[1]), 255, lineType=cv2.LINE_AA)
    img_blur = cv2.GaussianBlur(line_hm, (diameter, 1), diameter / 6, borderType=cv2.BORDER_CONSTANT)
    if np.max(img_blur) == 0:
        return heatmap
    img_blur = img_blur / np.max(img_blur)

    line_int = np.where(img_blur == 1)
    # assert len(line_int[0]) > 0
    if len(line_int[0]) == 0:
        return heatmap
    line_alpha[line_int] = alpha_norm
    # x = m * line_int[0] + b  # x = my+b
    y = a * line_int[1] + c # y = ax + c
    line_offset[line_int] = y - line_int[0]

    heatmap[1] = np.where(heatmap[0] >= img_blur, heatmap[1], line_alpha)
    heatmap[2] = np.where(heatmap[0] >= img_blur, heatmap[2], line_offset)
    heatmap[0] = np.maximum(heatmap[0], img_blur)
    return heatmap

def line_gaussian(heatmap, points, radius=1): # point[i][0]是某一列，point[i][1]是某一行 按照坐标系来说的话就是 x, y
    diameter = 2 * radius + 1
    h, w = heatmap.shape[1], heatmap.shape[2]
    # line function x = my+b
    m = (points[0, 0] - points[1, 0]) / (points[0, 1] - points[1, 1])
    b = points[0, 0] - m * points[0, 1]
    alpha = np.pi / 2 if m == 0 else np.arctan(1 / m)
    alpha = alpha + np.pi if alpha < 0 else alpha
    assert alpha >= 0 and alpha <= np.pi
    alpha_norm = alpha / np.pi

    line_hm = np.zeros((h, w), dtype=np.float32)
    line_alpha = np.zeros((h, w), dtype=np.float32)
    line_offset = np.zeros((h, w), dtype=np.float32)

    points_int = points.astype(np.int32)

    cv2.line(line_hm, tuple(points_int[0]), tuple(points_int[1]), 255, lineType=cv2.LINE_AA)
    img_blur = cv2.GaussianBlur(line_hm, (diameter, 1), diameter / 6, borderType=cv2.BORDER_CONSTANT)
    if np.max(img_blur) == 0:
        return heatmap
    img_blur = img_blur / np.max(img_blur)

    line_int = np.where(img_blur == 1)
    # assert len(line_int[0]) > 0
    if len(line_int[0]) == 0:
        return heatmap
    line_alpha[line_int] = alpha_norm
    x = m * line_int[0] + b  # x = my+b
    line_offset[line_int] = x - line_int[1]

    heatmap[1] = np.where(heatmap[0] >= img_blur, heatmap[1], line_alpha)
    heatmap[2] = np.where(heatmap[0] >= img_blur, heatmap[2], line_offset)
    heatmap[0] = np.maximum(heatmap[0], img_blur)
    return heatmap


# postprocess and evaluate
def _nms(heat, kernel=(3, 3)):
    pad_h = (kernel[0] - 1) // 2 # 做padding用的
    pad_w = (kernel[1] - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, kernel, stride=1, padding=(pad_h, pad_w))
    keep = (hmax == heat).float() # kernel范围内最大的保留
    return heat * keep


# @numba.jit(nopython=True)
def _line_nms(lines, h, w):
    # B*K*4 line must be numpy array(x, y, alpha, score)
    b, n = lines.shape[0], lines.shape[1]
    lines_ = np.zeros((b, n, 4), dtype=np.float32)
    for i in range(b):
        select = np.ones(n, dtype=np.float32)
        for j in range(n):
            if select[j] == 0:
                continue
            l1 = lines[i, j, :3]

            if l1[2] == 0. or l1[2] == 1.:  # the horizontal line discards. 这个l1[2]存放的似乎是角度吧
                select[j] = 0
                continue

            if l1[2] != 0.5:  # convert (x, y, alpha) -> (m, b) x=my+b  这条线不是竖直的
                rad = l1[2] * np.pi
                m1 = 1. / np.tan(rad)
                b1 = l1[0] - m1 * l1[1]
            else: # 这条线是竖直的
                m1 = 0
                b1 = l1[0]
            score = lines[i, j, 3]
            lines_[i, j, :3] = np.array([m1, b1, score]) # 只把斜率、偏置和分数拿了，那么点呢？
            lines_[i, j, 3] = 1.0 # 给个1是干什么呢
            # 是否进行接下来的select操作似乎没有意义，经过测试，与不进行select没有差别，也许是不符合条件的线照样会被剔除
            for k in range(j+1, n):
                clear = 0
                l2 = lines[i, k, :3]

                if l2[2] == 0. or l2[2] == 1.:  # the horizontal line discards.
                    select[k] = 0
                    continue

                if l2[2] != 0.5:  # convert (x, y, alpha) -> (m, b) x=my+b
                    rad = l2[2] * np.pi
                    m2 = 1. / np.tan(rad)
                    b2 = l2[0] - m2 * l2[1]
                else:
                    m2 = 0
                    b2 = l2[0]

                if m1 != m2:  # if two lines are not parallel and their intersection point locates in image, and discard it.
                    common_y = (b2 - b1) / (m1 - m2)
                    common_x = m1 * common_y + b1
                    if common_x > 0 and common_x < w:
                        if common_y > 0 and common_y < h:
                            clear = 1
                # if two line is close, discard it.
                area = _line_similar([m1, b1], [m2, b2], h=128.)
                if area < 5:
                    clear = 1

                if clear == 1:
                    select[k] = 0
    # select = select.reshape(-1,1)
    # lines_ = lines_[0] * select
    # lines_ = lines_.reshape(-1,200,4)
    return lines_


# @numba.jit(nopython=True)
def _box_nms(bbox, threshold=0.5):
    # nms bbox with different threshold with same-class and different-class
    # nms bbox with floor and ceiling not intersection
    # bbox B*K*6 (x y x y score cls)
    b, n = bbox.shape[0], bbox.shape[1]
    bbox_ = np.zeros((b, n, 7), dtype=np.float32)
    for i in range(b):
        select = np.ones(n, dtype=np.float32)
        for j in range(n):
            if select[j] == 1:
                bbox_[i, j, :6] = bbox[i, j]
                bbox_[i, j, 6] = 1
                for k in range(j+1, n):
                    box1 = bbox[i, j, :4]
                    box2 = bbox[i, k, :4]
                    iou = _iou(box1, box2)

                    threshold_ = 0.9 if bbox[i, j, 5] != bbox[i, k, 5] else threshold  # the different category has rarely intersect.
                    threshold_ = 0. if bbox[i, j, 5] == 1 and bbox[i, k, 5] == 2 else threshold_  # floor and ceiling do not intersect.
                    threshold_ = 0. if bbox[i, j, 5] == 2 and bbox[i, k, 5] == 1 else threshold_  # floor and ceiling do not intersect.

                    if iou > threshold_:
                        select[k] = 0.
    return bbox_


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()# 批数， 通道，高，宽
    # 每一个cat 代表什么呢
    # 就是找到置信度最高的20个平面吧
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K) # 将所有像素值归为一个通道，返回所有像素位置最大的K个值，第一个返回值，第二个返回值的索引
    # topk_scores [1,3,20]  topk_inds [1,3,20]
    topk_inds = topk_inds % (height * width) # 这一步的意义何在呢，没有任何变化
    topk_ys = (topk_inds / width).int().float() # 索引位置恢复
    # topk_ys = torch.true_divide(topk_inds,width)
    topk_xs = (topk_inds % width).int().float()
                                        # topk_scores [1,60]
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K) # 在所有层上再做一次取最大，取20个
    topk_clses = (topk_ind / K).int() # 确定是属于哪个cat
    # topk_clses = torch.true_divide(topk_ind ,K)
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    # 从原来 1*3*160*96 个中最终提取到20个置信度最高的索引位置
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def post_process(outputs, K1=20, K2=200, Mnms=1):
    # plane, 3D params and lines parser
    # planes prediction
    # outputs是HRNET32提取的各种特征图，特征图大小是输入图像分辨率的1/4
    plane_center = outputs['plane_center']
    plane_wh = outputs['plane_wh']
    plane_offset = outputs['plane_offset']
    # 3D params prediction
    plane_params_pixelwise = outputs['plane_params_pixelwise']
    plane_params_instance = outputs['plane_params_instance']
    # lines prediction
    line_region = outputs['line_region'] # 这是线的点位置特征吗
    line_params = outputs['line_params']
    line_offset = line_params[:, 0:1]   # 线x方向偏置特征
    line_alpha = line_params[:, 1:2]    # 线角度特征

    # plane location decode
    batch, cat, height, width = plane_center.size() # 每个cat是代表着是墙还是天花板还是地板
    plane_center = _nms(plane_center, (7, 7)) # 做了一个7*7的same_size最大池化
    scores, inds, clses, ys, xs = _topk(plane_center, K=K1) # 把plane_center 作为scores传了进去，出来20个置信度最高的，返回它们的分数，索引位置，，中心点坐标
    # scores [1,20] 得分最高的20个平面的得分，inds [1,20]，这20个平面的索引，clses [1,20] 所对应的cat ,ys,xs[1,20] 平面中心坐标
    reg = _transpose_and_gather_feat(plane_offset, inds) # 偏置 [1,20,2],第一个存的是x偏移，第二个存的y的偏移
    reg = reg.view(batch, K1, 2)
    xs = xs.view(batch, K1, 1) + reg[:, :, 0:1]# 获取最终位置信息
    ys = ys.view(batch, K1, 1) + reg[:, :, 1:2]
    wh = _transpose_and_gather_feat(plane_wh, inds) # 这20个平面多宽多高
    wh = wh.view(batch, K1, 2)
    clses = clses.view(batch, K1, 1).float()# 保证形状一致
    scores = scores.view(batch, K1, 1) # 保证形状一致

    bboxes = torch.cat([xs - wh[..., 0:1] / 2, # 两个角点坐标，确定20个平面的box
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    dt_planes = torch.cat([bboxes, scores, clses], dim=2).data.cpu().numpy()  #B*K*6 位置，分数，所属cat拼接起来[1,20,6]

    # norm plane 3D params and get instance plane params # p指定范数 此处为2范数 dim指定维度进行范数运算 keepdim 保留运算的维度
    # 这里为什么要算2范数呢？？？？？？？？？？止步于此
    plane_params_instance /= torch.norm(plane_params_instance[:, :3], p=2, dim=1, keepdim=True)
    plane_params_instance = _transpose_and_gather_feat(plane_params_instance, inds).data.cpu().numpy()

    # norm plane 3D params and get pixelwise plane params
    plane_params_pixelwise /= torch.norm(plane_params_pixelwise[:, :3], p=2, dim=1, keepdim=True)
    plane_params_pixelwise = plane_params_pixelwise.data.cpu().numpy()

    # line location decode
    line_region = _nms(line_region, (1, 3)) # 一行中如果是连续3个最大的，保留，否则，不保留
    scores, inds, clses, ys, xs = _topk(line_region, K=K2)
    reg_offset = _transpose_and_gather_feat(line_offset, inds)[:, :, 0]
    reg_alpha = _transpose_and_gather_feat(line_alpha, inds)[:, :, 0]
    xs = xs + reg_offset
    dt_lines = torch.stack([xs, ys, reg_alpha, scores], dim=2).data.cpu().numpy()  #B*K*4

    # filter plane and line by the second step nms (intersection case)
    if Mnms == 1:
        dt_planes = _box_nms(dt_planes)
        dt_lines = _line_nms(dt_lines, height, width)
    return dt_planes, dt_lines, plane_params_instance, plane_params_pixelwise


def gt_check(batch):
    # extract gt
    plane_hm = batch['plane_hm']
    b, c, h, w = plane_hm.size()

    ind = batch['ind']
    plane_wh = batch['plane_wh']
    plane_offset = batch['plane_offset']
    reg_mask = batch['reg_mask']

    plane_params_instance = batch['params3d']

    # extract plane
    gt_planes = []
    gt_params3d = []
    for i in range(b):
        index = ind[i, reg_mask[i] == 1]
        y = index // w
        x = index % w
        wh = plane_wh[i, reg_mask[i] == 1]
        # wh[:, 0] *= w
        # wh[:, 1] *= h
        offset = plane_offset[i, reg_mask[i] == 1]
        ys = y + offset[:, 1]
        xs = x + offset[:, 0]
        box = torch.stack([xs - wh[:, 0] / 2,
                            ys - wh[:, 1] / 2,
                            xs + wh[:, 0] / 2,
                            ys + wh[:, 1] / 2], dim=1)
        gt_planes.append(box.data.cpu().numpy())
        gt_params3d.append(plane_params_instance[i, reg_mask[i] == 1])

    # extract line
    line_hm = batch['line_hm']
    line_alpha = batch['line_alpha']
    line_offset = batch['line_offset']

    # max 500 proposal for line
    lines = torch.zeros((b, 200, 4), device=line_hm.device)
    ones = torch.ones((b, 200), device=line_hm.device)
    for i in range(b):
        y, x = torch.nonzero(line_hm[i, 0] == 1, as_tuple=True)
        xs = x + line_offset[i, 0, y, x]
        alpha = line_alpha[i, 0, y, x]
        num = min(len(y), 200)
        lines[i, :num] = torch.stack([xs[:num], y[:num].float(), alpha[:num], ones[i, :num]], dim=1)

    lines = lines.data.cpu().numpy()
    gt_lines = _line_nms(lines, h, w)

    return gt_planes, gt_lines, gt_params3d


# @numba.jit(nopython=True)
def _iou(box1, box2):
  area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
  area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
  inter = max(min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1, 0) * \
          max(min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1, 0)
  iou = 1.0 * inter / (area1 + area2 - inter)
  return iou


# @numba.jit(nopython=True)
def evaluate_planes(dt_planes, gt_planes):
    # eval planes
    bs = len(dt_planes) # batch size
    score_threshold = np.arange(0.1, 1.1, 0.1, dtype=np.float32) 
    num = len(score_threshold) # 10
    APs = np.zeros((bs, num), dtype=np.float32) # 1, 10
    ARs = np.zeros((bs, num), dtype=np.float32) # 1, 10
    for b in range(bs):
        m = len(dt_planes[b]) # 拿到20个plane
        n = len(gt_planes[b]) # 拿到gt plane
        success = np.zeros(m) # 
        pick = np.zeros(n) # 
        scores = np.zeros(m) # 
        for i in range(m): # 遍历检测到的20个面
            dt = dt_planes[b][i] # 循环遍历
            scores[i] = dt[4] # 拿到当前面的分数
            for j in range(n): # 把这个面与gt面循环比对
                if pick[j] == 0: # 第j个面没有与之匹配的检测面
                    gt = gt_planes[b][j]
                    iou = _iou(dt[:4], gt[:4])
                    if iou > 0.5: # 交并比大于0.5的认为是匹配的面，一个gt面只能与一个检测面匹配
                        success[i] = 1
                        pick[j] = 1
                        continue
        scores = np.reshape(scores, (m, 1))
        scores = scores > score_threshold # 还要验证一下分数阈值 一个检测分数与10个指定分数比较，生成矩阵
        success = np.reshape(success, (m, 1))
        success = success * scores # 如果success是1，scores对应的行保留，否则scores对应的行舍弃
        AP = np.sum(success, axis=0) / (np.sum(scores, axis=0) + 1e-5) # score大于阈值的才应该是好，但是会让分母变大，
        AR = np.sum(success, axis=0) / (n + 1e-5) # 召回率升高，匹配到的结果好，召回率减少，匹配到的效果差
        APs[b] = AP
        ARs[b] = AR

    APs = np.sum(APs, axis=0) / bs # batch 是1时，无效果
    APs = APs[::-1]
    ARs = np.sum(ARs, axis=0) / bs
    ARs = ARs[::-1]
    APs = np.concatenate((np.array([1.]), APs), axis=0)
    ARs = np.concatenate((np.array([0.]), ARs), axis=0)

    dARs = ARs[1:] - ARs[:-1] # 为什么？
    dAPs = APs[1:] # ？
    mAP = np.sum(dARs * dAPs)
    mAR = ARs[-1]

    return mAR, mAP


# @numba.jit(nopython=True)
def _line_similar(line1, line2, h):
    # (m, b) x = my + b
    x10, y10 = line1[1] * 1., 0.
    x11, y11 = line1[0] * h + line1[1], h
    x20, y20 = line2[1] * 1., 0.
    x21, y21 = line2[0] * h + line2[1], h
    # xmin, xmax = min(x10, x11, x20, x21), max((x10, x11, x20, x21))
    # ymin, ymax = min((y10, y11, y20, y21)), max((y10, y11, y20, y21))
    # area = (xmax - xmin) * (ymax - ymin)
    dupx = abs(x10 - x20)
    ddownx = abs(x11 - x21)
    area = max(dupx, ddownx)
    # print(area)
    return area


# @numba.jit(nopython=True)
def evaluate_lines(dt_lines, gt_lines):
    # eval lines
    bs = len(dt_lines)
    score_threshold = np.arange(0.1, 1.1, 0.1)
    num = len(score_threshold)
    APs = np.zeros((bs, num), dtype=np.float32)
    ARs = np.zeros((bs, num), dtype=np.float32)
    for b in range(bs):
        m = len(dt_lines[b])
        n = len(gt_lines[b])
        success = np.zeros(m)
        pick = np.zeros(n)
        scores = np.zeros(m)
        for i in range(m):
            dt = dt_lines[b][i]
            scores[i] = dt[2]
            for j in range(n):
                if pick[j] == 0:
                    gt = gt_lines[b][j]
                    area = _line_similar(dt[:2], gt[:2], float(128.0))
                    if area < 10:
                        success[i] = 1
                        pick[j] = 1
                        continue

        scores = np.reshape(scores, (m, 1))
        scores = scores > score_threshold
        success = np.reshape(success, (m, 1))
        success = success * scores
        AP = np.sum(success, axis=0) / (np.sum(scores, axis=0) + 1e-5)
        AR = np.sum(success, axis=0) / (n + 1e-5)
        APs[b] = AP
        ARs[b] = AR

    APs = np.sum(APs, axis=0) / bs
    APs = APs[::-1]
    ARs = np.sum(ARs, axis=0) / bs
    ARs = ARs[::-1]
    APs = np.concatenate((np.array([1.]), APs), axis=0)
    ARs = np.concatenate((np.array([0.]), ARs), axis=0)

    dARs = ARs[1:] - ARs[:-1]
    dAPs = APs[1:]
    mAP = np.sum(dARs * dAPs)
    mAR = ARs[-1]

    return mAR, mAP


# @numba.jit(nopython=True)
def evaluate(dt_planes, dt_lines, gt_planes, gt_lines):
    mAR_p, mAP_p = evaluate_planes(dt_planes, gt_planes)
    mAR_l, mAP_l = evaluate_lines(dt_lines, gt_lines)
    return mAR_p, mAP_p, mAR_l, mAP_l

import logging
import os

# display training info
from tensorboardX import SummaryWriter


def printfs(cfg):
    base_path = f'./Logs/{cfg.model_name}'
    if not os.path.isdir(base_path):
        os.makedirs(os.path.join(base_path, 'summary', 'train'))
        os.makedirs(os.path.join(base_path, 'summary', 'validation'))
    writer_train = SummaryWriter(os.path.join(base_path, 'summary', 'train'))
    writer_val = SummaryWriter(os.path.join(base_path, 'summary', 'validation'))
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #create handler (file and terminal)
    fh = logging.FileHandler(os.path.join(base_path, 'log.log'), mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    #set format
    formatter = logging.Formatter('%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    #add handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return writer_train, writer_val, logger


# display layout results
def DisplayLayout(img, segs_opt, depth_opt, polys_opt, segs_noopt, depth_noopt, polys_noopt, segs_gt, label, iters):
    palette = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
        (1.0, 0.4980392156862745, 0.054901960784313725),
        (1.0, 0.7333333333333333, 0.47058823529411764),
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
        (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
        (1.0, 0.596078431372549, 0.5882352941176471),
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
        (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
        (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
        (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
        (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
        (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
        (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
    ]
    palette = np.concatenate([np.array(palette)[[0, 1, 5,  10,  16,  19]], np.random.uniform(0, 1, (20, 3))], axis=0)
    img /= 255
    palette = np.array(palette)
    from scipy.optimize import linear_sum_assignment
    _segs_gt = []
    _segs_noopt = []
    _segs_opt = []
    for i in np.unique(segs_gt):
        if i == -1:
            continue
        else:
            _segs_gt.append(segs_gt==i)
    for i in np.unique(segs_noopt):
        if i == -1:
            continue
        else:
            _segs_noopt.append(segs_noopt==i)
    for i in np.unique(segs_opt):
        if i == -1:
            continue
        else:
            _segs_opt.append(segs_opt==i)
    _segs_gt = np.array(_segs_gt).astype(np.int)
    _segs_noopt = np.array(_segs_noopt).astype(np.int)
    _segs_opt = np.array(_segs_opt).astype(np.int)
    cost1 = ((_segs_gt[:, np.newaxis] + _segs_noopt)==2).sum((2, 3))
    cost2 = ((_segs_gt[:, np.newaxis] + _segs_opt)==2).sum((2, 3))
    r1, c1 = linear_sum_assignment(-1*cost1)
    r2, c2 = linear_sum_assignment(-1*cost2)

    color1 = np.arange(2, len(_segs_gt)+2)
    for i in range(len(color1)):
        if label[i] == 1:
            color1[i] = 0
        elif label[i] == 2:
            color1[i] = 1
    color2 = []
    color3 = []
    for i in range(len(_segs_noopt)):
        if i in c1:
            idx = np.where(c1==i)[0][0]
            color2.append(color1[r1[idx]])
        else:
            color2.append(None)
    j = 0
    for i in range(len(color2)):
        if color2[i] is None:
            color2[i] = 2+len(color1) + j
            j = j + 1
    for i in range(len(_segs_opt)):
        if i in c2:
            idx = np.where(c2 == i)[0][0]
            color3.append(color1[r2[idx]])
        else:
            color3.append(None)
    j = 0
    for i in range(len(color3)):
        if color3[i] is None:
            color3[i] = 2+len(color1) + len(color2) + j
            j = j + 1


    masks_opt = np.zeros_like(img, dtype=np.float32)  # opt
    bound_opt = np.zeros_like(img[:,:,0], dtype=np.float32)
    masks_noopt = np.zeros_like(img, dtype=np.float32)  # no opt
    bound_noopt = np.zeros_like(img[:, :, 0], dtype=np.float32)
    masks_gt = np.zeros_like(img, dtype=np.float32)  # gt
    bound_gt = np.zeros_like(img[:, :, 0], dtype=np.float32)
    img1 = img
    for i in range(len(color1)):
        alpha_fill = (_segs_gt[i]==1)[..., None].astype(np.float32)
        sx = cv2.Sobel(alpha_fill, cv2.CV_32F, 1, 0, ksize=5)
        sy = cv2.Sobel(alpha_fill, cv2.CV_32F, 0, 1, ksize=5)
        alpha_edge = (sx ** 2 + sy ** 2) ** 0.5
        alpha_edge /= max(0.001, np.max(alpha_edge))
        alpha_edge = alpha_edge[..., None]
        alpha_fill *= 0.5
        color = palette[color1][i]
        img1 = img1 * (1 - alpha_fill) + alpha_fill * color
        img1 = img1 * (1 - alpha_edge) + alpha_edge * color
        bound_gt[alpha_edge[:,:,0]>0] = 1.

    img2 = img
    for i in range(len(color2)):
        alpha_fill = (_segs_noopt[i] == 1)[..., None].astype(np.float32)
        sx = cv2.Sobel(alpha_fill, cv2.CV_32F, 1, 0, ksize=5)
        sy = cv2.Sobel(alpha_fill, cv2.CV_32F, 0, 1, ksize=5)
        alpha_edge = (sx ** 2 + sy ** 2) ** 0.5
        alpha_edge /= max(0.001, np.max(alpha_edge))
        alpha_edge = alpha_edge[..., None]
        alpha_fill *= 0.5
        color = palette[color2][i]
        img2 = img2 * (1 - alpha_fill) + alpha_fill * color
        img2 = img2 * (1 - alpha_edge) + alpha_edge * color

        bound_noopt[alpha_edge[:,:,0]>0]=1

    img3=img
    for i in range(len(color3)):
        alpha_fill = (_segs_opt[i] == 1)[..., None].astype(np.float32)
        sx = cv2.Sobel(alpha_fill, cv2.CV_32F, 1, 0, ksize=5)
        sy = cv2.Sobel(alpha_fill, cv2.CV_32F, 0, 1, ksize=5)
        alpha_edge = (sx ** 2 + sy ** 2) ** 0.5
        alpha_edge /= max(0.001, np.max(alpha_edge))
        alpha_edge = alpha_edge[..., None]
        alpha_fill *= 0.5
        color = palette[color3][i]
        img3 = img3 * (1 - alpha_fill) + alpha_fill * color
        img3 = img3 * (1 - alpha_edge) + alpha_edge * color

        bound_opt[alpha_edge[:,:,0]>0]=1.

    img4=np.copy(img)
    img4[bound_gt==1]=np.array([0, 1, 0])
    img5=np.copy(img)
    img5[bound_noopt==1]=np.array([0, 1, 0])
    img6=np.copy(img)
    img6[bound_opt==1]=np.array([0, 1, 0])

    cv2.imwrite(f'results/{iters}_select.png',
                np.concatenate([img[:360], img1[:360], img2[:360], img3[:360]], axis=1) * 255)

def display2Dseg(img, segs_pred, segs_gt, label, iters, method=9, draw_gt=0):
    img = img.cpu().numpy().transpose([1, 2, 0])
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img = ((img * std) + mean) * 255
    img = img[:, :, ::-1]
    h, w = img.shape[0], img.shape[1]

    palette = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
        (1.0, 0.4980392156862745, 0.054901960784313725),
        (1.0, 0.7333333333333333, 0.47058823529411764),
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
        (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
        (1.0, 0.596078431372549, 0.5882352941176471),
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
        (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
        (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
        (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
        (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
        (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
        (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
    ]
    img /= 255
    palette = np.array(palette)
    from scipy.optimize import linear_sum_assignment
    _segs_gt = []
    _segs_pred = []
    for i in np.unique(segs_gt):
        if i == -1:
            continue
        else:
            _segs_gt.append(segs_gt == i)
    for i in np.unique(segs_pred):
        if i == -1:
            continue
        else:
            _segs_pred.append(segs_pred == i)
    _segs_gt = np.array(_segs_gt).astype(np.int)
    _segs_pred = np.array(_segs_pred).astype(np.int)
    cost1 = ((_segs_gt[:, np.newaxis] + _segs_pred) == 2).sum((2, 3))
    r1, c1 = linear_sum_assignment(-1 * cost1)
    color1 = np.arange(2, len(_segs_gt) + 2)
    for i in range(len(color1)):
        if label[i] == 1:
            color1[i] = 0
        elif label[i] == 2:
            color1[i] = 1
    color2 = []
    for i in range(len(_segs_pred)):
        if i in c1:
            idx = np.where(c1 == i)[0][0]
            color2.append(color1[r1[idx]])
        else:
            color2.append(None)
    j = 0
    for i in range(len(color2)):
        if color2[i] is None:
            color2[i] = 2 + len(color1) + j
            j = j + 1
    bound_pred = np.zeros_like(img[:, :, 0], dtype=np.float32)
    bound_gt = np.zeros_like(img[:, :, 0], dtype=np.float32)
    img1 = img
    for i in range(len(color1)):
        alpha_fill = (_segs_gt[i] == 1)[..., None].astype(np.float32)
        sx = cv2.Sobel(alpha_fill, cv2.CV_32F, 1, 0, ksize=5)
        sy = cv2.Sobel(alpha_fill, cv2.CV_32F, 0, 1, ksize=5)
        alpha_edge = (sx ** 2 + sy ** 2) ** 0.5
        alpha_edge /= max(0.001, np.max(alpha_edge))
        alpha_edge = alpha_edge[..., None]
        alpha_fill *= 0.5
        color = palette[color1][i]
        img1 = img1 * (1 - alpha_fill) + alpha_fill * color
        img1 = img1 * (1 - alpha_edge) + alpha_edge * color
        bound_gt[alpha_edge[:, :, 0] > 0] = 1.

    img2 = img
    for i in range(len(color2)):
        alpha_fill = (_segs_pred[i] == 1)[..., None].astype(np.float32)
        sx = cv2.Sobel(alpha_fill, cv2.CV_32F, 1, 0, ksize=5)
        sy = cv2.Sobel(alpha_fill, cv2.CV_32F, 0, 1, ksize=5)
        alpha_edge = (sx ** 2 + sy ** 2) ** 0.5
        alpha_edge /= max(0.001, np.max(alpha_edge))
        alpha_edge = alpha_edge[..., None]
        alpha_fill *= 0.5
        color = palette[color2][i]
        img2 = img2 * (1 - alpha_fill) + alpha_fill * color
        img2 = img2 * (1 - alpha_edge) + alpha_edge * color

        bound_pred[alpha_edge[:, :, 0] > 0] = 1

    if not os.path.exists(f'results'):
        os.mkdir(f'results/')
    
    cv2.imwrite(f'results/{iters}.png', img2 * 255)
    if draw_gt > 0:
        cv2.imwrite(f'results/{iters}_{draw_gt}.png', img1 * 255)

def match_by_Hungarian(gt, pred):
    n = len(gt)
    m = len(pred)
    gt = np.array(gt)
    pred = np.array(pred)
    valid = (gt.sum(0) > 0).sum()
    if m == 0:
        raise IOError
    else:
        gt = gt[:, np.newaxis, :, :]
        pred = pred[np.newaxis, :, :, :]
        cost = np.sum((gt+pred) == 2, axis=(2, 3))  # n*m
        row, col = linear_sum_assignment(-1 * cost)
        inter = cost[row, col].sum()
        PE = inter / valid
        return 1 - PE


def test_evaluate(gtseg, gtdepth, preseg, predepth, evaluate_2D=True, evaluate_3D=True):
    # 四个输入，分别是分割和深度图，两个是真值，两个预测值
    image_iou, image_pe, merror_edge, rmse, us_rmse = 0, 0, 0, 0, 0
    if evaluate_2D:
        # Parse GT polys
        gt_polys_masks = []
        h, w = gtseg.shape
        gt_polys_edges_mask = np.zeros((h, w))
        edge_thickness = 1
        gt_valid_seg = np.ones((h, w)) # 有效区域，有效的像素值为1 # 有效区域意味着什么呢
        labels = np.unique(gtseg) # 取分割图中一共有哪些类，即图像中有哪些所关注的对象
        for i, label in enumerate(labels):
            gt_poly_mask = gtseg == label # gt_poly_mask存放的是对应lable的掩膜

            #############显式单个掩膜############
            # cv2.imshow('2',gt_poly_mask.astype(np.uint8)*255)
            # cv2.waitKey()
            #####################################

            if label == -1:
                gt_valid_seg[gt_poly_mask] = 0  # zero pad region   label=-1被认为是padding的0，为无效区域
            else:
                # contours中存放的是连续边界点的坐标值 hierarchy 第三个参数是判断是否是外轮廓，为-1，是外轮廓
                contours_, hierarchy = cv2.findContours(gt_poly_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cv2.polylines(gt_polys_edges_mask, contours_, isClosed=True, color=[
                              1.], thickness=edge_thickness)
                
                ###############显式边线#########################
                # cv2.imshow('2',gt_polys_edges_mask.astype(np.uint8)*255)
                # cv2.waitKey()
                ################################################
                
                gt_polys_masks.append(gt_poly_mask.astype(np.int32)) # 这里将所有掩膜存进来是干嘛呢

        def sortPolyBySize(mask):
            return mask.sum()
        gt_polys_masks.sort(key=sortPolyBySize, reverse=True) # 按大小从小到大排序然后反排序，总的来说是从大到小，就是平面的面积最大的在前面

        # Parse predictions
        pred_polys_masks = []
        pred_polys_edges_mask = np.zeros((h, w))
        pre_invalid_seg = np.zeros((h, w))
        labels = np.unique(preseg)
        for i, label in enumerate(labels):
            pred_poly_mask = np.logical_and(preseg == label, gt_valid_seg == 1)
            if pred_poly_mask.sum() == 0: # 没有预测出平面
                continue
            if label == -1: # 有效区域
                # zero pad and infinity region
                pre_invalid_seg[pred_poly_mask] = 1
            else:
                contours_, hierarchy = cv2.findContours(pred_poly_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_SIMPLE
                cv2.polylines(pred_polys_edges_mask, contours_, isClosed=True, color=[
                              1.], thickness=edge_thickness)
                pred_polys_masks.append(pred_poly_mask.astype(np.int32))
        if len(pred_polys_masks) == 0.: # 没有预测出平面，为什么要这样写，把预测的边缘图全置0，多边形掩膜全置1
            pred_polys_edges_mask[edge_thickness:-
                                  edge_thickness, edge_thickness:-edge_thickness] = 1
            pred_polys_edges_mask = 1 - pred_polys_edges_mask
            pred_poly_mask = np.ones((h, w))
            pred_polys_masks = [pred_poly_mask]

        pred_polys_masks_cand = copy.copy(pred_polys_masks) # 浅拷贝
        # 这个预测的没有经过过排序哈
        # Assign predictions to ground truth polygons
        ordered_preds = []
        for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
            best_iou_score = 0.3
            best_pred_ind = None
            best_pred_poly_mask = None
            if len(pred_polys_masks_cand) == 0: # 没有预测出分割平面
                break
            for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand):
                gt_pred_add = gt_poly_mask + pred_poly_mask # 重叠的地方值为2，非重叠地方为1，
                inter = np.equal(gt_pred_add, 2.).sum() # 计算重叠像素总数，equal返回的是布尔值，就是交集
                union = np.greater(gt_pred_add, 0.).sum() # 计算重叠区域和非重叠区域，就是并集
                iou_score = inter / union # 交并比IoU

                if iou_score > best_iou_score: # 如果发现交并比更好的平面，进行迭代更新
                    best_iou_score = iou_score
                    best_pred_ind = pred_ind
                    best_pred_poly_mask = pred_poly_mask
            # 循环结束后，能够拿到预测平面中与某个gt平面交并比最高的平面index
            ordered_preds.append(best_pred_poly_mask) # 将其一一保存起来

            pred_polys_masks_cand = [pred_poly_mask for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand)
                                     if pred_ind != best_pred_ind] # 更新pred_polys_masks_cand，把best_pred_ind对应的掩膜给去掉
            if best_pred_poly_mask is None: # 经过一轮循环，没有交并比大于0.3的，换下一个
                continue
        # 此处循环结束后，ordered_preds存放的是原预测平面中与某个gt平面交并比大于0.3的平面，pred_polys_masks_cand中还有与所有gt平面都不大于0.3的平面
        ordered_preds += pred_polys_masks_cand # 这俩是个list类型，所以是拼接，大小恢复。
        class_num = max(len(ordered_preds), len(gt_polys_masks)) # 为什么要取最大的呢，如果预测了10个面，gt只有4个呢
        colormap = _validate_colormap(None, class_num + 1) # 为不同的类生成不同的颜色

        # Generate GT poly mask
        gt_layout_mask = np.zeros((h, w))
        gt_layout_mask_colored = np.zeros((h, w, 3))
        for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
            gt_layout_mask = np.maximum(
                gt_layout_mask, gt_poly_mask * (gt_ind + 1)) # 两个数组按位相与，取最大值，但是如果有重叠会怎么样呢，可能有重叠吗？
            gt_layout_mask_colored += gt_poly_mask[:,
                                                   :, None] * colormap[gt_ind + 1]
########################显式两种真值掩膜################################
        # cv2.imshow('2',gt_layout_mask.astype(np.uint8)*np.array((255/np.max(gt_layout_mask.astype(np.uint8))),dtype='uint8'))
        # cv2.imshow('2',gt_layout_mask_colored)
        # cv2.waitKey()
######################################################################

        # Generate pred poly mask
        pred_layout_mask = np.zeros((h, w))
        pred_layout_mask_colored = np.zeros((h, w, 3))
        for pred_ind, pred_poly_mask in enumerate(ordered_preds):
            if pred_poly_mask is not None:
                pred_layout_mask = np.maximum(
                    pred_layout_mask, pred_poly_mask * (pred_ind + 1))
                pred_layout_mask_colored += pred_poly_mask[:,
                                                           :, None] * colormap[pred_ind + 1] # 切片加个None,增加一个维度，用来存放色彩信息
########################显式两种预测掩膜################################
        # cv2.imshow('2',pred_layout_mask.astype(np.uint8)*np.array((255/np.max(pred_layout_mask.astype(np.uint8))),dtype='uint8'))
        # cv2.imshow('2',pred_layout_mask_colored)
        # cv2.waitKey()
######################################################################

        # Calc IOU
        ious = []
        for layout_comp_ind in range(1, len(gt_polys_masks) + 1):
            inter = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                   np.equal(pred_layout_mask, layout_comp_ind)).sum() # 将平面一种一种地比较
            fp = np.logical_and(np.not_equal(gt_layout_mask, layout_comp_ind),
                                np.equal(pred_layout_mask, layout_comp_ind)).sum()
            fn = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                np.not_equal(pred_layout_mask, layout_comp_ind)).sum() # 为什么不用np.logical_or()呢，还要这样子复杂
            union = inter + fp + fn
            iou = inter / union
            ious.append(iou)

        image_iou = sum(ious) / class_num # 平均交并比

        # Calc PE # 这样计算是否意味着墙面与墙面之间没有区别
        image_pe = 1 - np.equal(gt_layout_mask[gt_valid_seg == 1], 
                                pred_layout_mask[gt_valid_seg == 1]).sum() / (np.sum(gt_valid_seg == 1))
        # Calc PE by Hungarian 匈牙利匹配
        image_pe_hung = match_by_Hungarian(gt_polys_masks, pred_polys_masks)
        # Calc edge error
        # ignore edges at image borders
        img_bound_mask = np.zeros_like(pred_polys_edges_mask)
        img_bound_mask[10:-10, 10:-10] = 1

        pred_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - pred_polys_edges_mask)).astype(np.uint8),
                                                cv2.DIST_L2, 3)
        gt_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - gt_polys_edges_mask)).astype(np.uint8),
                                              cv2.DIST_L2, 3)

        chamfer_dist = pred_polys_edges_mask * gt_dist_trans + \
            gt_polys_edges_mask * pred_dist_trans
        merror_edge = 0.5 * np.sum(chamfer_dist) / np.sum(
            np.greater(img_bound_mask * (gt_polys_edges_mask), 0))

    # Evaluate in 3D
    if evaluate_3D:
        max_depth = 50
        gt_layout_depth_img_mask = np.greater(gtdepth, 0.)
        gt_layout_depth_img = 1. / gtdepth[gt_layout_depth_img_mask]
        gt_layout_depth_img = np.clip(gt_layout_depth_img, 0, max_depth)
        gt_layout_depth_med = np.median(gt_layout_depth_img)
        # max_depth = np.max(gt_layout_depth_img)
        # may be max_depth should be max depth of all scene
        predepth[predepth == 0] = 1 / max_depth
        pred_layout_depth_img = 1. / predepth[gt_layout_depth_img_mask]
        pred_layout_depth_img = np.clip(pred_layout_depth_img, 0, max_depth)
        pred_layout_depth_med = np.median(pred_layout_depth_img)

        # Calc MSE
        ms_error_image = (pred_layout_depth_img - gt_layout_depth_img) ** 2
        rmse = np.sqrt(np.sum(ms_error_image) /
                       np.sum(gt_layout_depth_img_mask))

        # Calc up to scale MSE
        if np.isnan(pred_layout_depth_med) or pred_layout_depth_med == 0:
            d_scale = 1.
        else:
            d_scale = gt_layout_depth_med / pred_layout_depth_med
        us_ms_error_image = (
            d_scale * pred_layout_depth_img - gt_layout_depth_img) ** 2
        us_rmse = np.sqrt(np.sum(us_ms_error_image) /
                          np.sum(gt_layout_depth_img_mask))

    return image_iou, image_pe, merror_edge, rmse, us_rmse, image_pe_hung