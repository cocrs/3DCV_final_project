"""
Demo code

Example usage:

python3 tools/our_demo.py --config=configs/smpl/tune.py --image_folder=data/Panoptic --output_folder=results/ --ckpt /path/to/model --annotation=data/Panoptic/processed/annotations/160906_pizza1.pkl
"""
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn

import argparse
import os
import os.path as osp
import sys
import cv2
import numpy as np
import pickle

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_PATH)

from mmcv import Config
from mmcv.runner import Runner

from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDataParallel
from mmdet.apis.train import build_optimizer
from mmdet.models.utils.smpl.renderer import Renderer
from mmdet import __version__
from mmdet.models import build_detector
from mmdet.datasets.transforms import ImageTransform
from mmdet.datasets.utils import to_tensor
from mmdet.models.utils.smpl_utils import perspective_projection
from pathlib import Path

denormalize = lambda x: x.transpose([1, 2, 0]) * np.array([0.229, 0.224, 0.225])[None, None, :] + \
                        np.array([0.485, 0.456, 0.406])[None, None,]

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

def renderer_bv(img_t, verts_t, trans_t, bboxes_t, focal_length, render):
    R_bv = torch.zeros(3, 3)
    R_bv[0, 0] = R_bv[2, 1] = 1
    R_bv[1, 2] = -1
    bbox_area = (bboxes_t[:, 2] - bboxes_t[:, 0]) * (bboxes_t[:, 3] - bboxes_t[:, 1])
    area_mask = torch.tensor(bbox_area > bbox_area.max() * 0.05)
    verts_t, trans_t = verts_t[area_mask], trans_t[area_mask]
    verts_t = verts_t + trans_t.unsqueeze(1)
    verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
    verts_tfar = verts_tr  # verts_tr + trans_t.unsqueeze(1)
    p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
    p_center = 0.5 * (p_min + p_max)
    # trans_tr = torch.einsum('bj,kj->bk', trans_t, R_bv)
    verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)

    dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (
            verts_tfar.view(-1, 3) - p_center).max(0)[0]
    h, w = img_t.shape[-2:]
    # h, w = min(h, w), min(h, w)
    ratio_max = abs(0.9 - 0.5)
    z_x = dis_max[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
    z_y = dis_max[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
    z_x_0 = (-dis_min[0]) * focal_length / (ratio_max * w) + torch.abs(
        dis_min[2])
    z_y_0 = (-dis_min[1]) * focal_length / (ratio_max * h) + torch.abs(
        dis_min[2])
    z = max(z_x, z_y, z_x_0, z_y_0)
    verts_right = verts_tfar - p_center + torch.tensor([0, 0, z])
    img_right = render([torch.ones_like(img_t)], [verts_right],
                       translation=[torch.zeros_like(trans_t)])
    return img_right[0]


def prepare_dump(gt_results, pred_results, img, render, bbox_results, FOCAL_LENGTH):
    verts = pred_results['pred_vertices'] + pred_results['pred_translation'][:, None]
    # 'pred_rotmat', 'pred_betas', 'pred_camera', 'pred_vertices', 'pred_joints', 'pred_translation', 'bboxes'
    pred_trans = pred_results['pred_translation'].cpu()
    pred_camera = pred_results['pred_camera'].cpu()
    pred_betas = pred_results['pred_betas'].cpu()
    pred_rotmat = pred_results['pred_rotmat'].cpu()
    pred_verts = pred_results['pred_vertices'].cpu()
    bboxes = pred_results['bboxes']
    img_bbox = img.copy()
    for bbox in bboxes:
        img_bbox = cv2.rectangle(img_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    img_th = torch.tensor(img_bbox.transpose([2, 0, 1]))
    _, H, W = img_th.shape
    try:
        fv_rendered = render([img_th.clone()], [pred_verts], translation=[pred_trans])[0]
        bv_rendered = renderer_bv(img_th, pred_verts, pred_trans, bbox_results[0], FOCAL_LENGTH, render)
    except Exception as e:
        print(e)
        return None
    
    # keypoint
    img_key_pred_bbox = img_bbox.copy()
    pred_camera = pred_results['pred_camera']
    pred_joints = pred_results["pred_joints"]
    pred_bboxes = torch.tensor(bbox_results[0][..., :-1]).to(pred_joints.device)
    batch_size = pred_joints.shape[0]
    img_size = torch.zeros(batch_size, 2).to(pred_joints.device)
    img_size += torch.tensor(img.shape[:-3:-1], dtype=img_size.dtype).to(img_size.device)
    bboxes_size = torch.max(torch.abs(pred_bboxes[..., 0] - pred_bboxes[..., 2]),
                        torch.abs(pred_bboxes[..., 1] - pred_bboxes[..., 3]))
    depth = 2 * FOCAL_LENGTH / (1e-6 + pred_camera[..., 0] * bboxes_size)
    rotation_Is = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_joints.device)
    translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(pred_joints.device)
    center_pts = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2
    translation[:, :-1] = depth[:, None] * (center_pts + pred_camera[:, 1:] * bboxes_size.unsqueeze(-1) - img_size / 2) / FOCAL_LENGTH
    translation[:, -1] = depth
    pred_keypoints_2d_smpl = perspective_projection(pred_joints,
                                        rotation_Is,
                                        translation,
                                        FOCAL_LENGTH,
                                        img_size / 2)
    pred_keypoints_2d_smpl = torch.cat([pred_keypoints_2d_smpl, torch.ones_like(pred_keypoints_2d_smpl[:, :, 0:1])], dim=-1)
    for person in pred_keypoints_2d_smpl:
        for k in person:
            img_key_pred_bbox = cv2.circle(img_key_pred_bbox, (k[0], k[1]), 2, (1, 0, 0), 2)
    # connects = [[0, 1], [0, 2], [0, 3], [0,9], [3, 4], [4,5], [9, 10], [10, 11], [2, 6], [6, 7], [7, 8], [2, 12], [12, 13], [13, 14]]
    # body_edges = np.array(
    #     [[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9], [3, 13], [13, 14], [14, 15], [1, 10], [10, 11],
    #     [11, 12]]) - 1
    # connects = body_edges.tolist()
    # Panoptic_to_J15 = [12, 13, 14, 9, 10, 11,  # 5s
    #                    3, 4, 5, 8, 7, 6,  # 11
    #                    2, 1, 0
    #                    ]
    # new_connects = []
    # for c in connects:
    #     new_connects.append([Panoptic_to_J15[c[0]], Panoptic_to_J15[c[1]]])
    # print(new_connects)
    # connects = [[12, 13], [12, 14], [12, 9], [12, 8], [9, 10], [10, 11], [8, 7], [7, 6], [14, 3], [3, 4], [4, 5], [14, 2], [2, 1], [1, 0]]
    # connects = [[12, 13], [12, 9], [9, 10], [10, 11], [12, 14], [14, 3], [3, 4], [4, 5], [14, 2], [2, 1], [1, 0], [12, 8], [8, 7], [7, 6]]
    connects = [[12, 9], [9, 10], [10, 11], [12, 14], [14, 3], [3, 4], [4, 5], [14, 2], [2, 1], [1, 0], [12, 8], [8, 7], [7, 6]]
    for person in pred_keypoints_2d_smpl:
        for k in connects:
            img_key_pred_bbox = cv2.line(img_key_pred_bbox, (person[k[0]][0], person[k[0]][1]), (person[k[1]][0], person[k[1]][1]), (0, 0, 1), 1)

    img_key_gt_box = img_bbox.copy()
    gt_keypoints_2d_smpl = gt_results['kpts2d']
    for person in gt_keypoints_2d_smpl:
        for k in person:
            img_key_gt_box = cv2.circle(img_key_gt_box, (int(np.round(k[0])), int(np.round(k[1]))), 2, (1, 0, 0), 2)
    for person in gt_keypoints_2d_smpl:
        for k in connects:
            img_key_gt_box = cv2.line(img_key_gt_box, (int(np.round(person[k[0]][0])), int(np.round(person[k[0]][1]))), (int(np.round(person[k[1]][0])), int(np.round(person[k[1]][1]))), (0, 0, 1), 1)

    total_img = np.zeros((5 * H, W, 3))
    total_img[:H] += img
    total_img[H:2*H] += img_key_pred_bbox
    total_img[2*H : 3*H] += img_key_gt_box
    total_img[3*H:4 * H] += fv_rendered.transpose([1, 2, 0])
    total_img[4 * H:] += bv_rendered.transpose([1, 2, 0])
    total_img = (total_img * 255).astype(np.uint8)
    return total_img

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--image_folder', help='Path to folder with images')
    parser.add_argument('--output_folder', default='model_output', help='Path to save results')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--annotation', type=str, help='Path to annotation file', default='')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.ckpt:
        cfg.resume_from = args.ckpt

    cfg.test_cfg.rcnn.score_thr = 0.5

    FOCAL_LENGTH = cfg.get('FOCAL_LENGTH', 1000)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=('Human',))
    # add an attribute for visualization convenience
    model.CLASSES = ('Human',)

    model = MMDataParallel(model, device_ids=[0]).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = Runner(model, lambda x: x, optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.resume(cfg.resume_from)
    model = runner.model
    model.eval()
    render = Renderer(focal_length=FOCAL_LENGTH)
    img_transform = ImageTransform(
            size_divisor=32, **img_norm_cfg)
    img_scale = cfg.common_val_cfg.img_scale

    image_gt = dict()    
    with open(args.annotation, 'rb') as f:
        data = pickle.load(f)
    for d in data:
        image_gt[d['filename'].replace("png", "jpg")] = d

    with torch.no_grad():
        folder_name = args.image_folder
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        images = os.listdir(folder_name)

        # for image in images:
        for image in image_gt.keys():
            file_name = osp.join(folder_name, image)
            img = cv2.imread(file_name)
            ori_shape = img.shape

            img_scale = (ori_shape[1], ori_shape[0])
            img, img_shape, pad_shape, scale_factor = img_transform(img, img_scale)

            # Force padding for the issue of multi-GPU training
            # padded_img = np.zeros((img.shape[0], img_scale[1], img_scale[0]), dtype=img.dtype)
            # padded_img[:, :img.shape[-2], :img.shape[-1]] = img
            # img = padded_img

            # assert img.shape[1] == 512 and img.shape[2] == 832, "Image shape incorrect"

            data_batch = dict(
                img=DC([to_tensor(img[None, ...])], stack=True),
                img_meta=DC([{'img_shape':img_shape, 'scale_factor':scale_factor, 'flip':False, 'ori_shape':ori_shape}], cpu_only=True),
                )
            bbox_results, pred_results = model(**data_batch, return_loss=False)

            if pred_results is not None:
                pred_results['bboxes'] = bbox_results[0]
                gt_results = image_gt[image]

                img = denormalize(img)
                img_viz = prepare_dump(gt_results, pred_results, img, render, bbox_results, FOCAL_LENGTH)
                os.makedirs(file_name[:-19].replace(folder_name, output_folder), exist_ok=True)
                cv2.imwrite(f'{file_name.replace(folder_name, output_folder)}.output.jpg', img_viz[:, :, ::-1])

if __name__ == '__main__':
    main()
