"""
Demo code

Example usage:

python3 tools/demo.py configs/smpl/tune.py ./demo/raw_teaser.png --ckpt /path/to/model
"""
# from copyreg import pickle
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn

import argparse
import os
import os.path as osp
import sys
import cv2
import numpy as np
from mmdet.models.utils.smpl_utils import perspective_projection
from torchvision import transforms

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
import pickle
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


def prepare_dump(pred_results, img, render, bbox_results, FOCAL_LENGTH):
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

    total_img = np.zeros((3 * H, W, 3))
    total_img[:H] += img
    total_img[H:2 * H] += fv_rendered.transpose([1, 2, 0])
    total_img[2 * H:] += bv_rendered.transpose([1, 2, 0])
    total_img = (total_img * 255).astype(np.uint8)
    return total_img

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--image_folder', help='Path to folder with images')
    parser.add_argument('--output_folder', default='model_output', help='Path to save results')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--color_jitter', action="store_true")
    parser.add_argument('--vis', action="store_true")
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

    pseudo_labels = []
    aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(0.15, 0.15, 0, 0.05),
                transforms.ToTensor(),
            ])
    with torch.no_grad():
        folder_name = args.image_folder
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        images = []
        for dirPath, dirNames, fileNames in os.walk(folder_name):
            for f in fileNames:
                images.append(os.path.join(dirPath.split("/")[-2], dirPath.split("/")[-1], f))
        cnt = 0
        for image in images:
            cnt += 1
            if cnt % 10 == 0:
                print(f"{cnt} / {len(images)}")
            
            file_name = osp.join(folder_name, image)
            img = cv2.imread(file_name)
            ori_shape = img.shape
            img_scale = (img.shape[1], img.shape[0])
            img_copy = img.copy()
            img, img_shape, pad_shape, scale_factor = img_transform(img, img_scale)

            data_batch = dict(
                    img=DC([to_tensor(img[None, ...])], stack=True),
                    img_meta=DC([{'img_shape':img_shape, 'scale_factor':scale_factor, 'flip':False, 'ori_shape':ori_shape}], cpu_only=True),
                    )
            bbox_results, pred_results = model(**data_batch, return_loss=False)
            if len(bbox_results[0]) == 0:
                continue
            
            # color jitter augmentation
            if args.color_jitter:
                lens = [len(bbox_results[0])]
                inliers = 0
                for i in range(8):
                    img = np.uint8(aug(img_copy).permute(1, 2, 0).numpy() * 255)
                    img, img_shape, pad_shape, scale_factor = img_transform(img, img_scale)
                    data_batch = dict(
                            img=DC([to_tensor(img[None, ...])], stack=True),
                            img_meta=DC([{'img_shape':img_shape, 'scale_factor':scale_factor, 'flip':False, 'ori_shape':ori_shape}], cpu_only=True),
                            )
                    bbox_results_, pred_results_ = model(**data_batch, return_loss=False)
                    
                    lens.append(len(bbox_results_[0]))
                    if len(bbox_results[0]) == len(bbox_results_[0]):
                        inliers += 1
                
                if cnt % 10 == 0:
                    print(lens, np.std(lens))
                if np.std(lens) > 0.42 or inliers < 6:
                    continue
            
            # pseudo label
            pred_joints = pred_results["pred_joints"]
            pred_camera = pred_results["pred_camera"]
            pred_bboxes = torch.tensor(bbox_results[0][..., :-1]).to(pred_joints.device)
            batch_size = pred_joints.shape[0]
            img_size = torch.zeros(batch_size, 2).to(pred_joints.device)
            img_size += torch.tensor(img.shape[:-3:-1], dtype=img_size.dtype).to(img_size.device)
            bboxes_size = torch.max(torch.abs(pred_bboxes[..., 0] - pred_bboxes[..., 2]),
                                torch.abs(pred_bboxes[..., 1] - pred_bboxes[..., 3]))
            depth = 2 * FOCAL_LENGTH / (1e-6 + pred_camera[..., 0] * bboxes_size)
            rotation_Is = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_joints.device)
            translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(
            pred_joints.device)
            center_pts = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2
            translation[:, :-1] = depth[:, None] * (center_pts + pred_camera[:, 1:] * bboxes_size.unsqueeze(-1) - img_size / 2) / FOCAL_LENGTH
            translation[:, -1] = depth
            focal_length = FOCAL_LENGTH
            
            pred_keypoints_2d_smpl = perspective_projection(pred_joints,
                                             rotation_Is,
                                             translation,
                                             focal_length,
                                             img_size / 2)
            
            pred_keypoints_2d_smpl = torch.cat([pred_keypoints_2d_smpl, torch.ones_like(pred_keypoints_2d_smpl[:, :, 0:1])], dim=-1)
            pred_keypoints_3d_smpl = torch.cat([pred_results["pred_joints"], torch.ones_like(pred_results["pred_joints"][:, :, 0:1])], dim=-1)
            pseudo_label = {
                'filename': "imgs/" + image,
                'width': ori_shape[1], 
                'height': ori_shape[0], 
                'bboxes': pred_bboxes.cpu().numpy(), 
                'kpts3d': pred_keypoints_3d_smpl.cpu().numpy(), 
                'kpts2d': pred_keypoints_2d_smpl.cpu().numpy()
            }
            
            pseudo_labels.append(pseudo_label)
            
            # draw box keypoints
            if args.vis and cnt % 10 == 0 and pred_results is not None:
                pred_results['bboxes'] = bbox_results[0]
                img = denormalize(img)
                img_bbox = img * 255
                for bbox in pred_bboxes:
                    img_bbox = cv2.rectangle(img_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                for person in pred_keypoints_2d_smpl:
                    for k in person:
                        img_bbox = cv2.circle(img_bbox, (k[0], k[1]), 2, (255, 0, 0), 2)
                out_path = output_folder + "/" + file_name.split("/")[-3] + "/" +file_name.split("/")[-2]
                Path(out_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f'{file_name.replace(folder_name, output_folder)}.output.jpg', np.uint8(img_bbox[:, :, ::-1]))
                
        print("# images with pseudo labels:", len(pseudo_labels))
        Path('data/pseudo/annotations/').mkdir(parents=True, exist_ok=True)
        name = "anno"
        if args.color_jitter:
            name = "anno_jitter"
        with open(f"data/pseudo/annotations/{name}.pkl", "wb") as f:
            pickle.dump(pseudo_labels, f)

if __name__ == '__main__':
    main()
