import os
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms
from PIL import Image
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO
import hashlib
import pickle

# ==================== 1. YOLO 特征提取函数 ====================
def extract_features_with_detections_gt(yolo_model, img_path, use_gt=False, gt_boxes=None, gt_classes=None,
                                        gt_class_names=None):
    """
    使用 YOLO 模型提取目标检测框、类别、RoI 特征（全局平均池化）。
    若 use_gt=True，则根据提供的真值框提取特征（忽略 YOLO 检测结果）。

    Args:
        yolo_model: YOLO 模型实例
        img_path: 图像路径
        use_gt: 是否使用真值框提取特征
        gt_boxes: 真值框列表，每个框为 [x1, y1, x2, y2] 绝对坐标（像素）
        gt_classes: 真值类别 ID 列表（整数索引）
        gt_class_names: 真值类别名称列表（字符串），可选
    Returns:
        detections: list of dict，每个 dict 包含 'bbox', 'conf', 'cls_id', 'cls_name', 'feature'
    """
    img0 = cv2.imread(img_path)
    if img0 is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    orig_shape = img0.shape[:2]  # (h, w)

    # 注册钩子获取中间层特征图（backbone 最后一层）
    features = []
    def hook_fn(module, inp, out):
        features.append(out.detach())
    target_layer = yolo_model.model.model[-2]   # YOLOv8/v11 特征层
    handle = target_layer.register_forward_hook(hook_fn)

    # YOLO 推理（同时触发钩子）
    results = yolo_model(img_path, verbose=False)
    handle.remove()
    feat_map = features[0]      # (1, C, Hf, Wf)

    detections = []

    if use_gt and gt_boxes is not None and len(gt_boxes) > 0:
        # 使用真值框提取特征
        h_orig, w_orig = orig_shape
        h_feat, w_feat = feat_map.shape[-2:]
        scale_h = h_feat / h_orig
        scale_w = w_feat / w_orig

        boxes_abs = []
        for box in gt_boxes:
            cx, cy, w, h = box
            x1 = (cx - w / 2) * w_orig
            y1 = (cy - h / 2) * h_orig
            x2 = (cx + w / 2) * w_orig
            y2 = (cy + h / 2) * h_orig
            boxes_abs.append([x1, y1, x2, y2])
        gt_boxes_ = np.array(boxes_abs)

        boxes = np.array(gt_boxes_) if not isinstance(gt_boxes_, np.ndarray) else gt_boxes_
        if gt_classes is None:
            gt_classes = [-1] * len(boxes)
        if gt_class_names is None:
            gt_class_names = [f"class_{c}" for c in gt_classes]

        for box, cls_id, cls_name in zip(boxes, gt_classes, gt_class_names):
            x1, y1, x2, y2 = box
            fx1 = int(x1 * scale_w)
            fy1 = int(y1 * scale_h)
            fx2 = int(x2 * scale_w)
            fy2 = int(y2 * scale_h)
            fx1 = max(0, min(fx1, w_feat-1))
            fx2 = max(0, min(fx2, w_feat-1))
            fy1 = max(0, min(fy1, h_feat-1))
            fy2 = max(0, min(fy2, h_feat-1))
            if fx2 <= fx1 or fy2 <= fy1:
                continue

            roi = feat_map[0, :, fy1:fy2, fx1:fx2]          # (C, Hr, Wr)
            feat_vec = roi.mean(dim=[1, 2]).cpu().numpy()   # (C,)

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'conf': 1.0,
                'cls_id': int(cls_id),
                'cls_name': cls_name,
                'feature': feat_vec
            })
    else:
        # 使用 YOLO 检测结果
        det = results[0].boxes
        if det is not None:
            h_orig, w_orig = orig_shape
            h_feat, w_feat = feat_map.shape[-2:]
            scale_h = h_feat / h_orig
            scale_w = w_feat / w_orig

            boxes = det.xyxy.cpu().numpy()
            confs = det.conf.cpu().numpy()
            clses = det.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(boxes, confs, clses):
                x1, y1, x2, y2 = box
                fx1 = int(x1 * scale_w)
                fy1 = int(y1 * scale_h)
                fx2 = int(x2 * scale_w)
                fy2 = int(y2 * scale_h)
                fx1 = max(0, min(fx1, w_feat-1))
                fx2 = max(0, min(fx2, w_feat-1))
                fy1 = max(0, min(fy1, h_feat-1))
                fy2 = max(0, min(fy2, h_feat-1))
                if fx2 <= fx1 or fy2 <= fy1:
                    continue

                roi = feat_map[0, :, fy1:fy2, fx1:fx2]
                feat_vec = roi.mean(dim=[1, 2]).cpu().numpy()

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': float(conf),
                    'cls_id': int(cls),
                    'cls_name': yolo_model.names[cls],
                    'feature': feat_vec
                })

    return detections

def extract_features_with_detections(yolo_model, img_path, args):
    """
    使用 YOLO 模型提取目标检测框、类别、RoI 特征（全局平均池化）。
    返回: list of dict，每个 dict 包含 'bbox', 'conf', 'cls_id', 'cls_name', 'feature'
    """
    img0 = cv2.imread(img_path)
    if img0 is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    orig_shape = img0.shape[:2]  # (h, w)

    # 注册钩子获取中间层特征图（backbone 最后一层）
    features = []
    def hook_fn(module, inp, out):
        features.append(out.detach())
    target_layer = yolo_model.model.model[-2]   # YOLOv8/v11 特征层
    handle = target_layer.register_forward_hook(hook_fn)

    # YOLO 推理（同时触发钩子）
    results = yolo_model(img_path, verbose=False)
    handle.remove()
    feat_map = features[0]      # (1, C, Hf, Wf)

    det = results[0].boxes
    detections = []
    if det is not None:
        h_orig, w_orig = orig_shape
        h_feat, w_feat = feat_map.shape[-2:]
        scale_h = h_feat / h_orig
        scale_w = w_feat / w_orig

        boxes = det.xyxy.cpu().numpy()
        confs = det.conf.cpu().numpy()
        clses = det.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(boxes, confs, clses):
            x1, y1, x2, y2 = box
            # 映射到特征图坐标
            fx1 = int(x1 * scale_w)
            fy1 = int(y1 * scale_h)
            fx2 = int(x2 * scale_w)
            fy2 = int(y2 * scale_h)
            fx1 = max(0, min(fx1, w_feat-1))
            fx2 = max(0, min(fx2, w_feat-1))
            fy1 = max(0, min(fy1, h_feat-1))
            fy2 = max(0, min(fy2, h_feat-1))
            if fx2 <= fx1 or fy2 <= fy1:
                continue

            roi = feat_map[0, :, fy1:fy2, fx1:fx2]          # (C, Hr, Wr)
            feat_vec = roi.mean(dim=[1, 2]).cpu().numpy()   # (C,)

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'conf': float(conf),
                'cls_id': int(cls),
                'cls_name': yolo_model.names[cls],
                'feature': feat_vec
            })
    return detections


# ==================== 2. 数据集（直接使用 YOLO 提取特征） ====================
class UAVMilitaryDataset(Dataset):
    def __init__(self, args, split, yolo_model, device='cuda'):
        """
        yolo_model: 已加载的 YOLO 模型实例
        """
        # 数据集路径
        ann_file = os.path.join(args.data_root, "annotations.json")
        split_file = os.path.join(args.data_root, "dataset_spilt.json")

        self.img_folder = os.path.join(args.data_root, "images")
        self.yolo_model = yolo_model
        self.device = device

        with open(split_file, 'r') as f:
            split_data = json.load(f)
        self.img_ids = split_data[split]

        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)

        # ---------- 新增：过滤掉既无关系也无意图的样本 ----------
        filtered_ids = []
        for img_id in self.img_ids:
            ann = self.annotations.get(img_id, {})
            has_relations = len(ann['relations']['label_id']) > 0
            has_intents = len(ann['intents']['label_id']) > 0
            has_entities = len(ann['entity']['label_id']) > 1 # 至少2个实体
            if has_entities:
                if has_relations or has_intents:
                    filtered_ids.append(img_id)
            # else: 跳过该样本
        self.img_ids = filtered_ids
        print(f"[{split}] 原始样本数: {len(self.img_ids)}, 过滤后: {len(self.img_ids)} (仅保留有关注或意图的)")


        # 预提取特征并缓存到内存（避免每次训练重复提取）
        # self.cache = {}
        # print(f"\nPre-extracting YOLO features for {split} split...")
        # for img_id in tqdm(self.img_ids):
        #     img_name = self.annotations[img_id]['image_name']
        #     img_path = os.path.join(img_folder, img_name)
        #     detections = extract_features_with_detections(yolo_model, img_path, device)
        #     self.cache[img_id] = detections
        #
        # 缓存文件路径（基于 ann_file 和 split 的唯一标识）
        cache_dir = os.path.join(args.data_root, '.yolo_cache')
        os.makedirs(cache_dir, exist_ok=True)
        ann_path = os.path.abspath(ann_file)
        cache_key = hashlib.md5(f"{ann_path}_{split}".encode()).hexdigest()
        cache_file = os.path.join(cache_dir,
                                  f"{args.detect_model}_use_gt_{cache_key}.pkl") if args.use_gt else os.path.join(
            cache_dir, f"{args.detect_model}_not_gt_{cache_key}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached YOLO features from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            print(f"\nPre-extracting YOLO features for {split} split...")
            self.cache = {}
            for img_id in tqdm(self.img_ids):
                img_name = self.annotations[img_id]['image_name']
                img_path = os.path.join(self.img_folder, img_name)

                ann = self.annotations[img_id]
                if args.use_gt:
                    # 假设 target 中包含真值框和类别（绝对坐标）
                    detections = extract_features_with_detections_gt(
                        yolo_model, img_path, use_gt=True,
                        gt_boxes=np.array(ann['entity']['bbox']),
                        gt_classes=np.array(ann['entity']['label_id']),
                        gt_class_names=np.array([self.yolo_model.names[c] for c in ann['entity']['label_id']])
                    )
                else:
                    detections = extract_features_with_detections_gt(yolo_model, img_path, use_gt=False)
                # detections = extract_features_with_detections(yolo_model, img_path, args)
                self.cache[img_id] = detections
            # 保存缓存
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Saved YOLO features to {cache_file}")

        self._extract_categories()  # 提取类别映射

    def _extract_categories(self):
        """从标注中提取关系、意图和场景的类别映射"""
        # 构建类别映射（从标注数据中自动提取关系、意图、场景的类别）
        self.rel_categories = ["behind", "in_front_of", "next_to", "hiding_behind", "concealed_in",
            "surrounding", "inside", "moving_along", "occupying", "passing_through", "on",
            "following", "above", "below", "under", "over", "bordering", "cutting_through",
            "part_of", "reconnoitering", "crossing", "along"]  # 22 {rel_label_id: category_name}
        self.intent_categories = ["assault", "reconnoitering", "infantry_tank_collaboration", "defense",
            "recon", "supply", "march", "rest", "assemble", "withdraw", "ambush",
            "moving_along", "hiding_behind", "occupying", "concealed_in"]  # 15 {intent_label_id: category_name}
        self.scene_categories = ["urban", "plain", "marsh", "mountain", "jungle"] # {scene_label_id: category_name}

        # 类别定义
        self.entity_categories = [
            "tank", "building", "explosion", "soldiergroup", "soldier", "truck",
            "trench", "road", "bridge", "tree", "forest", "river", "hill"
        ]  # 13个


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann = self.annotations[img_id]
        detections = self.cache[img_id]

        # 构建模型需要的输入
        num_objs = len(detections)
        if num_objs > 0:
            boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
            labels = torch.tensor([d['cls_id'] for d in detections], dtype=torch.int64)
            feature_list = [d['feature'] for d in detections]
            features = torch.from_numpy(np.array(feature_list)).float()
        else:
            # 返回空张量
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            features = torch.zeros((0, 256), dtype=torch.float32)  # 特征维度需与模型匹配


        # 关系三元组和意图标注
        rel_annotations = torch.tensor(ann['relations']['triplet'], dtype=torch.int64)
        intent_annotations = torch.tensor(ann['intents']['triplet'], dtype=torch.int64)
        scene_label = torch.tensor(ann['scene']['label_id'], dtype=torch.int64) if 'scene' in ann else torch.tensor(-1)

        target = {
            'image_id': img_id,
            'boxes': boxes,
            'labels': labels,
            'features': features,          # RoI 特征向量
            'rel_annotations': rel_annotations,
            'intent_annotations': intent_annotations,
            'scene': scene_label,
            'orig_size': torch.tensor(ann['image_size'], dtype=torch.int64),  # 无实际意义，保留兼容
        }
        return target   # 不返回图像，仅返回目标信息

    def get_entity_categories(self):
        return self.yolo_model.names  # YOLO 的类别名称列表


# ==================== 4. 训练与评估循环 ====================
def collate_fn(batch):
    """直接返回 batch 列表，因为每个样本已是 dict"""
    return batch
