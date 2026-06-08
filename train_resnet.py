import os
import yaml
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# ---------------------------- 配置参数 ----------------------------
DATASET_PATH = "/home/ps/MyDataSets/UAV_military/new_yolo"
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")

BACKBONE = "resnet50"          # 可选 "resnet18" 或 "resnet50"
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
IMG_SIZE = (640, 640)          # 训练时统一缩放到该尺寸 (宽, 高)
CONF_THRESH = 0.05
NMS_THRESH = 0.5

PRETRAINED_WEIGHTS = "./ckpt/best_fasterrcnn_resnet50.pth"  # 设置路径，例如 "./ckpt/best_fasterrcnn_resnet50.pth"
EVAL_ONLY = True  # 是否仅评估
# -----------------------------------------------------------------

# 读取 YOLO data.yaml
with open(DATA_YAML, 'r') as f:
    data_cfg = yaml.safe_load(f)

train_img_dir = os.path.join(DATASET_PATH, data_cfg['train'])
val_img_dir = os.path.join(DATASET_PATH, data_cfg['val'])
label_dir = train_img_dir.replace('images', 'labels')
val_label_dir = val_img_dir.replace('images', 'labels')
if not os.path.exists(label_dir):
    label_dir = os.path.join(os.path.dirname(train_img_dir), 'labels')
    val_label_dir = os.path.join(os.path.dirname(val_img_dir), 'labels')

class_names = data_cfg['names']
num_classes = len(class_names) + 1     # +1 for background

# ---------------------------- 自定义数据集------------------------
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(640, 640), transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transforms = transforms

        # 筛选有效样本：标注文件存在且至少有一个有效目标
        self.img_files = []
        for f in os.listdir(img_dir):
            if not f.endswith(('.jpg', '.png', '.jpeg')):
                continue
            label_name = os.path.splitext(f)[0] + '.txt'
            label_path = os.path.join(label_dir, label_name)
            if not os.path.exists(label_path):
                print(f"Warning: No label file for {f}, skipping.")
                continue
            # 快速检查是否有有效目标（至少一行宽度高度>0）
            valid = False
            with open(label_path, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        w, h = float(parts[3]), float(parts[4])
                        if w > 0 and h > 0:
                            valid = True
                            break
            if not valid:
                print(f"Warning: No valid annotations in {label_path}, skipping image {f}.")
                continue
            self.img_files.append(f)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        if width <= 0 or height <= 0:
                            continue  # 跳过原始无效框
                        x1 = (x_center - width/2) * orig_w
                        y1 = (y_center - height/2) * orig_h
                        x2 = (x_center + width/2) * orig_w
                        y2 = (y_center + height/2) * orig_h
                        # 确保转换后仍然有效
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id + 1)  # COCO 类别从1开始

        if len(boxes) == 0:
            # 此情况理论上已经在初始化时过滤，但如果还是发生，返回空目标
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        # 应用数据增强
        if self.transforms:
            image, boxes, labels = self.transforms(image, boxes, labels)

        # 再次过滤变换后可能产生的无效框（例如取整后宽高为0）
        if len(boxes) > 0:
            valid_mask = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
            boxes = boxes[valid_mask]
            labels = labels[valid_mask]

        # 转为 tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.zeros(0, dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        return image, target

# ---------------------------- 自定义 transforms（同步处理 boxes）------------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, labels):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels

class Resize:
    """将图像缩放到固定尺寸，同时缩放边界框"""
    def __init__(self, size):
        self.size = size  # (w, h)

    def __call__(self, image, boxes, labels):
        h, w = image.shape[:2]
        new_w, new_h = self.size
        scale_x = new_w / w
        scale_y = new_h / h
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        return image, boxes, labels

class RandomHorizontalFlip:
    """随机水平翻转，同步翻转边界框"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
        if np.random.random() < self.p:
            image = cv2.flip(image, 1)
            if len(boxes) > 0:
                w = image.shape[1]
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1
        return image, boxes, labels

class ToTensor:
    """将图像转换为 tensor（CHW），并归一化到 [0,1]"""
    def __call__(self, image, boxes, labels):
        image = image.astype(np.float32) / 255.0
        # 此时保持 HWC，后续在 dataset 中转换为 CHW
        return image, boxes, labels

class Normalize:
    """标准化图像（要求输入图像为 numpy float32，值域 [0,1]）"""
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image, boxes, labels):
        image = (image - self.mean) / self.std
        return image, boxes, labels

# ---------------------------- 创建训练/验证 transforms ---------------------------
def get_transform(train=True):
    if train:
        return Compose([
            Resize(IMG_SIZE),
            RandomHorizontalFlip(0.5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return Compose([
            Resize(IMG_SIZE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ---------------------------- 创建模型（与之前相同）------------------------
def create_model(backbone_name, num_classes):
    if backbone_name == 'resnet18':
        base_backbone = torchvision.models.resnet18(pretrained=True)
        backbone_out_channels = 512
    elif backbone_name == 'resnet50':
        base_backbone = torchvision.models.resnet50(pretrained=True)
        backbone_out_channels = 2048
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # 提取特征提取部分（去掉最后的全连接层和池化层）
    feature_extractor = torch.nn.Sequential(*list(base_backbone.children())[:-2])

    # 包装 backbone 使其具有 out_channels 属性
    class BackboneWithOutChannels(torch.nn.Module):
        def __init__(self, backbone, out_channels):
            super().__init__()
            self.backbone = backbone
            self.out_channels = out_channels

        def forward(self, x):
            return self.backbone(x)

    backbone = BackboneWithOutChannels(feature_extractor, backbone_out_channels)

    # 生成 anchor
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       min_size=IMG_SIZE[1],
                       max_size=IMG_SIZE[0])

    # 替换预测头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ---------------------------- COCO 评估函数 ---------------------------
def evaluate(model, data_loader, device, conf_thresh=0.05, nms_thresh=0.5):
    model.eval()
    results = []
    gt_results = []
    image_ids = []

    with torch.no_grad():
        for idx, (images, targets) in tqdm(enumerate(data_loader)):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                h_img, w_img = images[i].shape[1], images[i].shape[2]

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                keep = scores >= conf_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    results.append({
                        'image_id': image_id,
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(score),
                        'category_id': int(label)
                    })

                # 收集 ground truth
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    x1, y1, x2, y2 = gt_box
                    w = x2 - x1
                    h = y2 - y1
                    gt_results.append({
                        'image_id': image_id,
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'category_id': int(gt_label),
                        'area': float(w * h),
                        'iscrowd': 0
                    })
                image_ids.append(image_id)

    if len(results) == 0:
        print("No predictions found, returning mAP=0")
        return 0.0

    # 构建 COCO 格式的 ground truth
    coco_gt = COCO()
    coco_gt.dataset = {
        'images': [{'id': id} for id in set(image_ids)],
        'annotations': [{'id': i, **ann} for i, ann in enumerate(gt_results)],
        'categories': [{'id': i, 'name': name} for i, name in enumerate(class_names, start=1)]
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results) if results else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = list(set(image_ids))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # mAP = coco_eval.stats[0]   # mAP@0.5:0.95
    return coco_eval

# ---------------------------- 训练一个 epoch ---------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    # 使用 tqdm 并设置初始描述
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_loss = losses.item()
        total_loss += batch_loss
        # 更新进度条显示当前 batch 损失和平均损失
        progress_bar.set_postfix(loss=batch_loss, avg_loss=total_loss/(batch_idx+1))
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch} - Training loss: {avg_loss:.4f}")
    return avg_loss

# ---------------------------- 主程序 ---------------------------
def main():
    print(f"Using device: {DEVICE}")
    # 数据集
    print("Preparing datasets and dataloaders...")
    train_dataset = YOLODataset(train_img_dir, label_dir, img_size=IMG_SIZE, transforms=get_transform(train=True))
    val_dataset = YOLODataset(val_img_dir, val_label_dir, img_size=IMG_SIZE, transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=NUM_WORKERS)

    print(f"Preparing model {BACKBONE}...")
    model = create_model(BACKBONE, num_classes)
    model.to(DEVICE)

    # 打印参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {n_parameters}')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # ================= 加载预训练模型（可选） =================
    if PRETRAINED_WEIGHTS and os.path.exists(PRETRAINED_WEIGHTS):
        print(f"Loading checkpoint from {PRETRAINED_WEIGHTS}")
        checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("Weights loaded successfully.")
        if EVAL_ONLY:
            print("Evaluation only mode. Evaluating on validation set...")
            # 确保 evaluate 返回 (coco_eval, val_loss)
            coco_eval = evaluate(model, val_loader, DEVICE, CONF_THRESH, NMS_THRESH)
            print(f"Validation mAP: {coco_eval.stats[0]:.4f}")
            return
    elif PRETRAINED_WEIGHTS:
        print(f"Warning: Pretrained weights file {PRETRAINED_WEIGHTS} not found. Training from scratch.")
    # =========================================================

    os.makedirs("./ckpt", exist_ok=True)
    model_save_path = f"./ckpt/best_fasterrcnn_{BACKBONE}.pth"

    # 初始化记录列表
    train_losses = []
    val_maps = []
    best_map = 0.0

    # 创建图形
    # plt.ion()  # 交互模式，可选
    fig, ax = plt.subplots(figsize=(8, 5))

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        lr_scheduler.step()
        train_losses.append(avg_loss)

        print(f"Evaluating on validation set after epoch {epoch}...")
        coco_eval = evaluate(model, val_loader, DEVICE, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH)
        current_map = coco_eval.stats[0]  # mAP@0.5:0.95
        val_maps.append(current_map)

        # 保存最佳模型
        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), f"./ckpt/best_fasterrcnn_{BACKBONE}_{epoch:03d}.pth")
            print(f"New best model saved with mAP: {best_map:.4f}")
        print(f"Best mAP so far: {best_map:.4f}\n")

        # 绘制并保存损失和 mAP 曲线
        ax.clear()
        ax.plot(train_losses, label='Training Loss', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.legend(loc='upper left')

        ax2 = ax.twinx()
        ax2.plot(val_maps, label='Validation mAP', color='tab:orange', marker='s')
        ax2.set_ylabel('mAP', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.legend(loc='upper right')

        plt.title('Training Loss and Validation mAP')
        plt.tight_layout()
        plt.savefig('./ckpt/training_curve_{}.png'.format(BACKBONE), dpi=150)
        plt.pause(0.1)  # 更新显示（如果 plt.ion())

    print(f"Training finished. Best mAP: {best_map:.4f}. Model saved at {model_save_path}")

if __name__ == "__main__":
    main()

