import os
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms as T
from train_resnet import create_model   # 引入您原有的模型创建函数

# ================= 配置 =================
BACKBONE = "resnet50"          # 可选 "resnet18" 或 "resnet50"
DATA_YAML = "/home/ps/MyDataSets/UAV_military/new_yolo/data.yaml"
WEIGHT_PATH = "./ckpt/best_fasterrcnn_{}.pth".format(BACKBONE)   # 替换为实际权重路径
TEST_IMAGE = "/home/ps/MyDataSets/UAV_military/new_yolo/images/test/000541.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (640, 640)        # 必须与训练时的 IMG_SIZE 一致
CONF_THRESH = 0.5            # 置信度阈值
# =========================================

# 读取 data.yaml 获取类别名
with open(DATA_YAML, 'r') as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg['names']      # 例如 ['person', 'car', ...]
num_classes = len(class_names) + 1   # +1 背景

# 定义图像预处理（与验证 transform 一致：Resize -> ToTensor -> Normalize）
def get_test_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize(IMG_SIZE),           # (height, width) 实际为 (640,640)
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

# 加载模型
def load_model(weight_path):
    model = create_model(BACKBONE, num_classes)   # backbone 必须与训练时一致
    checkpoint = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {weight_path}")
    return model

# 将检测框从 resize 后的坐标映射回原始图像坐标
def scale_boxes(boxes, orig_size, resize_size):
    """
    boxes: tensor of shape (N, 4) in (x1, y1, x2, y2) format, 坐标位于 resize 后的图像
    orig_size: (orig_w, orig_h)
    resize_size: (resize_w, resize_h)
    """
    orig_w, orig_h = orig_size
    resize_w, resize_h = resize_size
    scale_x = orig_w / resize_w
    scale_y = orig_h / resize_h
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes

# 主函数
def main():
    # 1. 加载模型
    model = load_model(WEIGHT_PATH)

    # 2. 读取原始图像
    orig_img = cv2.imread(TEST_IMAGE)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_img.shape[:2]

    # 3. 预处理
    transform = get_test_transform()
    img_tensor = transform(orig_img).unsqueeze(0).to(DEVICE)   # (1, C, H, W)

    # 4. 推理
    with torch.no_grad():
        predictions = model(img_tensor)   # predictions 是一个 list，每个元素是 dict: boxes, scores, labels

    # 5. 解析预测结果（取第一张图）
    pred = predictions[0]
    boxes = pred['boxes'].cpu()
    scores = pred['scores'].cpu()
    labels = pred['labels'].cpu()

    # 6. 置信度过滤
    keep = scores >= CONF_THRESH
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # 7. 将框坐标映射回原图尺寸
    boxes_orig = scale_boxes(boxes, (orig_w, orig_h), IMG_SIZE)

    # 8. 可视化
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(orig_img)
    for box, score, label in zip(boxes_orig, scores, labels):
        x1, y1, x2, y2 = box.numpy()
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        class_name = class_names[label-1]   # label 从 1 开始
        ax.text(x1, y1-5, f'{class_name} {score:.2f}',
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=10, color='white')

    ax.set_title(f'Detection Results (Confidence >= {CONF_THRESH})')
    ax.axis('off')
    plt.tight_layout()
    # 保存结果图片
    out_path = "test_{}_result.jpg".format(BACKBONE)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Result saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()

