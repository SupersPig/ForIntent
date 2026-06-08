import torch
import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from model import UAVSceneGraphModel  # 确保能正确导入

# ================== 特征提取（与训练时保持一致）==================
def extract_features_with_detections(yolo_model, img_path, device='cuda'):
	"""
	完全复用你提供的训练用特征提取函数，不加任何修改。
	返回 list of dict, 每个 dict 包含 'bbox', 'conf', 'cls_id', 'cls_name', 'feature'
	"""
	img0 = cv2.imread(img_path)
	if img0 is None:
		raise FileNotFoundError(f"Image not found: {img_path}")
	orig_shape = img0.shape[:2]

	# 钩子获取 backbone 最后一层特征图
	features = []

	def hook_fn(module, inp, out):
		features.append(out.detach())

	target_layer = yolo_model.model.model[-2]  # 与训练时一致
	handle = target_layer.register_forward_hook(hook_fn)

	results = yolo_model(img_path, verbose=False)
	handle.remove()
	feat_map = features[0]  # (1, C, Hf, Wf)

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
			fx1 = max(0, min(fx1, w_feat - 1))
			fx2 = max(0, min(fx2, w_feat - 1))
			fy1 = max(0, min(fy1, h_feat - 1))
			fy2 = max(0, min(fy2, h_feat - 1))
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


def build_targets_from_detections(detections, img_path, device):
	"""
	将特征提取的输出转换为模型需要的 target 格式。
	返回：单个 target 字典 (dict) 或 list of dict (这里只处理单张图片)
	"""
	img = cv2.imread(img_path)
	orig_h, orig_w = img.shape[:2]

	if len(detections) > 0:
		boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32, device=device)
		labels = torch.tensor([d['cls_id'] for d in detections], dtype=torch.long, device=device)
		# 注意：特征维度由实际提取的特征向量长度决定，确保与训练时一致
		features = torch.tensor(np.array([d['feature'] for d in detections]), dtype=torch.float32, device=device)
	else:
		# 无检测结果时，需要知道特征维度（与 YOLO 模型输出通道数一致）
		# 简单起见，从任意一张图提取特征时获得一个空的特征维度；或者默认为 256/512
		# 这里根据 args.detect_model 猜测，更健壮的方式是从第一次特征提取后记住维度
		feature_dim = 256 if args.detect_model.endswith('n') else 512  # 与训练时逻辑一致
		boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
		labels = torch.zeros((0,), dtype=torch.long, device=device)
		features = torch.zeros((0, feature_dim), dtype=torch.float32, device=device)

	target = {
		'boxes': boxes,
		'labels': labels,
		'features': features,
		'orig_size': torch.tensor([orig_h, orig_w], dtype=torch.long, device=device)
	}
	return target


def predict_one_image(image_path, model, detect_model, args, device):
	"""
	对单张图像进行意图和关系预测。
	返回模型原始输出字典。
	"""
	# 1. 提取检测结果和 RoI 特征
	detections = extract_features_with_detections(detect_model, image_path, device)

	# 2. 构建 target
	target = build_targets_from_detections(detections, image_path, device)

	# 3. 模型推理
	model.eval()
	with torch.no_grad():
		predictions = model([target])  # 输入必须是列表
	return predictions[0]  # 单图取第一个元素


def process_one_image(pred, image_path, pritext=False):
	# 只有这些目标才有意图
	has_intent_mask = [0, 3, 4, 5]

	# ========== 5. 解析并输出结果 ==========
	boxes = pred['boxes']
	if boxes.shape[0] == 0:
		print("No objects detected.")
		return None

	# 类别名称
	obj_names = args.entity_categories  # 实体类别
	intent_names = args.intent_categories  # 意图类别
	rel_names = args.rel_categories  # 关系类别

	# ---------- 意图预测 ----------
	intent_logits = pred['intent_logits']  # shape (N, num_intent_classes+1)，最后一维是背景
	# 计算非背景类的概率
	intent_probs = torch.softmax(intent_logits[:, :-1], dim=1)  # 排除背景
	intent_scores, intent_preds = torch.max(intent_probs, dim=1)

	if pritext:
		print("========== 检测目标及其意图 ==========")
	for i in range(boxes.shape[0]):
		obj_label = pred['obj_labels'][i].item()
		obj_name = obj_names[obj_label]
		intent_id = intent_preds[i].item()
		intent_name = intent_names[intent_id] if intent_scores[i] > 0.2 and obj_label in has_intent_mask else "background"
		if pritext:
			print(f"  目标 {i}: {obj_name} (bbox={boxes[i].tolist()}), "
			      f"意图: {intent_name} (置信度 {intent_scores[i]:.2f})")

	# ---------- 关系预测 ----------
	rel_logits = pred['relation_logits']  # shape (E, num_relation_classes+1)
	edge_index = pred['edge_index']  # shape (2, E)

	if rel_logits.shape[0] > 0:
		rel_probs = torch.softmax(rel_logits[:, :-1], dim=1)  # 排除背景
		rel_scores, rel_preds = torch.max(rel_probs, dim=1)
		if pritext:
			print("\n========== 关系预测 ==========")
		for e in range(edge_index.shape[1]):
			src = edge_index[0, e].item()
			dst = edge_index[1, e].item()
			rel_id = rel_preds[e].item()
			rel_name = rel_names[rel_id] if rel_scores[e] > 0.2 else "background"
			if pritext:
				print(f"  {src} -> {dst}: {rel_name} (置信度 {rel_scores[e]:.2f})")

	# ========== 6. 可视化预测结果 ==========
	# 定义哪些物体类别可以有意图（根据你的实际需求调整）
	# 示例：只有坦克、士兵组、士兵、卡车可以有意图
	intent_able_classes = {"tank", "soldiergroup", "soldier", "truck"}
	has_intent_mask = [cat in intent_able_classes for cat in args.entity_categories]

	# 读取原始图像
	img = cv2.imread(image_path)
	if img is None:
		print(f"Cannot read image for visualization: {image_path}")
		return
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换到 RGB 以便显示/保存
	img[:,:,:] = 255
	# 颜色设置
	node_color = (214, 98, 20)   # (214, 98, 20)  (0, 255, 0)  # 绿色
	intent_text_color = (214, 98, 20) # (68, 160, 221)  # 蓝色
	rel_line_color = (255, 0, 0)  # 红色
	rel_text_color = (244, 163, 36)  # 橙色

	# 获取各框的中心点（用于画关系连线）
	centers = []
	for box in boxes:
		x1, y1, x2, y2 = box.int().tolist()
		cx = (x1 + x2) // 2
		cy = (y1 + y2) // 2
		centers.append((cx, cy))

	# 绘制每个目标框和意图
	for i, box in enumerate(boxes):
		x1, y1, x2, y2 = box.int().tolist()
		obj_label = pred['obj_labels'][i].item()
		obj_name = obj_names[obj_label]
		intent_id = intent_preds[i].item()
		intent_score = intent_scores[i].item()
		has_intent = has_intent_mask[obj_label]

		# 决定是否显示意图
		show_intent = has_intent and intent_score > 0.2
		label_text = obj_name
		if show_intent:
			intent_name = intent_names[intent_id]
			label_text = f"{obj_name} {intent_name}"

		r = int(min(img.shape[0], img.shape[1])/30)
		# 根据是否有意图选择实心/空心圆标记在框的左上角
		if has_intent:
			cv2.circle(img, (int(0.5*(x1+x2)), int(0.5*(y1+y2))), r, node_color, -1)  # 实心红点表示有意图能力
		else:
			cv2.circle(img, (int(0.5*(x1+x2)), int(0.5*(y1+y2))), r, node_color, 2)  # 空心红圈表示无意图能力

		# 画边界框
		# cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
		# 写标签（类别+意图）
		cv2.putText(img, label_text, (int(10+0.5*(x1+x2)), int(0.5*(y1+y2))), #(x1, max(y1 - 10, 20)),
		            cv2.FONT_HERSHEY_SIMPLEX, 1, intent_text_color, 2, cv2.LINE_AA)

	# 绘制关系
	if rel_logits.shape[0] > 0:
		for e in range(edge_index.shape[1]):
			src = edge_index[0, e].item()
			dst = edge_index[1, e].item()
			rel_id = rel_preds[e].item()
			rel_score = rel_scores[e].item()
			rel_name = rel_names[rel_id] if rel_score > 0.1 else None
			if rel_name is None:
				continue

			# 画线（从src中心到dst中心）
			pt1 = centers[src]
			pt2 = centers[dst]
			cv2.line(img, pt1, pt2, rel_line_color, 2)

			# 在线中点附近标注关系名称
			mid_x = (pt1[0] + pt2[0]) // 2
			mid_y = (pt1[1] + pt2[1]) // 2
			cv2.putText(img, rel_name, (mid_x, mid_y),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, rel_text_color, 2, cv2.LINE_AA)
		# cv2.imshow("Predicted Intent and Relations", img)
	return img


def get_parser(yolo=None, gnn=None, use_gt=None, use_rel=None, post=None):
	parser = argparse.ArgumentParser(description='UAV Scene Graph Training')
	# 一些学习设置
	parser.add_argument('--name', type=int, default=-1, )
	parser.add_argument('--device', type=str, default='cuda:7', help='Device to use')
	parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
	parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
	parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
	parser.add_argument('--gnn_hidden_dim', type=int, default=256, help='GNN hidden dimension')
	parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN layers')
	parser.add_argument('--yolo_feat_dim', type=int, default=512, help='YOLO feature dimension')
	parser.add_argument('--use_spatial_feat', type=str, default='mlp', choices=['none', 'mlp', 'manual'],
	                    help='Use spatial features')
	parser.add_argument('--use_class_embed', default=True, help='Use class embeddings')

	# 模型和数据设置
	parser.add_argument('--detect_model', type=str, default='yolo11n',
	                    choices=['yolo11s', 'yolo11n', 'yolo26s', 'yolo26n', 'yolov8s', 'yolov8n',
	                             'yolo11l', 'yolo11m'], help='YOLO model path')
	parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'res_gcn'])
	parser.add_argument('--data_name', type=str, default='UAVmilitary', )
	parser.add_argument('--use_gt', type=bool, default=False, help='Use target detection boxes')
	parser.add_argument('--use_rel', type=bool, default=True)
	parser.add_argument('--post', type=bool, default=True, help='Use post process')

	# 目类设置
	parser.add_argument('--resume', type=bool, default=True, help='Resume from checkpoint path')
	parser.add_argument('--eval', type=bool, default=True, help='Eval from checkpoint path')
	parser.add_argument('--evalDetails', type=bool, default=False, )

	parser.add_argument('--save_dir', type=str, default='./ckpt', help='Directory to save checkpoints')
	parser.add_argument('--data_root', type=str, default='/home/ps/MyDataSets/UAV_military/relation',
	                    help='Data root directory')
	parser.add_argument('--image_path', type=str,
	                    default='/home/ps/MyDataSets/UAV_military/new_yolo/images/test/', ) # 002965.jpg

	parser.add_argument('--num_object_classes', type=int, default=13, )
	parser.add_argument('--num_relation_classes', type=int, default=22, )
	parser.add_argument('--num_intent_classes', type=int, default=15, )
	parser.add_argument('--num_scene_classes', type=int, default=5, )

	parser.add_argument('--entity_categories', nargs='+', default=[
		"tank", "building", "explosion", "soldiergroup", "soldier", "truck",
		"trench", "road", "bridge", "tree", "forest", "river", "hill"], )
	parser.add_argument('--rel_categories', nargs='+', default=[
		"behind", "in_front_of", "next_to", "hiding_behind", "concealed_in",
		"surrounding", "inside", "moving_along", "occupying", "passing_through", "on",
		"following", "above", "below", "under", "over", "bordering", "cutting_through",
		"part_of", "reconnoitering", "crossing", "along"])
	parser.add_argument('--intent_categories', nargs='+', default=[
		"assault", "reconnoitering", "infantry_tank_collaboration", "defense",
		"recon", "supply", "march", "rest", "assemble", "withdraw", "ambush",
		"moving_along", "hiding_behind", "occupying", "concealed_in"])
	parser.add_argument('--scene_categories', nargs='+', default=[
		"urban", "plain", "marsh", "mountain", "jungle"])

	parser.add_argument('--alpha', type=float, default=2.0, )

	args = parser.parse_args()

	# yolo_name_lib = ['yolo11s', 'yolo11n', 'yolo26s', 'yolo26n', 'yolov8s', 'yolov8n', 'yolo11l', 'yolo11m']
	# args.device = args.device if args.name == -1 else 'cuda:{}'.format(str(args.name))
	# args.detect_model = args.detect_model if args.name == -1 else yolo_name_lib[args.name]

	args.detect_model = yolo if yolo is not None else args.detect_model
	args.gnn_type = gnn if gnn is not None else args.gnn_type
	args.use_gt = use_gt if use_gt is not None else args.use_gt
	args.use_rel = use_rel if use_rel is not None else args.use_rel

	task_name = "PredITs" if args.use_gt else "ITDet"

	if args.gnn_type == "gcn":
		if args.use_rel:
			args.alpha = 2.0 if args.use_gt else 1.0
		else:
			args.alpha = 1.0
	else:
		args.alpha = 1.4 if args.use_gt else 1.6

	task_name = task_name if args.use_rel else task_name + "_norel"
	args.save_dir = os.path.join(args.save_dir, args.data_name, args.detect_model, args.gnn_type, task_name)
	os.makedirs(args.save_dir, exist_ok=True)

	return args


def main(args):
	device = args.device if torch.cuda.is_available() else 'cpu'

	# ========== 1. 加载 YOLO 检测模型 ==========
	if "yolo" in args.detect_model:
		yolo_path = f"/home/ps/MyProject/YOLOvX/runs/detect/train_{args.detect_model}_UAVMilitary/weights/best.pt"
		if not os.path.isfile(yolo_path):
			print(f"YOLO model not found: {yolo_path}")
			return
		detect_model = YOLO(yolo_path)
	else:
		print("Unsupported detection model")
		return
	detect_model.to(device)

	## ========== 2. 加载训练好的 UAVSceneGraphModel ==========
	model = UAVSceneGraphModel(args).to(device)

	resume_path = os.path.join(args.save_dir, 'best.pth')  # 或 'newest.pth'
	if not os.path.exists(resume_path):
		print(f"Model checkpoint not found: {resume_path}")
		return
	checkpoint = torch.load(resume_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	print(f"Loaded UAVSceneGraphModel from {resume_path}")
	model.eval()

	# ========== 3. 指定要推理的图像路径 ==========
	# 建议通过命令行参数传入，这里给一个示例路径
	image_path = args.image_path  # <-- 请修改为实际路径
	if not os.path.exists(image_path):
		print(f"Image not found: {image_path}")
		return

	# 判断路径是否是一个目录
	if not os.path.isdir(image_path):
		# ========== 4. 预测 ==========
		pred = predict_one_image(image_path, model, detect_model, args, device)
		img = process_one_image(pred, image_path)
		# 保存结果
		filename = os.path.basename(image_path)
		output_path = f'result/{filename}'
		cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		print(f"Visualization saved to {output_path}")
	else:
		for filename in os.listdir(image_path):
			if filename.endswith(".jpg"):
				filepath = os.path.join(image_path, filename)
				pred = predict_one_image(filepath, model, detect_model, args, device)
				img = process_one_image(pred, filepath)
				if img is None:
					print(f"No predictions for {filepath}, skipping visualization.")
					continue
				output_path = f'result/test/{filename}'
				cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
				print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
	args = get_parser()

	# 注意：推理时 use_gt 必须为 False（不使用真值框），use_rel 根据你的模型设置
	# 请确保 args 与训练时的配置一致（如 gnn_type, detect_model 等）
	main(args)
