import argparse
import os
import json
import logging
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from dataset.UAVmilitary import UAVMilitaryDataset, collate_fn
from model import UAVSceneGraphModel
from engine_intent import train_one_epoch, evaluate
import pandas as pd
from util.util import plot_metrics, save_metrics


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
	parser.add_argument('--image_path', type=str, default='/home/ps/MyDataSets/UAV_military/new_yolo/images/test/002965.jpg',)

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

	log_name = "eval_log.txt" if args.eval else "train_log.txt"
	# os.remove(os.path.join(args.save_dir, "results.txt"))
	# 配置日志格式和级别
	logging.basicConfig(level=logging.INFO,
	                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	                    handlers=[logging.FileHandler(os.path.join(args.save_dir, log_name)),
	                              logging.StreamHandler()  # 输出到控制台
	                              ]
	                    )
	logger = logging.getLogger("ForIntent")

	return args, logger


def main(args, logger):
	logger.info(args)
	device = args.device if torch.cuda.is_available() else 'cpu'

	# 加载 YOLO 模型
	if "yolo" in args.detect_model:
		yolo_path = f"/home/ps/MyProject/YOLOvX/runs/detect/train_{args.detect_model}_UAVMilitary/weights/best.pt"
		assert os.path.isfile(yolo_path), f"Pretrained YOLO model not found at {yolo_path}"
		detect_model = YOLO(yolo_path)
	else:
		logger.error(
			f"Error: Unsupported detection model path {args.detect_model}. Please provide a valid YOLO model path.")
		return
	detect_model.to(device)

	# 创建数据集
	train_dataset = UAVMilitaryDataset(args, 'train', detect_model, device)
	args.entity_categories = train_dataset.entity_categories
	args.rel_categories = train_dataset.rel_categories
	args.intent_categories = train_dataset.intent_categories
	args.scene_categories = train_dataset.scene_categories

	if args.eval:
		val_dataset = UAVMilitaryDataset(args, 'val', detect_model, device)
	else:
		val_dataset = UAVMilitaryDataset(args, 'val', detect_model, device)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
	                          collate_fn=collate_fn, num_workers=4)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
	                        collate_fn=collate_fn, num_workers=4)

	# 初始化模型
	model = UAVSceneGraphModel(args=args).to(device)

	# 打印参数量
	yolo_parameters = sum(p.numel() for p in detect_model.parameters())
	logger.info(f'Number of YOLO parameters: {yolo_parameters}')
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	logger.info(f'Number of trainable parameters: {n_parameters}')

	# 优化器
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	# 加载 checkpoint（如果提供了 --resume）
	start_epoch = 0
	best_acc = 0.0

	if args.resume:
		resume_path = os.path.join(args.save_dir, 'best.pth') if args.eval else os.path.join(
			args.save_dir, "newest.pth")
		if os.path.exists(resume_path):
			logger.info(f"Resuming from {resume_path}")
			checkpoint = torch.load(resume_path, map_location=device)
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			start_epoch = checkpoint['epoch'] + 1
			best_acc = checkpoint.get('best_acc', 0.0)
			logger.info(f"Resumed from {resume_path}, epoch {start_epoch - 1}, best_acc {best_acc:.4f}")

			f = open(os.path.join(args.save_dir, 'parameter.txt'), 'w+')
			f.write("alpha\tAP\tmAP\n")
			if args.post:
				model.post = False
				logger.info("model.post = False \t 评估不适用后处理方法处理的指标...")
				results = evaluate(model, val_loader, device, args, logger=logger)
				# json.dump(results, open(os.path.join(args.save_dir, 'results_ori.json'), 'w+'))
				model.post = True
				f.write(f"0.0\t{results['intent_accuracy']:.4f}\t{results['mean_intent_accuracy']:.4f}\n")

			for alpha in range(0, 30):
				model.corrector.alpha = alpha / 10
				logger.info(f"Testing with alpha={model.corrector.alpha:.2f}")
				# logger.info("model.post = True \t 评估使用后处理方法处理的指标...")
				results = evaluate(model, val_loader, device, args, logger=logger)
				# json.dump(results, open(os.path.join(args.save_dir, 'results_pp.json'), 'w+'))
				f.write(f"{model.corrector.alpha:.2f}\t{results['intent_accuracy']:.4f}\t{results['mean_intent_accuracy']:.4f}\n")
			f.close()

			if args.eval:
				return
		else:
			logger.info(f"Checkpoint {resume_path} not found. Starting training from scratch.")
	else:
		logger.info("Starting training from scratch.")

	# 如果只评估模型，直接返回
	if args.eval:
		return

	# 在训练循环之前初始化
	history = {
		'epoch': [],
		'train_loss': [],
		'val_intent_acc': []
	}

	csv_path = os.path.join(args.save_dir, 'training_history.csv')
	plot_path = os.path.join(args.save_dir, 'training_curves.svg')

	# 如果已有历史文件，加载并继续记录
	if os.path.exists(csv_path):
		df_existing = pd.read_csv(csv_path)
		history['epoch'] = df_existing['epoch'].tolist()
		history['train_loss'] = df_existing['train_loss'].tolist()
		history['val_intent_acc'] = df_existing['val_intent_acc'].tolist()

	# 训练循环
	for epoch in range(start_epoch, args.epochs):
		train_loss = train_one_epoch(model, train_loader, optimizer, device)
		logger.info(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}")

		# 验证
		val_metrics = evaluate(model, val_loader, device, args, logger=logger)
		intent_acc = val_metrics.get('intent_accuracy', 0.0)
		logger.info(f"Validation Intent Accuracy: {intent_acc:.4f}")

		# 记录历史
		history['epoch'].append(epoch)
		history['train_loss'].append(train_loss)
		history['val_intent_acc'].append(intent_acc)

		# 保存历史到CSV
		df = pd.DataFrame(history)
		df.to_csv(csv_path, index=False)

		# 实时绘制曲线
		plot_metrics(history, plot_path)

		realtime_path = os.path.join(args.save_dir, "newest.pth")
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'best_acc': best_acc,
			'val_metrics': val_metrics
		}, realtime_path)

		# 保存最佳模型（根据 intent_accuracy）
		if intent_acc > best_acc:
			best_acc = intent_acc
			best_path = os.path.join(args.save_dir, f"better_epoch{epoch:03d}.pth")
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'best_acc': best_acc,
				'val_metrics': val_metrics
			}, best_path)
			logger.info(f"New best model saved with accuracy {best_acc:.4f}")

	logger.info(f"Training {args.detect_model} completed.")



if __name__ == "__main__":
	# 输出不同级别的日志
	# logger.debug("调试信息")
	# logger.info("普通信息")
	# logger.warning("警告信息")
	# logger.error("错误信息")
	# logger.critical("严重错误")

	args, logger = get_parser()
	main(args, logger)

	yolo_name_lib = ['yolo11s', 'yolo11n', 'yolo26s', 'yolo26n', 'yolov8s', 'yolov8n', 'yolo11l', 'yolo11m']
	for yolo_name in yolo_name_lib:
		for gnn in ['gcn', 'gat']:
			# for use_gt in [True, False]:
			# 	for use_rel in [True, False]:
			use_gt = False
			use_rel = True
			args, logger = get_parser(yolo_name, gnn, use_gt, use_rel, post=True)
			main(args, logger)

'''
conda activate yolo
cd /home/ps/MyProject/YOLOvX
python train_intent.py --name 0
'''
