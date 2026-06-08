# 训练YOLOvX代码
from ultralytics import YOLO

if __name__ == '__main__':
	# model_name = 'yolo11s'
	# model_name = 'yolo11n'
	# model_name = 'yolo26n'
	# model_name = 'yolo26s'
	model_name = 'yolov8n'
	# model_name = 'yolov8s'
	# model_name = 'yolo11l'
	# model_name = 'yolo11m'

	# model = YOLO("yolo11n.yaml")
	model = YOLO("./pretrained/{}.pt".format(model_name))
	# model = YOLO("yolov11n.yaml").load("yolo11n.pt")

	# 训练参数配置
	results = model.train(
		data="/home/ps/MyDataSets/UAV_military/new_yolo/data.yaml",
		epochs=100,
		imgsz=640,
		device=4,
		name="train_{}_UAVMilitary".format(model_name),
		resume=True
	)

	# results = model.train(
	# 	data="/home/ps/DataSets/UAV_military/data.yaml",
	# 	epochs=300,
	# 	imgsz=640,
	# 	batch=16,
	# 	workers=4,
	# 	device=[0, 1],
	# 	name="train_UAVMilitaryTarget",
	# 	# 强烈建议添加的参数
	# 	patience=50,  # 早停机制，50个epoch没有改进则停止
	# 	save=True,  # 保存训练检查点
	# 	save_period=10,  # 每10个epoch保存一次
	# 	cache=False,  # 禁用缓存以避免内存问题
	# 	optimizer='auto',  # 自动选择优化器results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])
	# 	lr0=0.01,  # 初始学习率
	# 	cos_lr=True,  # 使用余弦学习率调度
	# 	weight_decay=0.0005,  # 权重衰减
	# 	warmup_epochs=3.0,  # 学习率预热
	# 	box=7.5,  # 边框损失权重
	# 	cls=0.5,  # 分类损失权重
	# 	dfl=1.5,  # 分布焦点损失权重
	# 	close_mosaic=10,  # 最后10个epoch关闭mosaic增强
	# 	resume=False  # 不从之前的检查点恢复
	# )

'''
# 下载预训练权重
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
python scripts/download_weights.py --model yolov11s

yolo detect train name="train_yolo11s_UAVMilitaryQwen3" data=/home/ps/DataSets/UAV_military_Qwen3/data.yaml model=yolo11s.pt epochs=10 imgsz=640 device=1
yolo detect train data=/home/ps/DataSets/UAV_military_Qwen3/data.yaml model=yolo11n.pt epochs=100 imgsz=640 device=[0, 1]
yolo detect train data=/home/ps/DataSets/UAV_military/data.yaml model=yolo11s.pt epochs=200 imgsz=640 device=[0, 1]
yolo detect train data=/home/ps/DataSets/UAV_military/data.yaml model=yolo11s.pt epochs=200 imgsz=640 device=[0, 1]
export NCCL_ALGO=ring
export NCCL_P2P_DISABLE=1
'''
