# 预测代码
from ultralytics import YOLO
import torch

# model_name = 'yolo11s'
# model_name = 'yolo11n'
# model_name = 'yolo26n'
# model_name = 'yolo26s'
# model_name = 'yolov8n'
# model_name = 'yolov8s'
# model_name = 'yolo11l'
# model_name = 'yolo11m'
# model_name_list = ['yolo11s', 'yolo11n', 'yolo26n', 'yolo26s', 'yolov8n', 'yolov8s', 'yolo11l', 'yolo11m']
model_name_list = ['yolo11s']
# ckpt = torch.load("/home/ps/MyProject/YOLOvX/runs/detect/train_yolo11s_UAVMilitary/weights/best.pt")
# print("hhh")

for model_name in model_name_list:
	model = YOLO(f"/home/ps/MyProject/YOLOvX/runs/detect/train_{model_name}_UAVMilitary/weights/best.pt")
	results = model.predict(
		source="/home/ps/MyDataSets/UAV_military/new_yolo/images/test/000541.jpg",
		save=False,
		imgsz=640,
		conf=0.1,
		iou=0.5,
		show_labels=True,
		show_conf=True
	)

	for result in results:
		print(f"\n检测到 {len(result.boxes)} 个目标:")

		# 处理边界框
		boxes = result.boxes
		for i, box in enumerate(boxes):
			cls_id = int(box.cls[0])
			confidence = float(box.conf[0])
			class_name = model.names[cls_id]
			print(f"  {i + 1}. {class_name}: {confidence:.3f}")

		# 显示结果
		result.show()

		# 可选：保存结果图像
		result.save(f"detection_{model_name}_result.jpg")