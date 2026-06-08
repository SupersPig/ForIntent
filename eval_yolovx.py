# 验证代码
from ultralytics import YOLO

yolo_name_lib = ['yolo11s', 'yolo11n', 'yolo26s', 'yolo26n', 'yolov8s', 'yolov8n', 'yolo11l', 'yolo11m']
f = open("./runs/detect/results.txt", "w+")
f.write(f'yolo_name\tmAP50-95\tmAP50\tP\tR\tinference\n')

for yolo_name in yolo_name_lib:
	best_ckpt = f"runs/detect/train_{yolo_name}_UAVMilitary/weights/best.pt"
	model = YOLO(best_ckpt)
	results = model.val(
		data='/home/ps/MyDataSets/UAV_military/new_yolo/data.yaml',
		project="eval",
		name=yolo_name,
		imgsz=640,
		batch=1,
		workers=4,
		device=4,
		split='val',
		conf=0.25,
		iou=0.7,
		plots=True
	)

	# 打印关键指标
	print(f"验证{yolo_name}完成!")
	print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
	print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
	print(f"精确度P: {results.results_dict['metrics/precision(B)']:.4f}")
	print(f"召回率R: {results.results_dict['metrics/recall(B)']:.4f}")
	print(f"推理时间: {results.speed['inference']:.4f}ms")
	f.write(
		f"{yolo_name}\t"
		f"{results.results_dict['metrics/mAP50-95(B)']:.4f}\t"
		f"{results.results_dict['metrics/mAP50(B)']:.4f}\t"
		f"{results.results_dict['metrics/precision(B)']:.4f}\t"
		f"{results.results_dict['metrics/recall(B)']:.4f}\t"
		f"{results.speed['inference']:.4f}\n")
f.close()
