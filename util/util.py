import os.path
import json
import matplotlib
matplotlib.use('Agg')   # 使用非交互式后端，不会弹出窗口
import matplotlib.pyplot as plt


def plot_metrics(history, save_path):
	"""绘制训练和验证曲线"""
	plt.figure(figsize=(12, 5))

	# 损失曲线
	plt.subplot(1, 2, 1)
	plt.plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training Loss')
	plt.legend()
	plt.grid(True)

	# 准确率曲线
	plt.subplot(1, 2, 2)
	plt.plot(history['epoch'], history['val_intent_acc'], 'r-', label='Val Intent Acc')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Validation Intent Accuracy')
	plt.legend()
	plt.grid(True)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150)
	plt.close()
	# print(f"曲线图已保存至: {save_path}")

def save_metrics(results, save_path):
	with open(os.path.join(save_path, "results.json"), "w+", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=4)

