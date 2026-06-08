import os
import numpy as np
import json
import torch


class EnhancedByPrior:
	def __init__(self, args):
		self.num_object_classes = len(args.entity_categories)
		self.num_relation_classes = len(args.rel_categories)
		self.num_intent_classes = len(args.intent_categories)
		self.num_scene_classes = len(args.scene_categories)

		self.device = args.device if torch.cuda.is_available() else 'cpu'

		self.args = args
		self.alpha = args.alpha

		self.prior = self.get_prior()
		self._compute_priors()
		self.rules = self.extract_strong_rules()

	def get_prior(self):
		prior_path = os.path.join("ckpt", "prior", self.args.data_name)

		if os.path.exists(os.path.join(prior_path, "prior_dict3.npz")):
			return np.load(os.path.join(prior_path, "prior_dict3.npz"))
		else:
			os.makedirs(prior_path, exist_ok=True)

			annotations_path = os.path.join(self.args.data_root, "annotations.json")
			dataset_spilt_path = os.path.join(self.args.data_root, "dataset_spilt.json")
			annotations = json.load(open(annotations_path, 'r', encoding='utf-8'))
			dataset_spilt = json.load(open(dataset_spilt_path, 'r', encoding='utf-8'))
			train_ids = dataset_spilt['train']

			# 矩阵维度 +1 预留背景/缺失列，背景位于最后一维
			I = self.num_intent_classes
			O = self.num_object_classes
			S = self.num_scene_classes
			R = self.num_relation_classes

			intent_entity = np.zeros((I, O + 1), dtype=np.float32)
			intent_scene = np.zeros((I, S + 1), dtype=np.float32)
			intent_relation = np.zeros((I, R + 1), dtype=np.float32)
			intent_entity_scene = np.zeros((I, O + 1, S + 1), dtype=np.float32)
			intent_entity_scene_relation = np.zeros((I, O + 1, S + 1, R + 1), dtype=np.float32)
			# 新增：意图 - 主语实体 - 关系 - 宾语实体
			intent_sub_rel_obj = np.zeros((I, O + 1, R + 1, O + 1), dtype=np.float32)

			for img_id in train_ids:
				ann = annotations[img_id]
				entities = np.array(ann['entity']['label_id'])
				intents = np.array(ann['intents']['triplet']) if 'triplet' in ann['intents'] else np.empty((0, 2),
				                                                                                           dtype=int)
				relations = np.array(ann['relations']['triplet']) if 'triplet' in ann['relations'] else np.empty((0, 3),
				                                                                                                 dtype=int)
				scene = ann['scene']['label_id']

				if intents.shape[0] == 0:
					continue

				# 为每个关系构建快速索引, ！！！是有问题的，应该只为主语构建
				rel_by_entity = {}
				for rel in relations:
					sub_idx, obj_idx, rel_id = rel
					# for e_idx in [sub_idx, obj_idx]:
					if sub_idx not in rel_by_entity:
						rel_by_entity[sub_idx] = []
					rel_by_entity[sub_idx].append(rel_id)
				# if e_idx not in rel_by_entity:
				# 	rel_by_entity[e_idx] = []
				# rel_by_entity[e_idx].append(rel_id)

				# 构建以意图实体为主语的三元组索引
				sub_rel_obj = {}  # entity_idx -> list of (rel_id, obj_class)
				for rel in relations:
					sub_idx, obj_idx, rel_id = rel
					if sub_idx not in sub_rel_obj:
						sub_rel_obj[sub_idx] = []
					obj_class = entities[obj_idx]  # if 0 <= obj_idx < len(entities) else 0 # 超出索引应该直接报错
					sub_rel_obj[sub_idx].append((rel_id, obj_class))

				for intent_ann in intents:
					entity_idx, intent_id = intent_ann
					if entity_idx < 0 or entity_idx >= len(entities):  # 就不应该超出索引，！！！修改为直接报错
						continue
					if intent_id < 0 or intent_id >= I:
						continue

					entity_class = entities[entity_idx]
					ec_idx = entity_class
					sc_idx = scene

					if ec_idx == 1 or ec_idx > 5:
						print(
							f"Warning: entity_class {entity_class} (index {ec_idx}) is out of expected range for image {img_id}")

					# 原有统计（部分可留用）
					intent_entity[intent_id, ec_idx] += 1
					intent_scene[intent_id, sc_idx] += 1
					intent_entity_scene[intent_id, ec_idx, sc_idx] += 1

					related_rels = rel_by_entity.get(entity_idx, [])
					if len(related_rels) > 0:
						for rel_id in related_rels:
							if 0 <= rel_id < R:
								intent_relation[intent_id, rel_id] += 1
								intent_entity_scene_relation[intent_id, ec_idx, sc_idx, rel_id] += 1
					else:
						intent_entity_scene_relation[intent_id, ec_idx, sc_idx, R] += 1

					# 新增：主语‑关系‑宾语统计
					if entity_idx in sub_rel_obj:
						for rel_id, obj_class in sub_rel_obj[entity_idx]:
							if 0 <= rel_id < R and 0 <= obj_class < O:
								intent_sub_rel_obj[intent_id, ec_idx, rel_id, obj_class] += 1
					else:
						intent_sub_rel_obj[intent_id, ec_idx, R, O] += 1

			prior_dict = {
				'intent_entity': intent_entity,
				'intent_scene': intent_scene,
				'intent_relation': intent_relation,
				'intent_entity_scene': intent_entity_scene,
				'intent_entity_scene_relation': intent_entity_scene_relation,
				'intent_sub_rel_obj': intent_sub_rel_obj,  # 新增
			}
			save_file = os.path.join(prior_path, "prior_dict3.npz")
			np.savez(save_file, **prior_dict)
			print(f"Prior statistics saved to {prior_path}/prior_dict.npz")
			return prior_dict

	def _compute_priors(self):
		eps = 1e-8
		device = self.device
		O = self.num_object_classes
		R = self.num_relation_classes
		I = self.num_intent_classes

		# 加载统计矩阵并去除背景类（最后一维）
		intent_entity = torch.from_numpy(self.prior['intent_entity']).to(device).float()[:, :O]  # (I, O)
		intent_relation = torch.from_numpy(self.prior['intent_relation']).to(device).float()[
			:, :R]  # (I, R)         # (I, O, S)
		self.intent_sub_rel_obj = torch.from_numpy(self.prior['intent_sub_rel_obj']).to(device).float()[
			:, :O, :R, :O]  # (I, O, R, O)

		# ---------- 基本意图先验 P(I) ----------
		self.P_I = intent_entity.sum(dim=1).clamp(min=1)
		self.P_I_prime = self.P_I / (self.P_I.sum() + eps)
		# P(I | T)
		self.P_I_given_T = intent_entity.permute(1, 0)
		# P(I | R)
		self.P_I_given_R = intent_relation.permute(1, 0)
		# P(I | T, R, To)
		self.P_I_given_TRTo = torch.cat(
			[self.intent_sub_rel_obj.permute(1, 2, 3, 0), torch.zeros((O, 1, O, I), device=device)], dim=1)

		# 便于调试的可选保存
		self.prior_numpy = {
			"P_I": self.P_I.cpu().numpy(),
			"P_I_given_T": self.P_I_given_T.cpu().numpy(),
			"P_I_given_R": self.P_I_given_R.cpu().numpy(),
			"P_I_given_TRTo": self.P_I_given_TRTo.cpu().numpy(),
		}

	def extract_strong_rules(self, prob_threshold=0.6, min_sample_for_common=20, rare_intent_sample_threshold=20):
		"""
		提取强规则： (主语类别, 关系类别, 宾语类别) -> 意图类别
		Args:
			intent_sub_rel_obj: numpy array, shape (I, O, R, O) 计数（未加背景）
			entity_categories: list of str, 长度 O
			relation_categories: list of str, 长度 R
			intent_categories: list of str, 长度 I
			prob_threshold: 常见意图的条件概率阈值
			min_sample_for_common: 视为常见意图的最小样本数
			rare_intent_sample_threshold: 视为稀有意图的样本数阈值（低于此值则所有出现过的三元组都视为强规则）
		Returns:
			rules: list of dict, 每个包含 subject, relation, object, intent, strength, sample_count
		"""
		intent_sub_rel_obj = self.intent_sub_rel_obj
		entity_categories = self.args.entity_categories
		relation_categories = self.args.rel_categories
		intent_categories = self.args.intent_categories

		I, O, R, _ = intent_sub_rel_obj.shape
		rules = []

		# 计算每个意图的总样本数（主语-关系-宾语三元组的总出现次数）
		intent_total_samples = intent_sub_rel_obj.sum(axis=(1, 2, 3))  # (I,)

		for intent_id, intent_name in enumerate(intent_categories):
			total_cnt = intent_total_samples[intent_id]
			is_rare = total_cnt <= rare_intent_sample_threshold

			# 遍历所有 (主语, 关系, 宾语) 组合
			for sub_id in range(O):
				for rel_id in range(R):
					for obj_id in range(O):
						cnt = intent_sub_rel_obj[intent_id, sub_id, rel_id, obj_id]
						if cnt == 0:
							continue
						# 计算条件概率 P(intent | sub, rel, obj) = cnt / sum_{i'} cnt_i'
						total_triplet_cnt = self.P_I_given_T[sub_id, intent_id]
						prob = cnt / total_triplet_cnt.clamp(min=1)

						# 判断是否强规则
						is_strong = False
						if is_rare:
							# 稀有意图：所有出现过样本的三元组都是强规则
							is_strong = True
						else:
							# 常见意图：要求条件概率高且样本数足够
							if prob >= prob_threshold and cnt >= min_sample_for_common:
								is_strong = True

						if is_strong:
							rules.append({
								'subject': entity_categories[sub_id],
								'relation': relation_categories[rel_id],
								'object': entity_categories[obj_id],
								'intent': intent_name,
								'probability': prob,
								'sample_count': cnt,
								'intent_total_samples': total_cnt,
								'is_rare': is_rare
							})

		# 按强度排序：稀有意图优先，然后按概率降序，再按样本数降序
		rules.sort(key=lambda x: (not x['is_rare'], -x['probability'], -x['sample_count']))
		return rules

	def _compute_priors_give_I(self):
		eps = 1e-8
		device = self.device
		O = self.num_object_classes
		S = self.num_scene_classes
		R = self.num_relation_classes

		# 加载统计矩阵并去除背景类（最后一维）
		intent_entity = torch.from_numpy(self.prior['intent_entity']).to(device).float()[:, :O]  # (I, O)
		intent_scene = torch.from_numpy(self.prior['intent_scene']).to(device).float()[:, :S]  # (I, S)
		intent_relation = torch.from_numpy(self.prior['intent_relation']).to(device).float()[:, :R]  # (I, R)
		intent_entity_scene = torch.from_numpy(self.prior['intent_entity_scene']).to(device).float()[
			:, :O, :S]  # (I, O, S)
		intent_entity_scene_relation = torch.from_numpy(self.prior['intent_entity_scene_relation']).to(device).float()[
			:, :O, :S, :R]  # (I, O, S, R)
		intent_sub_rel_obj = torch.from_numpy(self.prior['intent_sub_rel_obj']).to(device).float()[
			:, :O, :R, :O]  # (I, O, R, O)

		self.prior_tensor_i = {
			"intent_entity": intent_entity.cpu().numpy(),
			"intent_scene": intent_scene.cpu().numpy(),
			"intent_relation": intent_relation.cpu().numpy(),
			"intent_entity_scene": intent_entity_scene.cpu().numpy(),
			"intent_entity_scene_relation": intent_entity_scene_relation.cpu().numpy(),
			"intent_sub_rel_obj2": intent_sub_rel_obj.permute(1, 2, 3, 0).sum(dim=(3)).cpu().numpy(),
			"intent_sub_rel_obj1": intent_sub_rel_obj.permute(1, 2, 3, 0).cpu().numpy(),
			"intent_sub_rel_obj0": intent_sub_rel_obj.cpu().numpy(),
			"intent_sub": intent_sub_rel_obj.sum(dim=(2, 3)).cpu().numpy(),
		}
		# ---------- 基本意图先验 P(I) ----------
		self.P_I_N = intent_entity.sum(dim=1)  # (I,)
		self.P_I = self.P_I_N / (self.P_I_N.sum() + eps)
		# ---------- P(S|I) ----------
		self.P_S_given_I = intent_scene / (intent_scene.sum(dim=1, keepdim=True) + eps)
		# ---------- P(T|I) ----------
		self.P_T_given_I = intent_entity / (intent_entity.sum(dim=1, keepdim=True) + eps)
		# ---------- P(R|I) ----------
		self.P_R_given_I = intent_relation / (intent_relation.sum(dim=1, keepdim=True) + eps)
		# ---------- P(T,S|I) ----------
		self.P_TS_given_I = intent_entity_scene / (intent_entity_scene.sum(dim=(1, 2), keepdim=True) + eps)
		# ---------- P(T,S,R|I) ----------
		self.P_TSR_given_I = intent_entity_scene_relation / (
					intent_entity_scene_relation.sum(dim=(1, 2, 3), keepdim=True) + eps)
		# ---------- P(R,To | T,I) 及衍生概率 ----------
		self.P_RTo_given_TI = intent_sub_rel_obj / (intent_sub_rel_obj.sum(dim=(2, 3), keepdim=True) + eps)
		# P(R|T,I) = sum_{To} P(R,To|T,I)
		self.P_R_given_TI = self.P_RTo_given_TI.sum(dim=3)  # (I, O, R)
		self.P_R_given_TI = self.P_R_given_TI / (self.P_R_given_TI.sum(dim=2, keepdim=True) + eps)
		# P(To|R,T,I) = P(R,To|T,I) / P(R|T,I)
		P_R_given_TI_expanded = self.P_R_given_TI.unsqueeze(-1)  # (I, O, R, 1)
		self.P_To_given_RTI = self.P_RTo_given_TI / (P_R_given_TI_expanded + eps)

		# 便于调试的可选保存
		self.prior_numpy_i = {
			"P_I": self.P_I.cpu().numpy(),
			"P_I_N": self.P_I_N.cpu().numpy(),
			"P_S_given_I": self.P_S_given_I.cpu().numpy(),
			"P_T_given_I": self.P_T_given_I.cpu().numpy(),
			"P_R_given_I": self.P_R_given_I.cpu().numpy(),
			"P_TS_given_I": self.P_TS_given_I.cpu().numpy(),
			"P_TSR_given_I": self.P_TSR_given_I.cpu().numpy(),
			"P_RTo_given_TI": self.P_RTo_given_TI.cpu().numpy(),
			"P_R_given_TI": self.P_R_given_TI.cpu().numpy(),
			"P_To_given_RTI": self.P_To_given_RTI.cpu().numpy(),
		}

		statistics_path = os.path.join(self.args.data_root, "statistics.json")
		self.statistics = json.load(open(statistics_path, 'r', encoding='utf-8'))

	def _compute_priors_give_TSR(self):
		eps = 1e-8
		device = self.device
		O = self.num_object_classes
		S = self.num_scene_classes
		R = self.num_relation_classes

		# 加载统计矩阵并去除背景类（最后一维）
		intent_entity = torch.from_numpy(self.prior['intent_entity']).to(device).float()[:, :O]  # (I, O)
		intent_scene = torch.from_numpy(self.prior['intent_scene']).to(device).float()[:, :S]  # (I, S)
		intent_relation = torch.from_numpy(self.prior['intent_relation']).to(device).float()[:, :R]  # (I, R)
		intent_entity_scene = torch.from_numpy(self.prior['intent_entity_scene']).to(device).float()[
			:, :O, :S]  # (I, O, S)
		intent_entity_scene_relation = torch.from_numpy(self.prior['intent_entity_scene_relation']).to(device).float()[
			:, :O, :S, :R]  # (I, O, S, R)
		intent_sub_rel_obj = torch.from_numpy(self.prior['intent_sub_rel_obj']).to(device).float()[
			:, :O, :R, :O]  # (I, O, R, O)

		# ---------- 用于推理的生成式条件分布（保留原有逻辑） ----------
		# P(I)
		self.P_I = intent_entity.sum(dim=1)  # (I,)
		self.P_I = self.P_I / (self.P_I.sum() + eps)

		# ---------- 转换为给定条件、意图在最后一维的后验概率表 ----------
		# 直接由统计矩阵按意图归一化得到 P(I | ...)
		self.P_I_given_T = intent_entity / (intent_entity.sum(dim=0, keepdim=True) + eps)  # (I, O) -> 转置为 (O, I)
		self.P_given_T = intent_entity.sum(dim=0)
		self.P_I_given_S = intent_scene / (intent_scene.sum(dim=0, keepdim=True) + eps)  # (I, S) -> (S, I)
		self.P_I_given_R = intent_relation / (intent_relation.sum(dim=0, keepdim=True) + eps)  # (I, R) -> (R, I)

		# P(I | T, S)
		self.P_I_given_TS = intent_entity_scene / (
				intent_entity_scene.sum(dim=0, keepdim=True) + eps)  # (I, O, S) -> (O, S, I)
		# P(I | T, S, R)
		self.P_I_given_TSR = intent_entity_scene_relation / (
				intent_entity_scene_relation.sum(dim=0, keepdim=True) + eps)  # (I, O, S, R) -> (O, S, R, I)
		# P(I | T, R, To)
		self.P_I_given_TRTo = intent_sub_rel_obj / (
				intent_sub_rel_obj.sum(dim=0, keepdim=True) + eps)  # (I, O, R, O) -> (O, R, O, I)

		# 调整维度顺序，使意图在最后一维
		self.prior_numpy_tsr = {
			"P_I_given_T": self.P_I_given_T.permute(1, 0).cpu().numpy(),  # (O, I)
			"P_given_T": self.P_given_T.cpu().numpy(),
			"P_I_given_S": self.P_I_given_S.permute(1, 0).cpu().numpy(),  # (S, I)
			"P_I_given_R": self.P_I_given_R.permute(1, 0).cpu().numpy(),  # (R, I)
			"P_I_given_TS": self.P_I_given_TS.permute(1, 2, 0).cpu().numpy(),  # (O, S, I)
			"P_I_given_TSR": self.P_I_given_TSR.permute(1, 2, 3, 0).cpu().numpy(),  # (O, S, R, I)
			"P_I_given_TRTo": self.P_I_given_TRTo.permute(1, 2, 3, 0).sum(dim=(3)).cpu().numpy(),  # (O, R, O, I)
			# 同时保留一些用于调试的生成式概率（可选）
			"P_I": self.P_I.cpu().numpy(),
		}

	def cond_prob_enhance(self, output, target, ues_rel_gt=False, aggr="mean"):  # 注意参数名应为 use_rel_gt，原文有误
			eps = 1e-8
			obj_labels = output['obj_labels']  # (N,)
			if obj_labels.shape[0] == 0:
				return output
			intent_logits = output['intent_logits']  # (N, I+1)
			relation_logits = output['relation_logits']  # (E, R+1)
			relation_labels = torch.argmax(relation_logits, dim=-1)  # (E,)
			edge_index = output['edge_index']  # (2, E)

			# 获取主语节点索引、主语类别、宾语类别
			sub_indices = edge_index[0]  # (E,)
			obj_indices = edge_index[1]  # (E,)
			sub_classes = obj_labels[sub_indices]  # (E,)
			obj_classes = obj_labels[obj_indices]  # (E,)
			rel_labels = relation_labels  # (E,)

			# 有效关系掩码：排除背景类（关系类别索引 == R）
			valid_mask = rel_labels < self.num_relation_classes  # (E,)

			# 如果没有有效关系，所有实体的先验设为全0
			if not valid_mask.any():
				prior_per_node = torch.zeros(obj_labels.shape[0], self.num_intent_classes, device=self.device)
			else:
				# 提取有效边信息
				valid_sub_indices = sub_indices[valid_mask]
				valid_sub_classes = sub_classes[valid_mask]
				valid_obj_classes = obj_classes[valid_mask]
				valid_rel_labels = rel_labels[valid_mask]

				# 计算每条有效边的先验分布 P(I | S, R, O)  (E_valid, I)
				# 从 self.P_I_given_TRTo 中索引：形状 (O, R, O, I)
				intent_sro = self.P_I_given_TRTo[
					valid_sub_classes, valid_rel_labels, valid_obj_classes, :]  # (E_valid, I)
				intent_s = self.P_I_given_T[valid_sub_classes, :]  # (E_valid, I)
				# 条件概率：P(I|S,R,O) = count(S,R,O,I) / count(S,I)
				prior_per_edge = intent_sro / (intent_s.clamp(min=1))  # (E_valid, I)
				# 分母为零的位置，概率置0（避免NaN）
				prior_per_edge[intent_s == 0] = 0.0

				# 按节点索引聚合：对每个节点的多条边取平均
				num_nodes = obj_labels.shape[0]
				prior_per_node = torch.zeros(num_nodes, self.num_intent_classes, device=self.device)

				if aggr == "max":
					print("Using max aggregation for prior enhancement")
				else:
					# 使用 scatter_add 先求和，再除以度数
					ones = torch.ones(valid_sub_indices.shape[0], 1, device=self.device)
					degree = torch.zeros(num_nodes, 1, device=self.device)
					degree.scatter_add_(0, valid_sub_indices.unsqueeze(1), ones)
					degree = degree.clamp(min=1)  # 避免除零
					prior_per_node.scatter_add_(0, valid_sub_indices.unsqueeze(1).expand(-1, self.num_intent_classes),
					                            prior_per_edge)
					prior_per_node /= degree  # 平均

			# intent_scores = torch.softmax(intent_logits[:, :-1], dim=-1)
			# intent_scores_ = torch.softmax(intent_scores / self.P_I_prime, dim=-1)
			# correct_scores = 0.55 * intent_scores + 0.45 * intent_scores_  # alpha=0.55
			# correct_scores[:, self.P_I>100] = 0
			# 原始意图分数（去掉背景类）
			# original_scores = intent_scores[:, :self.num_intent_classes]  # (N, I)
			prior_per_node[:, self.P_I>100] = 0
			intent_scores = torch.softmax(intent_logits[:, :-1], dim=-1)
			# 线性组合：final = alpha * original + (1-alpha) * prior
			# final_scores = intent_scores + 2 * prior_per_node # 对有rel predIT
			final_scores = intent_scores + self.alpha * prior_per_node
			# 重新归一化（可选，实际上 softmax 后再 log 等价于直接 log）
			# final_scores = final_scores / (final_scores.sum(dim=-1, keepdim=True) + eps)
			'''
			2026-05-19 11:08:15,420 - ForIntent - INFO - Testing with alpha=2.00    self.P_I>100
			Evaluating: 100%|█████████████████████████████| 104/104 [00:12<00:00,  8.56it/s]
			2026-05-19 11:08:27,575 - ForIntent - INFO - Inference time: 23.5900 ms per batch
			2026-05-19 11:08:27,576 - ForIntent - INFO - Per-class Intent Accuracy:
			2026-05-19 11:08:27,576 - ForIntent - INFO - intent_accuracy:0.3496
			2026-05-19 11:08:27,576 - ForIntent - INFO - mean_intent_accuracy:0.3325
			'''

			# 构造新的 logits
			new_intent_logits = intent_logits.clone()
			new_intent_logits[:, :self.num_intent_classes] = torch.log(final_scores + eps)
			output['intent_logits'] = new_intent_logits
			return output

	def cond_prob_enhance_fuse(self, output, target, ues_rel_gt=False, aggr = "mean"):  # 注意参数名应为 use_rel_gt，原文有误
		eps = 1e-8
		obj_labels = output['obj_labels']  # (N,)
		if obj_labels.shape[0] == 0:
			return output
		intent_logits = output['intent_logits']  # (N, I+1)
		relation_logits = output['relation_logits']  # (E, R+1)
		relation_labels = torch.argmax(relation_logits, dim=-1)  # (E,)
		edge_index = output['edge_index']  # (2, E)

		# 获取主语节点索引、主语类别、宾语类别
		sub_indices = edge_index[0]  # (E,)
		obj_indices = edge_index[1]  # (E,)
		sub_classes = obj_labels[sub_indices]  # (E,)
		obj_classes = obj_labels[obj_indices]  # (E,)
		rel_labels = relation_labels  # (E,)

		# 有效关系掩码：排除背景类（关系类别索引 == R）
		valid_mask = rel_labels < self.num_relation_classes  # (E,)

		# 如果没有有效关系，所有实体的先验设为全0
		if not valid_mask.any():
			prior_per_node = torch.zeros(obj_labels.shape[0], self.num_intent_classes, device=self.device)
		else:
			# 提取有效边信息
			valid_sub_indices = sub_indices[valid_mask]
			valid_sub_classes = sub_classes[valid_mask]
			valid_obj_classes = obj_classes[valid_mask]
			valid_rel_labels = rel_labels[valid_mask]

			# 计算每条有效边的先验分布 P(I | S, R, O)  (E_valid, I)
			# 从 self.P_I_given_TRTo 中索引：形状 (O, R, O, I)
			intent_sro = self.P_I_given_TRTo[valid_sub_classes, valid_rel_labels, valid_obj_classes, :]  # (E_valid, I)
			intent_s = self.P_I_given_T[valid_sub_classes, :]  # (E_valid, I)
			# 条件概率：P(I|S,R,O) = count(S,R,O,I) / count(S,I)
			prior_per_edge = intent_sro / (intent_s.clamp(min=1))  # (E_valid, I)
			# 分母为零的位置，概率置0（避免NaN）
			prior_per_edge[intent_s == 0] = 0.0

			# 按节点索引聚合：对每个节点的多条边取平均
			num_nodes = obj_labels.shape[0]
			prior_per_node = torch.zeros(num_nodes, self.num_intent_classes, device=self.device)

			if aggr == "max": # max效果变差
				# 手动循环：每个节点取所有关联边的先验在每个意图维度上的最大值
				for i in range(valid_sub_indices.shape[0]):  # 遍历每条有效边
					node = valid_sub_indices[i]  # 主语节点索引
					prior = prior_per_edge[i]  # (I,)
					# 逐元素取最大值
					prior_per_node[node] = torch.max(prior_per_node[node], prior)
			else:
				# 使用 scatter_add 先求和，再除以度数
				ones = torch.ones(valid_sub_indices.shape[0], 1, device=self.device)
				degree = torch.zeros(num_nodes, 1, device=self.device)
				degree.scatter_add_(0, valid_sub_indices.unsqueeze(1), ones)
				degree = degree.clamp(min=1)  # 避免除零
				prior_per_node.scatter_add_(0, valid_sub_indices.unsqueeze(1).expand(-1, self.num_intent_classes),
				                            prior_per_edge)
				prior_per_node /= degree  # 平均

		# 原始意图分数（去掉背景类）
		# original_scores = intent_scores[:, :self.num_intent_classes]  # (N, I)

		intent_scores = torch.softmax(intent_logits[:, :-1], dim=-1)
		intent_scores_ = torch.softmax(intent_scores / self.P_I_prime, dim=-1)
		original_scores = 0.55 * intent_scores + 0.45 * intent_scores_  # alpha=0.55
		prior_per_node[:, self.P_I > 20] = 0
		# 线性组合：final = alpha * original + (1-alpha) * prior
		final_scores = original_scores + 5 * prior_per_node
		# 重新归一化（可选，实际上 softmax 后再 log 等价于直接 log）
		# final_scores = final_scores / (final_scores.sum(dim=-1, keepdim=True) + eps)
		'''
		2026-05-19 10:11:11,803 - ForIntent - INFO - Testing with alpha=0.10
		2026-05-19 10:11:24,019 - ForIntent - INFO - Inference time: 23.7837 ms per batch
		2026-05-19 10:11:24,020 - ForIntent - INFO - intent_accuracy:0.3323
		2026-05-19 10:11:24,020 - ForIntent - INFO - mean_intent_accuracy:0.3488
		
		2026-05-19 12:19:16,656 - ForIntent - INFO - intent_accuracy:0.3243
		2026-05-19 12:19:16,656 - ForIntent - INFO - mean_intent_accuracy:0.3709
		'''

		# 构造新的 logits
		new_intent_logits = intent_logits.clone()
		new_intent_logits[:, :self.num_intent_classes] = torch.log(final_scores + eps)
		output['intent_logits'] = new_intent_logits
		return output


	def cond_prob_enhance_with_alpha(self, output, target, ues_rel_gt=False):
		'''
		本方法
		2026-05-19 09:55:13,754 - ForIntent - INFO - intent_accuracy:0.3318
		2026-05-19 09:55:13,754 - ForIntent - INFO - mean_intent_accuracy:0.3431
		原始：
		2026-05-19 10:15:14,403 - ForIntent - INFO - intent_accuracy:0.3588
		2026-05-19 10:15:14,403 - ForIntent - INFO - mean_intent_accuracy:0.2681
		'''
		eps = 1e-8
		obj_labels = output['obj_labels']  # (N,)
		if obj_labels.shape[0] == 0:
			return output
		intent_logits = output['intent_logits']  # (N, I+1)

		I = self.num_intent_classes

		intent_scores = torch.softmax(intent_logits[:, :-1], dim=-1)
		intent_scores_ = torch.softmax(intent_scores / self.P_I_prime, dim=-1)
		intent_scores_final = self.alpha * intent_scores + (1 - self.alpha) * intent_scores_  # alpha=0.55

		new_intent_logits = intent_logits.clone()
		new_intent_logits[:, :I] = torch.log(intent_scores_final + eps)
		# new_intent_logits[:, -1] = torch.log(bg + eps)
		output['intent_logits'] = new_intent_logits
		return output
