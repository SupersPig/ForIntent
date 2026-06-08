import torch
import time

from caffe2.perfkernels.hp_emblookup_codegen import args
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
    for targets in pbar:
        # 将 targets 中的所有张量移到 device
        for t in targets:
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    t[k] = v.to(device)
        optimizer.zero_grad()
        predictions = model(targets)
        loss = model.compute_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(dataloader)

def box_iou(boxes1, boxes2):
    """计算边界框 IoU"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter + 1e-6)
    return iou


@torch.no_grad()
def evaluate_with_gt(model, dataloader, device, args, recall_k_list=[20, 50, 100], has_background=True, logger=None):
    model.eval()
    total_intent_correct = 0
    total_intent_count = 0

    # 意图分类别统计
    num_intents = model.num_intent_classes - 1 if has_background else model.num_intent_classes
    intent_correct_counts = torch.zeros(num_intents, dtype=torch.long, device='cpu')
    intent_total_counts = torch.zeros(num_intents, dtype=torch.long, device='cpu')

    # 关系类别统计
    num_relations = model.num_relation_classes
    gt_rel_counts = torch.zeros(num_relations, dtype=torch.long, device='cpu')
    hit_rel_counts = {k: torch.zeros(num_relations, dtype=torch.long, device='cpu') for k in recall_k_list}
    total_gt_triplets = 0
    total_hit_triplets = {k: 0 for k in recall_k_list}
    times_model = []
    with torch.no_grad():
        for targets in tqdm(dataloader, desc="Evaluating"):
            for t in targets:
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t[k] = v.to(device)

            start_time = time.perf_counter()
            predictions = model(targets)
            end_time = time.perf_counter()
            times_model.append((end_time - start_time) * 1000)

            for pred, target in zip(predictions, targets):
                if pred['boxes'].shape[0] == 0:
                    continue

                gt_intents = target['intent_annotations']
                gt_rels = target['rel_annotations']

                # ----- 意图准确率评估（直接使用真实框索引）-----
                if gt_intents.shape[0] > 0:
                    # 构建意图标签：预测框与真实框一一对应（索引一致）
                    intent_gt = torch.full((pred['boxes'].shape[0],), -1, dtype=torch.long, device=device)
                    for i in range(pred['boxes'].shape[0]):
                        mask = (gt_intents[:, 0] == i)  # i 是目标ID
                        if mask.any():
                            intent_gt[i] = gt_intents[mask, 1][0]

                    valid = intent_gt != -1
                    if valid.any():
                        pred_intent_logits = pred['intent_logits'][valid]
                        if has_background:
                            pred_intent = torch.argmax(pred_intent_logits[:, :-1], dim=1)
                        else:
                            pred_intent = torch.argmax(pred_intent_logits, dim=1)

                        correct = (pred_intent == intent_gt[valid]).sum().item()
                        total_intent_correct += correct
                        total_intent_count += len(valid)

                        # 更新分类别统计
                        pred_intent_cpu = pred_intent.cpu()
                        true_intent_cpu = intent_gt[valid].cpu()
                        for pc, tc in zip(pred_intent_cpu, true_intent_cpu):
                            intent_total_counts[tc] += 1
                            if pc == tc:
                                intent_correct_counts[tc] += 1

                # ----- 关系评估（直接使用真实框索引）-----
                if gt_rels.shape[0] == 0:
                    continue

                edge_index = pred['edge_index']
                rel_logits = pred['relation_logits']
                rel_scores = torch.softmax(rel_logits, dim=1)

                if has_background:
                    rel_scores_valid = rel_scores[:, :-1]
                    pred_rel_class = torch.argmax(rel_scores_valid, dim=1)
                    pred_rel_scores = torch.max(rel_scores_valid, dim=1)[0]
                else:
                    pred_rel_class = torch.argmax(rel_scores, dim=1)
                    pred_rel_scores = torch.max(rel_scores, dim=1)[0]

                # 构建预测三元组（直接使用预测框索引作为目标ID）
                pred_triplets = []
                for e in range(edge_index.shape[1]):
                    src = edge_index[0, e].item()
                    dst = edge_index[1, e].item()
                    rel_pred = pred_rel_class[e].item()
                    rel_score = pred_rel_scores[e].item()
                    pred_triplets.append((src, rel_pred, dst, rel_score))

                pred_triplets.sort(key=lambda x: x[3], reverse=True)

                # 构建真实三元组
                gt_triplets = set()
                for rel in gt_rels.cpu().numpy():
                    sub, obj, rel_label = int(rel[0]), int(rel[1]), int(rel[2])
                    gt_triplets.add((sub, rel_label, obj))

                # 更新统计
                total_gt_triplets += len(gt_triplets)
                for sub, rel, obj in gt_triplets:
                    gt_rel_counts[rel] += 1

                for k in recall_k_list:
                    hit_set = set()
                    for i in range(min(k, len(pred_triplets))):
                        sub, rel, obj, _ = pred_triplets[i]
                        if (sub, rel, obj) in gt_triplets:
                            hit_set.add((sub, rel, obj))
                    total_hit_triplets[k] += len(hit_set)
                    for sub, rel, obj in hit_set:
                        hit_rel_counts[k][rel] += 1

    # 计算结果
    recall = {}
    for k in recall_k_list:
        recall[f'recall@{k}'] = total_hit_triplets[k] / total_gt_triplets if total_gt_triplets > 0 else 0.0

    per_class_recall = {k: torch.zeros(num_relations) for k in recall_k_list}
    for k in recall_k_list:
        for r in range(num_relations):
            if gt_rel_counts[r] > 0:
                per_class_recall[k][r] = hit_rel_counts[k][r] / gt_rel_counts[r]

    mean_recall = {}
    for k in recall_k_list:
        valid_recall = per_class_recall[k][gt_rel_counts > 0]
        mean_recall[f'mR@{k}'] = valid_recall.mean().item() if len(valid_recall) > 0 else 0.0

    # 意图准确率
    intent_accuracy = total_intent_correct / total_intent_count if total_intent_count > 0 else 0.0

    # 意图分类别准确率
    per_class_intent_acc = intent_correct_counts / (intent_total_counts + 1e-8)
    mean_intent_acc = per_class_intent_acc[intent_total_counts > 0].mean().item() if (
                intent_total_counts > 0).any() else 0.0
    intent_class_acc = {i: per_class_intent_acc[i].item() for i in range(num_intents) if intent_total_counts[i] > 0}

    results = {
        'intent_accuracy': intent_accuracy,
        'mean_intent_accuracy': mean_intent_acc,
        'per_class_intent_accuracy': intent_class_acc,
        **recall,
        **mean_recall,
        'per_class_recall': [per_class_recall[D].cpu().tolist() for D in per_class_recall],
        'gt_rel_counts': gt_rel_counts.cpu().tolist(),
        "infer_time": sum(times_model) / len(times_model)
    }

    if not args.eval or not logger:
        return results

    # 打印结果
    logger.info(f"Inference time: {results['infer_time']:.4f} ms per batch")
    logger.info("Per-class Intent Accuracy:")
    logger.info(f"intent_accuracy:{results['intent_accuracy']:.4f}")
    logger.info(f"mean_intent_accuracy:{results['mean_intent_accuracy']:.4f}")
    intent_categories = dataloader.dataset.intent_categories
    rel_categories = dataloader.dataset.rel_categories
    if args.evalDetails:
        for i, acc in intent_class_acc.items():
            total = intent_total_counts[i].item()
            logger.info(f"Intent {intent_categories[i]}: acc={acc:.4f}, total={total}")

        logger.info("Relation\tR@20\tR@50\tR@100\ttotal")
        logger.info(f"R@20:{results['recall@20']:.4f}\tR@50:{results['recall@50']:.4f}\tR@100:{results['recall@100']:.4f}")
        logger.info(f"mR@20:{results['mR@20']:.4f}\tmR@50:{results['mR@50']:.4f}\tmR@100:{results['mR@100']:.4f}")

        logger.info("Per-class Recall@20\t50\t100\ttotal")

        for r in range(num_relations):
            if gt_rel_counts[r] > 0:
                logger.info(
                    f"{rel_categories[r]}:\t{per_class_recall[20][r]:.4f}\t{per_class_recall[50][r]:.4f}\t{per_class_recall[100][r]:.4f}\t{gt_rel_counts[r]}")
    results["intent_categories"] = intent_categories
    results["rel_categories"] = rel_categories
    return results

@torch.no_grad()
def evaluate(model, dataloader, device, args, recall_k_list=[20, 50, 100], has_background=True, use_gt=False, logger=None):

    if use_gt:
        return evaluate_with_gt(model, dataloader, device, args=args, logger=logger)

    model.eval()
    total_intent_correct = 0
    total_intent_count = 0

    # ---------- 新增：意图分类别统计 ----------
    num_intents = model.num_intent_classes # + 1 if has_background else model.num_intent_classes
    intent_correct_counts = torch.zeros(num_intents, dtype=torch.long, device='cpu')
    intent_total_counts = torch.zeros(num_intents, dtype=torch.long, device='cpu')

    # 获取关系类别总数（包含背景）
    num_relations = model.num_relation_classes
    # 统计变量
    gt_rel_counts = torch.zeros(num_relations, dtype=torch.long, device='cpu')
    hit_rel_counts = {k: torch.zeros(num_relations, dtype=torch.long, device='cpu') for k in recall_k_list}
    total_gt_triplets = 0
    total_hit_triplets = {k: 0 for k in recall_k_list}

    times_model = []
    with torch.no_grad():
        for targets in tqdm(dataloader, desc="Evaluating"):
            for t in targets:
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t[k] = v.to(device)

            start_time = time.perf_counter()
            predictions = model(targets)
            end_time = time.perf_counter()
            times_model.append((end_time - start_time) * 1000)

            for pred, target in zip(predictions, targets):
                if pred['boxes'].shape[0] == 0:
                    continue

                gt_boxes = target['boxes']
                gt_intents = target['intent_annotations']
                gt_rels = target['rel_annotations']

                # ----- 意图准确率评估（含分类别统计）-----
                if gt_intents.shape[0] > 0:
                    ious = box_iou(pred['boxes'], gt_boxes)
                    pred_to_gt = torch.argmax(ious, dim=1)
                    matched = ious[torch.arange(pred['boxes'].shape[0]), pred_to_gt] > 0.5
                    matched_indices = torch.where(matched)[0]
                    if len(matched_indices) > 0:
                        matched_gt_idx = pred_to_gt[matched_indices]
                        intent_gt = torch.full((len(matched_indices),), -1, dtype=torch.long, device=device)
                        for i, gt_idx in enumerate(matched_gt_idx):
                            mask = (gt_intents[:, 0] == gt_idx)
                            if mask.any():
                                intent_gt[i] = gt_intents[mask, 1][0]
                        valid = intent_gt != -1
                        if valid.any():
                            pred_intent_logits = pred['intent_logits'][matched_indices][valid]
                            pred_intent = torch.argmax(pred_intent_logits[:, :-1], dim=1)  # 去掉背景
                            correct = (pred_intent == intent_gt[valid]).sum().item()
                            total_intent_correct += correct
                            total_intent_count += len(valid)

                            # ---------- 更新分类别统计 ----------
                            pred_intent_cpu = pred_intent.cpu()
                            true_intent_cpu = intent_gt[valid].cpu()
                            for pc, tc in zip(pred_intent_cpu, true_intent_cpu):
                                intent_total_counts[tc] += 1
                                if pc == tc:
                                    intent_correct_counts[tc] += 1

                # ----- 关系评估（原有代码不变）-----
                if gt_rels.shape[0] == 0:
                    continue

                ious = box_iou(pred['boxes'], gt_boxes)
                pred_to_gt = torch.argmax(ious, dim=1)
                matched = ious[torch.arange(pred['boxes'].shape[0]), pred_to_gt] > 0.5
                matched_indices = torch.where(matched)[0]
                if len(matched_indices) == 0:
                    continue
                matched_gt_idx = pred_to_gt[matched_indices]
                node_match = torch.full((pred['boxes'].shape[0],), -1, dtype=torch.long, device=device)
                node_match[matched_indices] = matched_gt_idx

                edge_index = pred['edge_index']
                rel_logits = pred['relation_logits']
                rel_scores = torch.softmax(rel_logits, dim=1)
                if has_background:
                    rel_scores_valid = rel_scores[:, :-1]
                    pred_rel_class = torch.argmax(rel_scores_valid, dim=1)
                    pred_rel_scores = torch.max(rel_scores_valid, dim=1)[0]
                else:
                    pred_rel_class = torch.argmax(rel_scores, dim=1)
                    pred_rel_scores = torch.max(rel_scores, dim=1)[0]

                pred_triplets = []
                for e in range(edge_index.shape[1]):
                    src = edge_index[0, e].item()
                    dst = edge_index[1, e].item()
                    if node_match[src] != -1 and node_match[dst] != -1:
                        sub_gt = node_match[src].item()
                        obj_gt = node_match[dst].item()
                        rel_pred = pred_rel_class[e].item()
                        rel_score = pred_rel_scores[e].item()
                        pred_triplets.append((sub_gt, rel_pred, obj_gt, rel_score))

                pred_triplets.sort(key=lambda x: x[3], reverse=True)

                gt_triplets = set()
                for rel in gt_rels.cpu().numpy():
                    sub, obj, rel_label = int(rel[0]), int(rel[1]), int(rel[2])
                    gt_triplets.add((sub, rel_label, obj))

                total_gt_triplets += len(gt_triplets)
                for sub, rel, obj in gt_triplets:
                    gt_rel_counts[rel] += 1

                for k in recall_k_list:
                    hit_set = set()
                    for i in range(min(k, len(pred_triplets))):
                        sub, rel, obj, _ = pred_triplets[i]
                        if (sub, rel, obj) in gt_triplets:
                            hit_set.add((sub, rel, obj))
                    total_hit_triplets[k] += len(hit_set)
                    for sub, rel, obj in hit_set:
                        hit_rel_counts[k][rel] += 1

    # ---------- 计算结果 ----------
    recall = {}
    for k in recall_k_list:
        recall[f'recall@{k}'] = total_hit_triplets[k] / total_gt_triplets if total_gt_triplets > 0 else 0.0

    per_class_recall = {k: torch.zeros(num_relations) for k in recall_k_list}
    for k in recall_k_list:
        for r in range(num_relations):
            if gt_rel_counts[r] > 0:
                per_class_recall[k][r] = hit_rel_counts[k][r] / gt_rel_counts[r]

    mean_recall = {}
    for k in recall_k_list:
        valid_recall = per_class_recall[k][gt_rel_counts > 0]
        mean_recall[f'mR@{k}'] = valid_recall.mean().item() if len(valid_recall) > 0 else 0.0

    # 意图整体准确率
    intent_accuracy = total_intent_correct / total_intent_count if total_intent_count > 0 else 0.0

    # ---------- 意图分类别准确率及平均准确率 ----------
    per_class_intent_acc = intent_correct_counts / (intent_total_counts + 1e-8)
    mean_intent_acc = per_class_intent_acc[intent_total_counts > 0].mean().item() if (intent_total_counts > 0).any() else 0.0

    # 将每个意图类别的准确率放入字典（可选）
    intent_class_acc = {i: per_class_intent_acc[i].item() for i in range(num_intents) if intent_total_counts[i] > 0}

    results = {
        'intent_accuracy': intent_accuracy,
        'mean_intent_accuracy': mean_intent_acc,
        'per_class_intent_accuracy': intent_class_acc,
        **recall,
        **mean_recall,
        'per_class_recall': [per_class_recall[D].cpu().tolist() for D in per_class_recall],
        'gt_rel_counts': gt_rel_counts.cpu().tolist(),
        "infer_time": sum(times_model) / len(times_model)
    }

    if not args.eval:
        return results

    # 打印意图分类别准确率
    logger.info(f"Inference time: {results['infer_time']:.4f} ms per batch")
    logger.info("Per-class Intent Accuracy:")
    logger.info(f"intent_accuracy:{results['intent_accuracy']:.4f}")
    logger.info(f"mean_intent_accuracy:{results['mean_intent_accuracy']:.4f}")
    intent_categories = dataloader.dataset.intent_categories
    rel_categories = dataloader.dataset.rel_categories
    if args.evalDetails:
        for i, acc in intent_class_acc.items():
            total = intent_total_counts[i].item()
            logger.info(f"Int {i}: {intent_categories[i]}: acc={acc:.4f}, total={total}")

        logger.info("Relation\tR@20\tR@50\tR@100\ttotal")
        logger.info(f"R@20:{results['recall@20']:.4f}\tR@50:{results['recall@50']:.4f}\tR@100:{results['recall@100']:.4f}")
        logger.info(f"mR@20:{results['mR@20']:.4f}\tmR@50:{results['mR@50']:.4f}\tmR@100:{results['mR@100']:.4f}")
        # 打印关系分类别Recall（原有）
        logger.info("Per-class Recall@20\t50\t100\ttotal")
        # 假设数据集有 rel_categories 属性
        for r in range(num_relations):
            if gt_rel_counts[r] > 0:
                logger.info(f"Rel {r}: {rel_categories[r]}:\t{per_class_recall[20][r]:.4f}\t{per_class_recall[50][r]:.4f}\t{per_class_recall[100][r]:.4f}\t{gt_rel_counts[r]}")

    return results
