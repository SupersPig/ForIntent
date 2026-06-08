import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# ==================== 3. GNN 模型（不含检测部分） ====================.
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # self.W = nn.Linear(in_dim, out_dim, bias=False)

        hidden_dim = in_dim * 2  # 隐藏层维度可调
        self.W = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.W.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier/Glorot 初始化（适用于 tanh/ReLU）
            init.xavier_uniform_(m.weight)
            # 或者使用 Kaiming 初始化（更适合 ReLU）
            # init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj + torch.eye(num_nodes, device=x.device)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        out = norm_adj @ x
        out = self.W(out)
        return F.relu(out)

class ResidualGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn = GCNLayer(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        identity = self.shortcut(x)
        out = self.gcn(x, edge_index)
        out = self.norm(out)
        out = self.dropout(out)
        return F.relu(out + identity)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        N = x.size(0)
        x = self.W(x).view(N, self.num_heads, self.head_dim)
        src, dst = edge_index

        x_src = x[src]
        x_dst = x[dst]
        concat = torch.cat([x_src, x_dst], dim=-1)
        e = self.leaky_relu(torch.einsum('ehd,hd->eh', concat, self.a))

        # 使用循环计算 softmax 归一化
        max_val = torch.full((N, self.num_heads), -1e9, device=x.device)
        unique_dst = torch.unique(dst)
        for u in unique_dst:
            mask = (dst == u)
            if mask.any():
                max_val[u] = e[mask].max(dim=0)[0]

        e = e - max_val[dst]
        exp = torch.exp(e)

        sum_exp = torch.zeros(N, self.num_heads, device=x.device)
        for u in unique_dst:
            mask = (dst == u)
            if mask.any():
                sum_exp[u] = exp[mask].sum(dim=0)

        att = exp / (sum_exp[dst] + 1e-8)
        att = self.dropout(att)

        # 聚合消息
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        for u in unique_dst:
            mask = (dst == u)
            if mask.any():
                att_u = att[mask].unsqueeze(-1)
                x_src_u = x_src[mask]
                out[u] = (att_u * x_src_u).sum(dim=0)

        return F.elu(out.view(N, -1))

# GNN 模型，包含多个 GNN 层（GCN/GAT/残差GCN），输出节点和边的特征
class SceneGraphGNN(nn.Module):
    def __init__(self, args, node_feat_dim, hidden_dim=256, num_gnn_layers=2):
        super().__init__()
        self.node_gnn = nn.ModuleList()
        assert args.gnn_type in ['gcn', 'gat', 'res_gcn']

        if args.gnn_type == 'gcn':
            self.gnn_layer = GCNLayer(node_feat_dim, hidden_dim)
        elif args.gnn_type == 'gat':
            self.gnn_layer = GATLayer(node_feat_dim, hidden_dim)
        elif args.gnn_type == 'res_gcn':
            self.gnn_layer = ResidualGCNBlock(node_feat_dim, hidden_dim)
        else:
            raise NotImplementedError


        for _ in range(num_gnn_layers):
            self.node_gnn.append(self.gnn_layer)
            # self.node_gnn.append(GATLayer(hidden_dim, hidden_dim))
            # self.node_gnn.append(ResidualGCNBlock(hidden_dim, hidden_dim))
        self.node_dim = hidden_dim

    def forward(self, x, edge_index):
        for layer in self.node_gnn:
            x = layer(x, edge_index)
        src, dst = edge_index
        # edge_feats = torch.cat([x[src], x[dst]], dim=1)
        edge_feats = x[src] - x[dst]
        return x, edge_feats

# =================== 4. 场景图模型 ====================
from .ConditionalProbability import EnhancedByPrior
class UAVSceneGraphModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_object_classes = len(args.entity_categories)
        self.num_relation_classes = len(args.rel_categories)
        self.num_intent_classes = len(args.intent_categories)

        self.use_spatial_feat = args.use_spatial_feat
        self.use_class_embed = args.use_class_embed
        self.gnn_hidden_dim = args.gnn_hidden_dim
        # YOLO 特征向量维度
        yolo_feat_dim = 256 if args.detect_model.endswith('n') else args.yolo_feat_dim
        self.use_gt = args.use_gt
        self.args = args
        self.post = args.post

        # 类别 embedding
        if self.use_class_embed:
            self.class_embed = nn.Embedding(self.num_object_classes, 64)

        # 空间特征 MLP
        assert args.use_spatial_feat in ['none', 'mlp', 'manual']
        if self.use_spatial_feat == 'mlp':
            self.spatial_mlp = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 32)
            )
            node_feat_dim = yolo_feat_dim + (64 if self.use_class_embed else 0) + 32
        else:
            node_feat_dim = yolo_feat_dim + (64 if self.use_class_embed else 0)

        # 节点特征投影
        self.node_proj = nn.Linear(node_feat_dim, args.gnn_hidden_dim)
        self.gnn = SceneGraphGNN(args, self.gnn_hidden_dim, args.gnn_hidden_dim, args.gnn_layers)

        # 预测头
        self.intent_head = nn.Linear(self.gnn_hidden_dim + node_feat_dim, self.num_intent_classes+1)
        self.relation_head = nn.Linear(self.gnn_hidden_dim + node_feat_dim, self.num_relation_classes+1)

        if self.args.use_rel:
            for name, param in self.relation_head.named_parameters():
                param.requires_grad = False

        if self.post:
            self.corrector = EnhancedByPrior(args)

        self._init_weights()

    def build_graph(self, boxes, features, labels, image_size):
        """
        boxes: (N,4) absolute (x1,y1,x2,y2)
        features: (N, feat_dim) from YOLO
        labels: (N,)
        image_size: (H, W)
        """
        N = boxes.shape[0]
        if N == 0:
            return (torch.zeros(0, self.node_proj.in_features, device=boxes.device),
                    torch.zeros(2, 0, dtype=torch.long, device=boxes.device))

        feats = [features]
        if self.use_class_embed:
            class_emb = self.class_embed(labels)
            feats.append(class_emb)

        if self.use_spatial_feat == 'mlp':
            H, W = image_size
            x1 = boxes[:, 0] / W
            y1 = boxes[:, 1] / H
            x2 = boxes[:, 2] / W
            y2 = boxes[:, 3] / H
            spatial = torch.stack([x1, y1, x2, y2], dim=1)
            spatial_feat = self.spatial_mlp(spatial)
            feats.append(spatial_feat)

        node_feats = torch.cat(feats, dim=1)

        # 全连接边（无自环）
        src, dst = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=boxes.device)
        return node_feats, edge_index

    def forward(self, targets):
        """
        targets: list of dict, 每个 dict 包含 'boxes', 'labels', 'features', 'orig_size'
        """
        outputs = []
        for t in targets:
            boxes = t['boxes']
            labels = t['labels']
            features = t['features']
            img_size = t['orig_size'].tolist()  # (H, W)

            if boxes.shape[0] == 0:
                outputs.append({
                    'boxes': boxes,
                    'obj_labels': labels,
                    'intent_logits': torch.zeros(0, self.num_intent_classes, device=boxes.device),
                    'relation_logits': torch.zeros(0, self.num_relation_classes, device=boxes.device),
                    'edge_index': torch.zeros(2, 0, dtype=torch.long, device=boxes.device),
                })
                continue

            node_feats, edge_index = self.build_graph(boxes, features, labels, img_size)

            # 使用神经网络更新节点
            node_feats_g = self.node_proj(node_feats)
            updated_nodes, updated_edge_feats = self.gnn(node_feats_g, edge_index)

            # 预测意图
            intent_logits = self.intent_head(torch.cat([node_feats, updated_nodes], dim=-1))

            # 预测关系
            src, dst = edge_index
            edge_feats = node_feats[src] - node_feats[dst]
            relation_logits = self.relation_head(torch.cat([edge_feats, updated_edge_feats], dim=-1))

            outputs.append({
                'boxes': boxes,
                'obj_labels': labels,
                'intent_logits': intent_logits,
                'relation_logits': relation_logits,
                'edge_index': edge_index,
            })

        if self.post:
            outputs = self.postprocess(outputs, targets)

        return outputs

    def postprocess(self, outputs, targets):
        for idx, (output, target) in enumerate(zip(outputs, targets)):
            pp_output = self.corrector.cond_prob_enhance(output, target)
            # pp_output = self.corrector.cond_prob_enhance_fuse(output, target)
            # pp_output = self.corrector.cond_prob_enhance_with_alpha(output, target)
            outputs[idx] = pp_output
        return outputs

    def compute_loss(self, predictions, targets):
        total_loss = torch.tensor(0.0, device=predictions[0]['boxes'].device)
        if self.use_gt:
            for pred, target in zip(predictions, targets):
                if pred['boxes'].shape[0] == 0:
                    continue

                gt_intents = target['intent_annotations'].to(pred['boxes'].device)
                gt_rels = target['rel_annotations'].to(pred['boxes'].device)

                # 意图损失（每个预测框对应一个真实意图）
                if gt_intents.shape[0] > 0:
                    # 构建意图标签：按照预测框的顺序（认为预测框与真实框一一对应）
                    intent_gt = torch.full((pred['boxes'].shape[0],), self.num_intent_classes, dtype=torch.long,
                                           device=pred['boxes'].device)
                    for i in range(pred['boxes'].shape[0]):
                        mask = (gt_intents[:, 0] == i)  # i 是目标ID，与预测框索引对应
                        if mask.any():
                            intent_gt[i] = gt_intents[mask, 1][0]

                    valid_intent = intent_gt != self.num_intent_classes
                    if valid_intent.any():
                        pred_intent = pred['intent_logits'][valid_intent]
                        intent_loss = F.cross_entropy(pred_intent, intent_gt[valid_intent])
                        total_loss += intent_loss

                # 关系损失
                if gt_rels.shape[0] > 0 and self.args.use_rel:
                    edge_index = pred['edge_index']
                    num_edges = edge_index.shape[1]
                    rel_gt = torch.full((num_edges,), self.num_relation_classes, dtype=torch.long,
                                        device=pred['boxes'].device)

                    # 构建节点匹配：预测框索引 = 真实目标ID
                    for e in range(num_edges):
                        src = edge_index[0, e].item()
                        dst = edge_index[1, e].item()
                        # 在真实关系中查找（sub, obj, rel_label）
                        mask = (gt_rels[:, 0] == src) & (gt_rels[:, 1] == dst)
                        if mask.any():
                            rel_gt[e] = gt_rels[mask, 2][0]

                    valid_rel = rel_gt != self.num_relation_classes
                    if valid_rel.any():
                        pred_rel = pred['relation_logits'][valid_rel]
                        rel_loss = F.cross_entropy(pred_rel, rel_gt[valid_rel])
                        total_loss += rel_loss
        else:
            for pred, target in zip(predictions, targets):
                if pred['boxes'].shape[0] == 0:
                    continue
                gt_boxes = target['boxes'].to(pred['boxes'].device)
                gt_intents = target['intent_annotations'].to(pred['boxes'].device)
                gt_rels = target['rel_annotations'].to(pred['boxes'].device)

                # 匹配预测框与真实框
                ious = self.box_iou(pred['boxes'], gt_boxes)
                pred_to_gt = torch.argmax(ious, dim=1)
                matched = ious[torch.arange(pred['boxes'].shape[0]), pred_to_gt] > 0.5
                matched_indices = torch.where(matched)[0]
                if len(matched_indices) == 0:
                    continue

                # 意图损失
                matched_gt_idx = pred_to_gt[matched_indices]
                if gt_intents.shape[0] > 0:
                    intent_gt = torch.full((len(matched_indices),), self.num_intent_classes, dtype=torch.long, device=pred['boxes'].device)
                    for i, gt_idx in enumerate(matched_gt_idx):
                        mask = (gt_intents[:, 0] == gt_idx)
                        if mask.any():
                            intent_gt[i] = gt_intents[mask, 1][0]
                    valid_intent = intent_gt != -1
                    if valid_intent.any():
                        intent_gt_shifted = intent_gt[valid_intent] # 加1把第一个留给背景类
                        pred_intent_ = pred['intent_logits'][matched_indices][valid_intent] # 加1把第一个留给背景类
                        intent_loss = F.cross_entropy(pred_intent_, intent_gt_shifted)
                        total_loss += intent_loss

                # 关系损失
                if gt_rels.shape[0] > 0 and self.args.use_rel:
                    edge_index = pred['edge_index']
                    num_edges = edge_index.shape[1]
                    rel_gt = torch.full((num_edges,), self.num_relation_classes, dtype=torch.long, device=pred['boxes'].device)
                    node_match = torch.full((pred['boxes'].shape[0],), -1, dtype=torch.long, device=pred['boxes'].device)
                    node_match[matched_indices] = matched_gt_idx
                    for e in range(num_edges):
                        src, dst = edge_index[0, e], edge_index[1, e]

                        if node_match[src] != -1 and node_match[dst] != -1:
                            sub_gt = node_match[src]
                            obj_gt = node_match[dst]
                            mask = (gt_rels[:, 0] == sub_gt) & (gt_rels[:, 1] == obj_gt)

                            if mask.any():
                                rel_gt[e] = gt_rels[mask, 2][0]

                    valid_rel = rel_gt != -1
                    if valid_rel.any():
                        rel_gt_shifted = rel_gt[valid_rel] # 没有加1，最后一个是背景类
                        pred_rel_ = pred['relation_logits'][valid_rel]
                        rel_loss = F.cross_entropy(pred_rel_, rel_gt_shifted)
                        total_loss += rel_loss

        return total_loss

    @staticmethod
    def box_iou(boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, None] + area2 - inter + 1e-6)
        return iou

    def _init_weights(self):
        """初始化除 gnn 以外的所有参数"""
        # 初始化 class_embed（如果存在）
        if hasattr(self, 'class_embed'):
            init.normal_(self.class_embed.weight, mean=0, std=0.01)

        # 初始化 spatial_mlp（如果存在）
        if hasattr(self, 'spatial_mlp'):
            for m in self.spatial_mlp.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

        # 初始化 node_proj
        if hasattr(self, 'node_proj'):
            init.kaiming_uniform_(self.node_proj.weight, mode='fan_in', nonlinearity='relu')
            if self.node_proj.bias is not None:
                init.constant_(self.node_proj.bias, 0)

        # 初始化 intent_head
        if hasattr(self, 'intent_head'):
            init.kaiming_uniform_(self.intent_head.weight, mode='fan_in', nonlinearity='relu')
            if self.intent_head.bias is not None:
                init.constant_(self.intent_head.bias, 0)

        # 初始化 relation_head
        if hasattr(self, 'relation_head'):
            init.kaiming_uniform_(self.relation_head.weight, mode='fan_in', nonlinearity='relu')
            if self.relation_head.bias is not None:
                init.constant_(self.relation_head.bias, 0)

