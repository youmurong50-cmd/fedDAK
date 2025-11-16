import copy
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader
from dataset.utils import read_client_data
from model import create_model  # 导入更新后的模型


class FedDAKClient:
    def __init__(self, args, client_id):
        self.experiment_name = args.experiment_name
        self.client_id = client_id
        self.device = args.device
        self.base_data_dir = args.base_data_dir
        self.dataset = args.dataset
        self.train_samples = 0  # 本地训练样本数
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.distill_weight = args.distill_weight  # 蒸馏基础权重

        # 初始化模型（不含feature_interaction）
        self.model = create_model(args.dataset).to(args.device)
        self.global_model = copy.deepcopy(self.model)  # 教师模型（全局模型）

        # 损失函数：仅分类损失（不使用 logits KD）
        self.cls_loss = nn.CrossEntropyLoss().to(self.device)
        # self.kl_loss = nn.KLDivLoss(reduction="batchmean").to(self.device)  # 不再使用 logits KD

        # 优化器：GFE+分类头（主优化器）
        self.opt_classification = torch.optim.Adam(
            list(self.model.gfe.parameters()) +
            list(self.model.phead.parameters()),  # 仅优化GFE和分类头
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # 优化器：CSFE微调（低学习率，仅数据量充足时启用）
        self.opt_csfe = torch.optim.Adam(
            self.model.csfe.parameters(),
            lr=args.lr * 0.1,  # 学习率为GFE的1/10
            weight_decay=args.weight_decay
        )

        # 数据加载（会自动计算本地类别稀缺性）
        self.train_loader = self.load_train_data()
        self.test_loader = self.load_test_data()

        # 新增：全局数据分布和CSFE微调阈值
        self.global_class_dist = None
        self.csfe_finetune_threshold = getattr(args, 'csfe_finetune_threshold', 1000)

        # 新增：本地类别数量统计和稀缺性权重（在load_train_data中初始化）
        self.local_class_counts = {}  # 键：类别，值：该类别在本地的样本数
        self.class_scarcity = {}  # 键：类别，值：稀缺性权重（会在训练前基于全局分布/本地分布计算）

    def set_model(self, model_dict):
        """从服务器更新GFE和PHead参数（不含feature_interaction）"""
        self.model.gfe.load_state_dict(model_dict)  # 直接加载 state_dict
        self.global_model.gfe.load_state_dict(model_dict)  # 直接加载 state_dict

    def set_global_distribution(self, global_dist):
        """由服务器下发全局数据的类别分布"""
        self.global_class_dist = global_dist
        # 收到全局分布后，重新计算本地的稀缺性权重（若本地有数据）
        if self.train_samples > 0:
            self._recompute_class_scarcity()

    def _compute_local_class_dist(self):
        """计算本地训练数据的类别分布（添加空数据检查）"""
        if self.train_samples == 0:
            return {}

        class_counts = {}
        for _, y in self.train_loader:
            for cls in y.cpu().numpy():
                class_counts[cls] = class_counts.get(cls, 0) + 1
        total = sum(class_counts.values())
        return {k: v / total for k, v in class_counts.items()}

    def _compute_heterogeneity(self):
        """计算本地与全局数据分布的KL散度（显式归一化概率）"""
        if self.global_class_dist is None or not self.global_class_dist:
            return torch.tensor(0.0, device=self.device)

        local_dist = self._compute_local_class_dist()
        if not local_dist:
            return torch.tensor(0.0, device=self.device)

        # 统一类别列表
        all_classes = set(self.global_class_dist.keys()).union(set(local_dist.keys()))
        all_classes = sorted(all_classes)

        # 构建概率分布（显式归一化）
        global_probs = torch.tensor(
            [self.global_class_dist.get(c, 0) for c in all_classes],
            dtype=torch.float32,
            device=self.device
        )
        global_probs = global_probs / global_probs.sum()

        local_probs = torch.tensor(
            [local_dist.get(c, 0) for c in all_classes],
            dtype=torch.float32,
            device=self.device
        )
        local_probs = local_probs / local_probs.sum()

        # 计算KL散度
        kl_div = dist.kl_divergence(
            dist.Categorical(probs=local_probs),
            dist.Categorical(probs=global_probs)
        )
        return kl_div

    def test_metrics(self):
        """评估模型在测试集上的性能"""
        self.model.eval()
        test_corrects = 0
        test_cls_loss = 0.0
        test_num_samples = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits, _, _ = self.model.classification(x)
                test_corrects += (torch.argmax(logits, dim=1) == y).sum().item()
                test_cls_loss += self.cls_loss(logits, y).item() * y.shape[0]
                test_num_samples += y.shape[0]

        return {
            "test_num_samples": test_num_samples,
            "test_corrects": test_corrects,
            "test_cls_loss": test_cls_loss / max(test_num_samples, 1)
        }

    def train_metrics(self):
        """评估模型在训练集上的性能"""
        self.model.eval()
        train_corrects = 0
        train_cls_loss = 0.0
        train_num_samples = 0

        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits, _, _ = self.model.classification(x)
                train_corrects += (torch.argmax(logits, dim=1) == y).sum().item()
                train_cls_loss += self.cls_loss(logits, y).item() * y.shape[0]
                train_num_samples += y.shape[0]

        return {
            "train_num_samples": train_num_samples,
            "train_corrects": train_corrects,
            "train_cls_loss": train_cls_loss / max(train_num_samples, 1)
        }

    def _recompute_class_scarcity(self, beta: float = 2.0, clip_min: float = 1e-6):
        """
        新的稀缺性计算：
        对每类 c 计算 ratio = p_local(c) / p_global(c)（若缺失则使用小常数）
        raw_weight_c = exp(-beta * ratio)
        最后将 raw_weight 归一化并缩放到 mean=1，避免整体放缩改变 loss 的量级。
        """
        # 如果本地没有样本，则置空
        if self.train_samples == 0:
            self.class_scarcity = {}
            return

        # 本地分布与全局分布
        local_dist = self._compute_local_class_dist()
        global_dist = self.global_class_dist if self.global_class_dist else {}

        # 确定所有可能类别集合（优先使用全局类别集合）
        classes = sorted(set(global_dist.keys()) | set(local_dist.keys()))
        raw_weights = {}
        for c in classes:
            p_local = local_dist.get(c, 1e-8)
            p_global = global_dist.get(c, 1e-8)
            ratio = p_local / p_global
            raw_w = float(torch.exp(torch.tensor(-beta * ratio)).item())
            raw_weights[c] = max(raw_w, clip_min)

        # 归一化并缩放使 mean = 1（避免整体改变 loss 量级）
        vals = list(raw_weights.values())
        mean_val = sum(vals) / len(vals) if len(vals) > 0 else 1.0
        self.class_scarcity = {c: (w / mean_val) for c, w in raw_weights.items()}

        # 对于本地不存在但全局存在的类别也有权重；对于本地存在但全局不存在（不常见），保留计算值

    def train_classification_with_distillation(self):
        """带特征蒸馏的本地训练（整合动态权重+稀缺性加权+CSFE微调）"""
        # 决定是否启用CSFE微调
        self.model.csfe_finetune = self.train_samples >= self.csfe_finetune_threshold
        self.model.train()

        # 计算本地与全局数据的异质性
        heterogeneity = self._compute_heterogeneity()
        hetero_weight = torch.exp(-heterogeneity).to(self.device)

        # 在训练前（或每轮开始）确保 class_scarcity 已基于 global distribution 计算
        if self.global_class_dist:
            self._recompute_class_scarcity()
        else:
            # 若没有全局分布，则回退为原始的 total/count 比例并缩放 mean=1
            if self.train_samples > 0 and self.local_class_counts:
                total_samples = self.train_samples
                raw = {cls: float(total_samples / cnt) for cls, cnt in self.local_class_counts.items()}
                mean_val = sum(raw.values()) / len(raw)
                self.class_scarcity = {c: (w / mean_val) for c, w in raw.items()}
            else:
                self.class_scarcity = {}

        for epoch in range(self.local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)  # 当前batch的标签（形状：[batch_size]）

                # 清零梯度
                self.opt_classification.zero_grad()
                if self.model.csfe_finetune:
                    self.opt_csfe.zero_grad()

                # 学生模型前向传播
                student_logits, gf_student, csf = self.model.classification(x)
                cls_loss = self.cls_loss(student_logits, y)

                # 教师模型前向传播（无梯度，仅获取全局 GFE 特征）
                with torch.no_grad():
                    _, gf_teacher, _ = self.global_model.classification(x)

                # --------------------------
                # 特征蒸馏：仅使用 GFE 的特征层 MSE（不涉及 logits 的 KD）
                # --------------------------
                # 1. 为当前batch的每个样本分配稀缺性权重（来自 self.class_scarcity）
                batch_scarcity_weights = []
                for cls in y.cpu().numpy():  # 遍历当前batch的每个样本类别
                    # 若类别不在计算表中（极端情况），默认权重为1.0
                    scarcity = self.class_scarcity.get(int(cls), 1.0)
                    batch_scarcity_weights.append(scarcity)
                # 转换为张量并适配设备（形状：[batch_size, 1]，便于广播）
                scarcity_weight = torch.tensor(
                    batch_scarcity_weights,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(1)

                # 2. 特征层面蒸馏（GFE特征MSE，用稀缺性权重加权）
                mse_per_sample = torch.mean((gf_student - gf_teacher) ** 2, dim=1, keepdim=True)  # [B,1]
                feat_distill_loss = torch.mean(scarcity_weight * mse_per_sample)  # 稀缺样本权重更高

                # 动态调整蒸馏总权重（基于特征差异和分布异质性）
                dist_gap = torch.mean((gf_student - gf_teacher) ** 2).detach()
                gap_weight = torch.exp(-dist_gap)
                final_distill_weight = self.distill_weight * gap_weight * hetero_weight
                # 总损失（无 logits KD）
                # total_loss = cls_loss + final_distill_weight * feat_distill_loss
                total_loss = cls_loss + final_distill_weight * feat_distill_loss
                total_loss.backward()

                # 更新参数
                self.opt_classification.step()
                if self.model.csfe_finetune:
                    self.opt_csfe.step()

    def train(self):
        """启动本地训练"""
        self.train_classification_with_distillation()

    def load_train_data(self):
        """加载本地训练数据，并计算本地类别稀缺性权重（初始）"""
        train_data = read_client_data(
            self.base_data_dir, self.dataset, self.experiment_name, self.client_id, is_train=True
        )
        self.train_samples = len(train_data)
        if self.train_samples == 0:
            # 空数据时重置类别统计
            self.local_class_counts = {}
            self.class_scarcity = {}
            return DataLoader([], self.batch_size, drop_last=False, shuffle=False)

        # 统计本地每个类别的样本数量
        self.local_class_counts = {}
        for _, label in train_data:  # train_data的每个元素是（数据，标签）
            cls = int(label.item())  # 取出标签值
            self.local_class_counts[cls] = self.local_class_counts.get(cls, 0) + 1

        # 初始稀缺性：使用简单比例（后续若收到全局分布会被 _recompute_class_scarcity 覆盖）
        total_samples = self.train_samples
        raw = {
            cls: total_samples / count
            for cls, count in self.local_class_counts.items()
        }
        mean_val = sum(raw.values()) / len(raw)
        self.class_scarcity = {c: (w / mean_val) for c, w in raw.items()}

        return DataLoader(train_data, self.batch_size, drop_last=False, shuffle=True)

    def load_test_data(self):
        """加载本地测试数据"""
        test_data = read_client_data(
            self.base_data_dir, self.dataset, self.experiment_name, self.client_id, is_train=False
        )
        return DataLoader(test_data, self.batch_size, drop_last=False, shuffle=False)
