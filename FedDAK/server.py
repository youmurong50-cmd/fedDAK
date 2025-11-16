import copy
import torch
import time
import numpy as np
from client import FedDAKClient
from model import create_model
import torch.distributions as dist  ### MODIFIED: Added for KL divergence calculation
import os  # ADDED: for file path handling


class FedDAKServer:
    def __init__(self, args):
        self.args = args
        self.global_epochs = args.global_epochs
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.current_num_join_clients = self.num_join_clients
        self.eval_interval = args.eval_interval
        self.device = args.device
        self.lambda_kl = getattr(args, 'lambda_kl', 0.5)  ### MODIFIED: Hyperparameter for KL weight

        # 初始化客户端
        self.clients = []
        for i in range(self.num_clients):
            client = FedDAKClient(args, client_id=i)
            self.clients.append(client)

        # 初始化全局 GFE：使用与客户端一致的结构
        dummy_model = create_model(args.dataset)
        self.gfe = copy.deepcopy(dummy_model.gfe).cpu()  # 初始放 CPU 节省显存

        # ### MODIFIED: Compute global class distribution at initialization
        self.global_class_dist = self.get_global_class_distribution()
        # Distribute global distribution to all clients
        for client in self.clients:
            client.set_global_distribution(self.global_class_dist)

        self.best_test_acc = 0.0
        self.Budget = []

        # === ADDED: Prepare result saving path and filename ===
        save_dir = r"D:\A2feddak\FedDAK\resault"
        os.makedirs(save_dir, exist_ok=True)
        dataset = args.dataset
        num_clients = args.num_clients
        partition = args.partition
        alpha = args.alpha
        filename = f"{dataset} {num_clients} {partition} {alpha}.txt"
        self.result_file = os.path.join(save_dir, filename)
        # =====================================================

    ### MODIFIED: New method to compute global class distribution
    def get_global_class_distribution(self):
        """Compute global class distribution from all clients"""
        class_counts = {}
        total_samples = 0

        for client in self.clients:
            # Get local class counts (not probabilities)
            for cls, count in client.local_class_counts.items():
                class_counts[cls] = class_counts.get(cls, 0) + count
                total_samples += count

        if total_samples == 0:
            return {}

        # Convert to probabilities
        return {cls: count / total_samples for cls, count in class_counts.items()}

    ### MODIFIED: New method to compute KL divergence between two distributions
    def compute_kl_divergence(self, local_dist, global_dist):
        """Compute KL divergence between local and global distributions"""
        if not local_dist or not global_dist:
            return torch.tensor(0.0, device=self.device)

        # Get all classes present in either distribution
        all_classes = sorted(set(local_dist.keys()) | set(global_dist.keys()))

        # Create probability tensors
        local_probs = torch.tensor([local_dist.get(c, 1e-8) for c in all_classes],
                                   dtype=torch.float32, device=self.device)
        global_probs = torch.tensor([global_dist.get(c, 1e-8) for c in all_classes],
                                    dtype=torch.float32, device=self.device)

        # Normalize to ensure valid probability distributions
        local_probs = local_probs / local_probs.sum()
        global_probs = global_probs / global_probs.sum()

        # Compute KL divergence: KL(local || global)
        kl_div = dist.kl_divergence(
            dist.Categorical(probs=local_probs),
            dist.Categorical(probs=global_probs)
        )
        return kl_div

    def send_models(self):
        """向所有客户端发送全局 GFE 的 state_dict（字典）"""
        gfe_state_dict = self.gfe.state_dict()
        for client in self.clients:
            client.set_model(gfe_state_dict)  # 传 state_dict，不是模型对象！

    ### MODIFIED: Updated receive_models with heterogeneity-aware weighting
    def receive_models(self):
        assert len(self.selected_clients) > 0
        active_train_samples = sum(c.train_samples for c in self.selected_clients)

        self.uploaded_weights = []
        self.uploaded_states = []

        for client in self.selected_clients:
            # Base weight: proportion of training samples
            base_weight = client.train_samples / active_train_samples

            # Get client's local distribution
            local_dist = client._compute_local_class_dist()

            # Compute KL divergence from global distribution
            kl_div = self.compute_kl_divergence(local_dist, self.global_class_dist)

            # Adjust weight: reduce weight for highly heterogeneous clients
            adjusted_weight = base_weight * (1 - self.lambda_kl * kl_div.detach().cpu().item())

            # Ensure non-negative weights
            adjusted_weight = max(adjusted_weight, 1e-6)

            # Store weight and model state
            state = copy.deepcopy(client.model.gfe.state_dict())
            self.uploaded_weights.append(adjusted_weight)
            self.uploaded_states.append(state)

        # ### MODIFIED: Renormalize weights to sum to 1
        total_weight = sum(self.uploaded_weights)
        self.uploaded_weights = [w / total_weight for w in self.uploaded_weights]

    def aggregate_parameters(self):
        assert len(self.uploaded_states) > 0

        avg_state = {}
        for key in self.uploaded_states[0].keys():
            avg_state[key] = torch.zeros_like(self.uploaded_states[0][key], dtype=torch.float32)

        for w, state in zip(self.uploaded_weights, self.uploaded_states):
            for key in avg_state.keys():
                avg_state[key] += w * state[key].float()

        # 恢复原始 dtype（如 torch.half / torch.float）
        final_state = {}
        for key, val in avg_state.items():
            final_state[key] = val.to(dtype=self.uploaded_states[0][key].dtype)

        self.gfe.load_state_dict(final_state)

    def run(self):
        self.send_models()  # 初始同步

        for round_idx in range(self.global_epochs):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if round_idx % self.eval_interval == 0:
                print(f"\n{'=' * 50}")
                print(f"Round {round_idx} Evaluation")
                print(f"{'=' * 50}")
                self.evaluate(round_idx)  # 传入 round_idx 用于保存

            # 客户端本地训练（含蒸馏）
            for client in self.selected_clients:
                client.train()

            # 聚合 GFE 参数
            self.receive_models()
            self.aggregate_parameters()
            self.send_models()

            round_time = time.time() - s_t
            self.Budget.append(round_time)
            print(f"{'-' * 20} Round {round_idx} time: {round_time:.2f}s {'-' * 20}")

        print(f"\n✅ Best Test Accuracy: {self.best_test_acc:.4f}")
        if len(self.Budget) > 1:
            avg_time = sum(self.Budget[1:]) / len(self.Budget[1:])
            print(f"⏱️  Avg Time per Round (excl. round 0): {avg_time:.2f}s")

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.randint(
                self.num_join_clients, self.num_clients + 1
            )
        else:
            self.current_num_join_clients = self.num_join_clients

        selected = np.random.choice(
            self.clients, self.current_num_join_clients, replace=False
        )
        return selected.tolist()

    def test_metrics(self):
        num_samples, corrects = [], []
        for c in self.clients:
            stats = c.test_metrics()
            num_samples.append(stats["test_num_samples"])
            corrects.append(stats["test_corrects"])
        return num_samples, corrects

    def train_metrics(self):
        num_samples, corrects = [], []
        for c in self.clients:
            stats = c.train_metrics()
            num_samples.append(stats["train_num_samples"])
            corrects.append(stats["train_corrects"])
        return num_samples, corrects

    def evaluate(self, round_idx=0):
        # 测试指标
        test_ns, test_corrects = self.test_metrics()
        test_acc = sum(test_corrects) / sum(test_ns)

        # 训练指标
        train_ns, train_corrects = self.train_metrics()
        train_acc = sum(train_corrects) / sum(train_ns)

        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc

        # 客户端 accuracy 标准差
        client_accs = [c / n for c, n in zip(test_corrects, test_ns)]
        acc_std = np.std(client_accs)

        print(f"[Test]  Acc: {test_acc:.4f} ± {acc_std:.4f}")

        # === ADDED: Save the result to file ===
        with open(self.result_file, "a") as f:
            f.write(f"Round {round_idx}: [Test] Acc: {test_acc:.4f} ± {acc_std:.4f}\n")
        # =====================================