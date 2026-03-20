"""
PSW-I: Optimal Transport based Imputation for Time Series.

This adaptation keeps the original OT-based imputation logic while exposing a
PyTorch nn.Module interface compatible with time_series.
"""

import numpy as np
import torch
import torch.nn as nn

try:
    import ot
except ImportError as e:  # pragma: no cover - runtime guard
    raise ImportError("Please install POT library: pip install POT") from e


class Model(nn.Module):
    """
    PSW-I imputation model using Sinkhorn-based optimal transport.
    Only supports the imputation task.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, "pred_len", 0)

        # PSW-I hyperparameters (default to original implementation values)
        self.lr = getattr(configs, "pswi_lr", 1e-2)
        self.n_epochs = getattr(configs, "pswi_n_epochs", 500)
        self.batch_size_ot = getattr(configs, "pswi_batch_size", 512)
        self.n_pairs = getattr(configs, "pswi_n_pairs", 1)
        self.noise = getattr(configs, "pswi_noise", 1e-2)
        self.reg_sk = getattr(configs, "pswi_reg_sk", 1.0)
        self.numItermax = getattr(configs, "pswi_numItermax", 1000)
        self.stopThr = getattr(configs, "pswi_stopThr", 1e-9)
        self.normalize = getattr(configs, "pswi_normalize", 0)

        # Dummy parameter so optimizers in the framework can initialize
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name != "imputation":
            raise NotImplementedError("PSW-I only supports the imputation task.")
        if mask is None:
            raise ValueError("Mask is required for PSW-I imputation.")
        return self.imputation(x_enc, mask)

    def imputation(self, x_enc, mask):
        """
        Args:
            x_enc: Tensor [B, T, N] with zeros at missing positions.
            mask:  Tensor [B, T, N], 1 = observed, 0 = missing.
        """
        B, T, N = x_enc.shape
        device = x_enc.device

        outputs = []
        for b in range(B):
            x_sample = x_enc[b].to(torch.double)
            mask_sample = mask[b].to(torch.double)
            imputed = self._ot_impute(x_sample, mask_sample, device, sample_idx=b, total_samples=B)
            outputs.append(imputed)
            # 每处理 10 个样本打印一次进度
            if (b + 1) % 10 == 0 or b == B - 1:
                print(f"  [PSWI] Batch progress: {b + 1}/{B} samples processed", flush=True)

        result = torch.stack(outputs, dim=0).float()
        
        # 添加一个与 dummy_param 相关的零值项，使输出保持在计算图中
        # 这样框架的 backward() 可以执行，但不会改变填补结果
        # dummy_param * 0 = 0，数值上不影响 result
        result = result + (self.dummy_param * 0.0).expand_as(result)
        
        return result

    def _ot_impute(self, X, mask, device, sample_idx=0, total_samples=1):
        """
        Core OT imputation for a single sample (2D: [T, N]).

        Mask convention in framework: 1 = observed, 0 = missing.
        """
        # 使用 enable_grad 确保在 no_grad 上下文（如验证阶段）中也能正常优化
        with torch.enable_grad():
            n, d = X.shape

            # Adjust batch size for small sequences
            batch_size = self.batch_size_ot
            if n > 1 and batch_size > n // 2:
                e = int(np.log2(max(n // 2, 1)))
                batch_size = max(2 ** e, 1)
            batch_size = max(1, min(batch_size, n))

            missing_mask = (mask == 0)
            if missing_mask.sum() == 0:
                return X  # no missing values

            # Initialize missing values around feature means with noise
            observed_sum = (X * mask).sum(dim=0)
            observed_count = mask.sum(dim=0).clamp(min=1)
            feature_means = observed_sum / observed_count  # [N]

            imps_init = self.noise * torch.randn_like(X) + feature_means.unsqueeze(0)
            imps = imps_init[missing_mask].clone().detach().requires_grad_(True)

            optimizer = torch.optim.Adam([imps], lr=self.lr)

            # # 打印间隔：对于第一个样本更频繁打印
            # print_interval = max(1, self.n_epochs // 5) if sample_idx == 0 else self.n_epochs + 1

            for epoch in range(self.n_epochs):
                X_filled = X.clone()
                X_filled[missing_mask] = imps

                loss = torch.tensor(0.0, device=device, dtype=torch.double)
                for _ in range(self.n_pairs):
                    idx1 = np.random.choice(n, batch_size, replace=False)
                    idx2 = np.random.choice(n, batch_size, replace=False)

                    X1 = X_filled[idx1]
                    X2 = X_filled[idx2]

                    M = ot.dist(X1, X2, metric="sqeuclidean", p=2)
                    if self.normalize == 1 and M.max() > 0:
                        M = M / M.max()

                    a = torch.ones(batch_size, device=device, dtype=torch.double) / batch_size
                    b = torch.ones(batch_size, device=device, dtype=torch.double) / batch_size

                    loss = loss + ot.sinkhorn2(
                        a, b, M, self.reg_sk, numItermax=self.numItermax, stopThr=self.stopThr
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"    [PSWI] Sample {sample_idx}: stopped at epoch {epoch} (loss nan/inf)", flush=True)
                    break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # # 对第一个样本打印 epoch 进度
                # if (epoch + 1) % print_interval == 0:
                #     print(f"    [PSWI] Sample {sample_idx}/{total_samples}: epoch {epoch + 1}/{self.n_epochs}, loss={loss.item():.6f}", flush=True)

            X_filled = X.clone()
            X_filled[missing_mask] = imps.detach()
            return X_filled

