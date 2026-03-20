from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings('ignore')


def _compute_beta(epoch, max_epoch, schedule='constant'):
    """Beta scheduler for InterpGN shapelet auxiliary loss."""
    if schedule == 'cosine':
        return 0.5 * (1 + np.cos(np.pi * epoch / max_epoch))
    elif schedule == 'linear':
        return 1 - epoch / max_epoch
    else:
        return 1.0


# ============== SGN 辅助函数 ==============
def _sgn_compute_distance_matrix(X):
    """计算欧氏距离矩阵"""
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(X, metric='euclidean'))


def _sgn_compute_double_centering(D):
    """双中心化处理"""
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ D @ H


def _sgn_compute_brownian_distance_covariance(X, Y):
    """计算Brownian距离协方差"""
    D_X = _sgn_compute_distance_matrix(X)
    D_Y = _sgn_compute_distance_matrix(Y)
    A = _sgn_compute_double_centering(D_X)
    B = _sgn_compute_double_centering(D_Y)
    return np.mean(A * B)


def _sgn_compute_similarity_matrix(data):
    """计算通道间的相似度矩阵
    
    Args:
        data: shape (B, C, T) - B个样本，C个通道，T个时间步
    Returns:
        S: shape (C, C) - 通道间相似度矩阵
    """
    from sklearn.preprocessing import StandardScaler
    B, C, T = data.shape
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data.reshape(B * C, T)).reshape(B, C, T)

    S = np.zeros((C, C))
    for i in range(C):
        for j in range(i, C):
            X = data_normalized[:, i, :]
            Y = data_normalized[:, j, :]
            bdc_xy = _sgn_compute_brownian_distance_covariance(X, Y)
            bdc_xx = _sgn_compute_brownian_distance_covariance(X, X)
            bdc_yy = _sgn_compute_brownian_distance_covariance(Y, Y)

            if bdc_xx > 0 and bdc_yy > 0:
                bdc = bdc_xy / np.sqrt(bdc_xx * bdc_yy)
            else:
                bdc = 0

            S[i, j] = bdc
            S[j, i] = bdc

    np.fill_diagonal(S, 1.0)
    return S


def _sgn_group_based_on_num_groups(similarity_matrix, num_groups, noise_std=0.1):
    """使用K-means从相似度矩阵生成分组矩阵
    
    Args:
        similarity_matrix: shape (C, C) - 通道间相似度矩阵
        num_groups: int - 分组数量
        noise_std: float - 噪声标准差
    Returns:
        groups_matrix: shape (C, num_groups) - 分组矩阵
    """
    from sklearn.cluster import KMeans
    C = similarity_matrix.shape[0]
    
    # 自动调整：当通道数小于分组数时，将分组数设为通道数
    if C < num_groups:
        print(f"SGN: Warning - enc_in ({C}) < num_groups ({num_groups}), adjusting num_groups to {C}")
        num_groups = C
    
    kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(similarity_matrix)
    labels = kmeans.labels_

    groups_matrix = np.zeros((C, num_groups), dtype=float)
    for i, label in enumerate(labels):
        groups_matrix[i, label] = 1

    # 添加噪声
    noise = np.random.normal(0, noise_std, size=groups_matrix.shape)
    groups_matrix += noise
    groups_matrix = np.clip(groups_matrix, 0, 1)

    return groups_matrix, num_groups  # 返回调整后的 num_groups


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)

        # SGN特有：计算分组矩阵
        if self.args.model == 'SGN':
            print("SGN: Computing groups matrix from training data...")
            # 获取训练数据用于计算相似度矩阵
            num_samples = len(train_data)
            sample_data = []
            # 最多使用2000个样本计算相似度矩阵
            for i in range(min(num_samples, 2000)):
                x, _ = train_data[i]  # UEAloader returns (x, label), not (x, x_mark, label)
                sample_data.append(x.numpy())
            sample_data = np.stack(sample_data, axis=0)  # (N, T, C)
            sample_data = sample_data.transpose(0, 2, 1)  # (N, C, T)
            
            # 计算相似度矩阵
            similarity_matrix = _sgn_compute_similarity_matrix(sample_data)
            # 生成分组矩阵
            num_groups = getattr(self.args, 'num_groups', 4)
            groups_matrix, num_groups = _sgn_group_based_on_num_groups(similarity_matrix, num_groups)
            # 更新 args 中的分组数（可能被自动调整过）
            self.args.num_groups = num_groups
            # 存储到args中供模型使用
            self.args.groups_matrix = groups_matrix
            print(f"SGN: Groups matrix computed with shape {groups_matrix.shape}, num_groups={num_groups}")

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # Keep original optimizer for other models
        if self.args.model == 'InterpGN':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.model == 'SGN':
            # SGN使用Adam优化器（与原版一致）
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        else:
            model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # InterpGN and SGN return (outputs, extra_info), others return outputs
                if self.args.model == 'InterpGN':
                    outputs, _ = self.model(batch_x, padding_mask, None, None)
                elif self.args.model == 'SGN':
                    outputs, _ = self.model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # InterpGN original scheduler
        interpgn_scheduler = None
        if self.args.model == 'InterpGN':
            interpgn_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                model_optim, T_0=self.args.train_epochs
            )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.args.model == 'InterpGN':
                    outputs, model_info = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long().squeeze(-1))
                    # Shapelet regularization
                    loss = loss + model_info.loss.mean()
                    # Shapelet prediction auxiliary loss with beta schedule
                    beta = _compute_beta(epoch, self.args.train_epochs, getattr(self.args, 'beta_schedule', 'constant'))
                    loss = loss + beta * criterion(model_info.shapelet_preds, label.long().squeeze(-1))
                elif self.args.model == 'SGN':
                    outputs, embed_loss = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long().squeeze(-1))
                    # 添加SGN的embedding loss（与原版一致，权重为0.1）
                    sgn_loss_weight = getattr(self.args, 'sgn_embed_loss_weight', 0.1)
                    if embed_loss is not None:
                        loss = loss + sgn_loss_weight * embed_loss
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

                # InterpGN weight clamp on classifier (always for InterpGN)
                if self.args.model == 'InterpGN':
                    self.model.step()

            # InterpGN scheduler step
            if interpgn_scheduler is not None:
                interpgn_scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = '/app/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.args.model == 'InterpGN':
                    outputs, _ = self.model(batch_x, padding_mask, None, None, gating_value=getattr(self.args, 'gating_value', None))
                elif self.args.model == 'SGN':
                    outputs, _ = self.model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = '/app/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
