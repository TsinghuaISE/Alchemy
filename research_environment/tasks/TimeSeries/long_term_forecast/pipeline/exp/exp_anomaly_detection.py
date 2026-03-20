from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        if self.args.model == 'CATCH':
            return self.train_catch(setting)
        if self.args.model == 'MtsCID':
            return self.train_mtscid(setting)
        if self.args.model == 'CrossAD':
            return self.train_crossad(setting)
        return self.train_default(setting)

    def train_default(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def train_mtscid(self, setting):
        """Training loop for MtsCID with AdamW + PolynomialDecayLR + Entropy Loss."""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        peak_lr = getattr(self.args, 'mtscid_peak_lr', 2e-3)
        end_lr = getattr(self.args, 'mtscid_end_lr', 5e-5)
        weight_decay = getattr(self.args, 'mtscid_weight_decay', 5e-5)
        warmup_epoch = getattr(self.args, 'mtscid_warmup_epoch', 0)
        alpha = getattr(self.args, 'mtscid_alpha', 1.0)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=peak_lr, weight_decay=weight_decay
        )

        # local import to avoid impacting other models
        # from models.MtsCID import PolynomialDecayLR
        from layers.MtsCID_Scheduler import PolynomialDecayLR
        scheduler = PolynomialDecayLR(
            optimizer,
            warmup_updates=warmup_epoch * self.args.batch_size,
            tot_updates=self.args.train_epochs * self.args.batch_size,
            lr=peak_lr,
            end_lr=end_lr,
            power=1.0
        )

        criterion = nn.MSELoss()
        model_instance = self.model.module if hasattr(self.model, 'module') else self.model

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            rec_losses = []
            entropy_losses = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                output, entropy_loss = model_instance.forward_with_loss(batch_x)

                rec_loss = criterion(output, batch_x)
                loss = rec_loss + alpha * entropy_loss

                train_loss.append(loss.item())
                rec_losses.append(rec_loss.item())
                entropy_losses.append(entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else float(entropy_loss))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | rec_loss: {2:.7f} | entropy_loss: {3:.7f}".format(
                        i + 1, epoch + 1, rec_loss.item(),
                        entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else float(entropy_loss)
                    ))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                optimizer.step()
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            avg_rec_loss = np.average(rec_losses)
            avg_entropy_loss = np.average(entropy_losses)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} (rec: {3:.7f}, entropy: {4:.7f}) | Vali Loss: {5:.7f} | Test Loss: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, avg_rec_loss, avg_entropy_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def train_catch(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        catch_lr = getattr(self.args, 'catch_lr', 0.0001)
        catch_mask_lr = getattr(self.args, 'catch_mask_lr', 0.00001)
        catch_pct_start = getattr(self.args, 'catch_pct_start', 0.3)
        dc_lambda = getattr(self.args, 'catch_dc_lambda', 0.005)
        auxi_lambda = getattr(self.args, 'catch_auxi_lambda', 0.005)

        model_instance = self.model.module if hasattr(self.model, 'module') else self.model

        main_params = model_instance.get_main_params()
        mask_params = model_instance.get_mask_generator_params()

        optimizer_main = optim.Adam(main_params, lr=catch_lr)
        optimizer_mask = optim.Adam(mask_params, lr=catch_mask_lr)

        scheduler_main = lr_scheduler.OneCycleLR(
            optimizer=optimizer_main,
            steps_per_epoch=train_steps,
            pct_start=catch_pct_start,
            epochs=self.args.train_epochs,
            max_lr=catch_lr,
        )
        scheduler_mask = lr_scheduler.OneCycleLR(
            optimizer=optimizer_mask,
            steps_per_epoch=train_steps,
            pct_start=catch_pct_start,
            epochs=self.args.train_epochs,
            max_lr=catch_mask_lr,
        )

        criterion = nn.MSELoss()
        mask_update_step = max(1, min(int(train_steps / 10), 100))

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            rec_losses = []
            auxi_losses = []
            dc_losses = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                optimizer_main.zero_grad()

                batch_x = batch_x.float().to(self.device)

                output, dcloss, auxi_loss = model_instance.forward_with_loss(batch_x)

                rec_loss = criterion(output, batch_x)
                loss = rec_loss + dc_lambda * dcloss + auxi_lambda * auxi_loss

                train_loss.append(loss.item())
                rec_losses.append(rec_loss.item())
                auxi_losses.append(auxi_loss.item())
                dc_losses.append(dcloss.item() if isinstance(dcloss, torch.Tensor) else float(dcloss))

                if (i + 1) % mask_update_step == 0:
                    optimizer_mask.step()
                    optimizer_mask.zero_grad()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | rec_loss: {2:.7f} | auxi_loss: {3:.7f} | dc_loss: {4:.7f}".format(
                        i + 1, epoch + 1, rec_loss.item(), auxi_loss.item(),
                        dcloss.item() if isinstance(dcloss, torch.Tensor) else float(dcloss)
                    ))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                optimizer_main.step()

                scheduler_main.step()
                scheduler_mask.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            avg_rec_loss = np.average(rec_losses)
            avg_auxi_loss = np.average(auxi_losses)
            avg_dc_loss = np.average(dc_losses)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} (rec: {3:.7f}, auxi: {4:.7f}, dc: {5:.7f}) | Vali Loss: {6:.7f} | Test Loss: {7:.7f}".format(
                epoch + 1, train_steps, train_loss, avg_rec_loss, avg_auxi_loss, avg_dc_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = '/app/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduction='none')

        def _score_batch(batch_x, outputs):
            if self.args.model == 'CATCH':
                model_instance = self.model.module if hasattr(self.model, 'module') else self.model
                temp_score = self.anomaly_criterion(batch_x, outputs).mean(dim=-1)  # [B, T]
                freq_score = model_instance.freq_criterion(outputs, batch_x)        # [B, T]
                score_lambda = getattr(self.args, 'catch_score_lambda', 0.05)
                return temp_score + score_lambda * freq_score
            elif self.args.model == 'MtsCID':
                model_instance = self.model.module if hasattr(self.model, 'module') else self.model
                return model_instance.compute_anomaly_score(batch_x)
            elif self.args.model == 'CrossAD':
                model_instance = self.model.module if hasattr(self.model, 'module') else self.model
                return model_instance.compute_anomaly_score(batch_x)
            else:
                return self.anomaly_criterion(batch_x, outputs).mean(dim=-1)

        # (1) statistic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                score = _score_batch(batch_x, outputs)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            outputs = self.model(batch_x, None, None, None)
            score = _score_batch(batch_x, outputs)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        ratios = self.args.anomaly_ratio if isinstance(self.args.anomaly_ratio, (list, tuple)) else [self.args.anomaly_ratio]

        f = open("/logs/result_anomaly_detection.txt", 'a')
        for ratio in ratios:
            threshold = np.percentile(combined_energy, 100 - ratio)
            print(f"Threshold (ratio={ratio}) : {threshold}")

            pred = (test_energy > threshold).astype(int)
            labels_flat = np.concatenate(test_labels, axis=0).reshape(-1)
            gt = labels_flat.astype(int)

            gt, pred = adjustment(gt, pred)

            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
            print("Ratio {0} -> Accuracy : {1:.4f}, Precision : {2:.4f}, Recall : {3:.4f}, F-score : {4:.4f}".format(
                ratio, accuracy, precision, recall, f_score))

            f.write(setting + f"  ratio={ratio}\n")
            f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score))
            f.write('\n')
        f.write('\n')
        f.close()
        return

    def train_crossad(self, setting):
        """Training loop for CrossAD with native multi-scale loss and lr schedule."""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_instance = self.model.module if hasattr(self.model, 'module') else self.model
        lradj = getattr(self.args, 'crossad_lradj', 'type1')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                ms_loss, q_distance = model_instance.forward_with_loss(batch_x)
                loss = ms_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali_crossad(vali_data, vali_loader, model_instance)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # CrossAD lr schedule (type1): halve lr every epoch
            if lradj == 'type1':
                lr = self.args.learning_rate * (0.5 ** epoch)
                for param_group in model_optim.param_groups:
                    param_group['lr'] = lr
                print('Updating learning rate to {}'.format(lr))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali_crossad(self, vali_data, vali_loader, model_instance):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                ms_loss, _ = model_instance.forward_with_loss(batch_x)
                total_loss.append(ms_loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
