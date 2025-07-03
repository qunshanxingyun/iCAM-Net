import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score,
    precision_score, recall_score, roc_auc_score
)

import wandb

class MultiTaskTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        hd_train_loader, hd_val_loader, hd_test_loader,model_name="best_val.pth",
        cp_train_loader=None, cp_val_loader=None, cp_test_loader=None,
        num_epochs=10,
        alpha_cp=1.0,
        result_root="result",
        use_wandb=False
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.hd_train_loader = hd_train_loader
        self.hd_val_loader = hd_val_loader
        self.hd_test_loader = hd_test_loader

        self.cp_train_loader = cp_train_loader
        self.cp_val_loader = cp_val_loader
        self.cp_test_loader = cp_test_loader

        self.num_epochs = num_epochs
        self.alpha_cp = alpha_cp

        self.use_wandb = use_wandb
        self.model_name = model_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(result_root, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_model_path = os.path.join(self.save_dir, model_name + '.pth')

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            train_hd_loss, train_cp_loss = self._train_one_epoch()

            val_loss_hd, val_metrics_hd = self._evaluate_hd(self.hd_val_loader)
            if self.cp_val_loader is not None:
                val_loss_cp, val_metrics_cp = self._evaluate_cp(self.cp_val_loader)
            else:
                val_loss_cp, val_metrics_cp = (0.0, {})

            print(f"[Epoch {epoch}] "
                  f"HD_TrainLoss={train_hd_loss:.4f}, CP_TrainLoss={train_cp_loss:.4f}, "
                  f"HD_ValLoss={val_loss_hd:.4f}, HD_ValAcc={val_metrics_hd.get('acc',0):.4f}, "
                  f"HD_ValF1={val_metrics_hd.get('f1',0):.4f}")

            # ------ wandb.log ----------
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "HD_TrainLoss": train_hd_loss,
                    "CP_TrainLoss": train_cp_loss,
                    "HD_ValLoss": val_loss_hd,
                    "HD_ValAcc": val_metrics_hd.get("acc", 0),
                    "HD_ValF1": val_metrics_hd.get("f1", 0),
                    "HD_ValPrecision": val_metrics_hd.get("precision", 0),
                    "HD_ValRecall": val_metrics_hd.get("recall", 0),
                    "HD_ValAUPRC": val_metrics_hd.get("auprc", 0),
                    "HD_ValAUROC": val_metrics_hd.get("auc_roc", 0),
                })
                if self.cp_val_loader is not None:
                    wandb.log({
                        "CP_ValLoss": val_loss_cp,
                        "CP_ValAcc": val_metrics_cp.get("acc", 0),
                        "CP_ValF1": val_metrics_cp.get("f1", 0),
                        "CP_ValPrecision": val_metrics_cp.get("precision", 0),
                        "CP_ValRecall": val_metrics_cp.get("recall", 0),
                        "CP_ValAUPRC": val_metrics_cp.get("auprc", 0),
                        "CP_ValAUROC": val_metrics_cp.get("auc_roc", 0),
                    })

            if val_loss_hd < self.best_val_loss:
                self.best_val_loss = val_loss_hd
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  [Info] => Best val_loss (HD) updated ({val_loss_hd:.4f}). Model saved.")

        print("\n[Info] Training finished. Loading Best Model for final testing & demos...")
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        self._final_test_and_inference_demo()
        print("[Info] Done. Training & evaluation pipeline complete.")

    def _train_one_epoch(self):
        self.model.train()
        criterion = LabelSmoothingBCELoss(smoothing=0.1)

        total_hd_loss = 0.0
        for herb_ids, disease_ids, labels in tqdm(self.hd_train_loader, desc="Train HD"):
            herb_ids = herb_ids.squeeze(-1).to(self.device)
            disease_ids = disease_ids.squeeze(-1).to(self.device)
            labels = labels.squeeze(-1).to(self.device)

            self.optimizer.zero_grad()
            scores, _attn_scores = self.model(herb_ids, disease_ids)
            loss_hd = criterion(scores, labels)
            loss_hd.backward()
            self.optimizer.step()
            total_hd_loss += loss_hd.item() * len(labels)
        avg_hd_loss = total_hd_loss / len(self.hd_train_loader.dataset)
 
        avg_cp_loss = 0.0
        if self.cp_train_loader is not None:
            total_cp_loss = 0.0
            for comp_ids, prot_ids, lbl_cp in tqdm(self.cp_train_loader, desc="Train CP"):
                comp_ids = comp_ids.squeeze(-1).to(self.device)
                prot_ids = prot_ids.squeeze(-1).to(self.device)
                lbl_cp   = lbl_cp.squeeze(-1).to(self.device)

                self.optimizer.zero_grad()
                scores_cp = self.model.forward_cp_score(comp_ids, prot_ids)
                loss_cp = criterion(scores_cp, lbl_cp)
                (self.alpha_cp * loss_cp).backward()
                self.optimizer.step()
                total_cp_loss += loss_cp.item() * len(lbl_cp)
            avg_cp_loss = total_cp_loss / len(self.cp_train_loader.dataset)

        return avg_hd_loss, avg_cp_loss

    @torch.no_grad()
    def _evaluate_hd(self, data_loader):
        self.model.eval()
        criterion = torch.nn.BCELoss()
        total_loss = 0.0
        all_labels, all_preds = [], []

        for herb_ids, disease_ids, labels in data_loader:
            herb_ids = herb_ids.squeeze(-1).to(self.device)
            disease_ids = disease_ids.squeeze(-1).to(self.device)
            labels = labels.squeeze(-1).to(self.device)

            scores, _attn_scores = self.model(herb_ids, disease_ids)
            loss = criterion(scores, labels)
            total_loss += loss.item() * len(labels)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(scores.cpu().numpy().tolist())

        avg_loss = total_loss / len(data_loader.dataset)
        metrics = self._calculate_metrics(all_labels, all_preds)
        return avg_loss, metrics

    @torch.no_grad()
    def _evaluate_cp(self, data_loader):
        self.model.eval()
        criterion = torch.nn.BCELoss()
        total_loss = 0.0
        all_labels, all_preds = [], []

        for comp_ids, prot_ids, lbl_cp in data_loader:
            comp_ids = comp_ids.squeeze(-1).to(self.device)
            prot_ids = prot_ids.squeeze(-1).to(self.device)
            lbl_cp = lbl_cp.squeeze(-1).to(self.device)

            scores_cp = self.model.forward_cp_score(comp_ids, prot_ids)
            loss_cp = criterion(scores_cp, lbl_cp)
            total_loss += loss_cp.item() * len(lbl_cp)

            all_labels.extend(lbl_cp.cpu().numpy().tolist())
            all_preds.extend(scores_cp.cpu().numpy().tolist())

        avg_loss = total_loss / len(data_loader.dataset)
        metrics = self._calculate_metrics(all_labels, all_preds)
        return avg_loss, metrics

    def _calculate_metrics(self, y_true, y_prob):
        y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        auprc = average_precision_score(y_true, y_prob)
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_roc = 0.0
        return {
            "acc": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auprc": auprc,
            "auc_roc": auc_roc,
        }

    @torch.no_grad()
    def _final_test_and_inference_demo(self):
        print("[Step] Final Testing on HD & CP with best model...")
        hd_test_loss, hd_test_metrics = self._test_hd(self.hd_test_loader)
        if self.cp_test_loader:
            cp_test_loss, cp_test_metrics = self._test_cp(self.cp_test_loader)
        else:
            cp_test_loss, cp_test_metrics = (0.0, {})

        print("\n=============== Final Test Results ===============")
        print("[HD Test] Loss={:.4f}".format(hd_test_loss))
        for k, v in hd_test_metrics.items():
            print("         {}: {:.4f}".format(k, v))

        if self.cp_test_loader:
            print("[CP Test] Loss={:.4f}".format(cp_test_loss))
            for k, v in cp_test_metrics.items():
                print("         {}: {:.4f}".format(k, v))

        if self.use_wandb:
            wandb.log({
                "HD_TestLoss": hd_test_loss,
                "HD_TestAcc": hd_test_metrics.get("acc", 0),
                "HD_TestF1": hd_test_metrics.get("f1", 0),
                "HD_TestPrecision": hd_test_metrics.get("precision", 0),
                "HD_TestRecall": hd_test_metrics.get("recall", 0),
                "HD_TestAUPRC": hd_test_metrics.get("auprc", 0),
                "HD_TestAUROC": hd_test_metrics.get("auc_roc", 0),
            })
            if self.cp_test_loader:
                wandb.log({
                    "CP_TestLoss": cp_test_loss,
                    "CP_TestAcc": cp_test_metrics.get("acc", 0),
                    "CP_TestF1": cp_test_metrics.get("f1", 0),
                    "CP_TestPrecision": cp_test_metrics.get("precision", 0),
                    "CP_TestRecall": cp_test_metrics.get("recall", 0),
                    "CP_TestAUPRC": cp_test_metrics.get("auprc", 0),
                    "CP_TestAUROC": cp_test_metrics.get("auc_roc", 0),
                })

    @torch.no_grad()
    def _test_hd(self, data_loader):
        self.model.eval()
        criterion = torch.nn.BCELoss()
        total_loss = 0.0
        all_labels, all_preds = [], []

        for herb_ids, disease_ids, labels in data_loader:
            herb_ids = herb_ids.squeeze(-1).to(self.device)
            disease_ids = disease_ids.squeeze(-1).to(self.device)
            labels = labels.squeeze(-1).to(self.device)

            scores, _ = self.model(herb_ids, disease_ids)
            loss = criterion(scores, labels)
            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(scores.cpu().numpy().tolist())

        avg_loss = total_loss / len(data_loader.dataset)
        metrics = self._calculate_metrics(all_labels, all_preds)
        return avg_loss, metrics

    @torch.no_grad()
    def _test_cp(self, data_loader):
        self.model.eval()
        criterion = torch.nn.BCELoss()
        total_loss = 0.0
        all_labels, all_preds = [], []

        for comp_ids, prot_ids, lbl_cp in data_loader:
            comp_ids = comp_ids.squeeze(-1).to(self.device)
            prot_ids = prot_ids.squeeze(-1).to(self.device)
            lbl_cp   = lbl_cp.squeeze(-1).to(self.device)

            scores_cp = self.model.forward_cp_score(comp_ids, prot_ids)
            loss_cp = criterion(scores_cp, lbl_cp)
            total_loss += loss_cp.item() * len(lbl_cp)
            all_labels.extend(lbl_cp.cpu().numpy().tolist())
            all_preds.extend(scores_cp.cpu().numpy().tolist())

        avg_loss = total_loss / len(data_loader.dataset)
        metrics = self._calculate_metrics(all_labels, all_preds)
        return avg_loss, metrics

    
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, preds, targets):
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing * 0.5
        epsilon = 1e-12
        loss = -(smooth_targets * torch.log(preds + epsilon) + 
                 (1 - smooth_targets) * torch.log(1 - preds + epsilon))
        
        return loss.mean()