import sys
import os
import glob
from sklearn.model_selection import train_test_split
from torchvision import models
from torch.utils.data import DataLoader
import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

import utils
import dataset

def build_model(num_classes=15):
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT') 
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    if getattr(model, 'aux_classifier', None) is not None:
        in_c_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_c_aux, num_classes, kernel_size=1)
    return model

## as DML not support lerp_, here is an AdamW optimizer without lerp_
class AdamW_DML(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Decoupled weight decay
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom
                p.add_(update, alpha=-lr)

        return loss

def train_one_epoch(model, loader, device, optimizer, num_classes=15, ignore_index=255):
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    running_loss = 0.0
    running_correct = 0
    running_labeled = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)['out'] 
        loss = ce(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        mask = labels != ignore_index
        running_correct += (preds[mask] == labels[mask]).sum().item()
        running_labeled += mask.sum().item()

    avg_loss = running_loss / max(1, len(loader.dataset))
    pixel_acc = running_correct / max(1, running_labeled)
    return {'loss': avg_loss, 'acc': pixel_acc}
    
@torch.no_grad()
def evaluate(model, loader, device, num_classes=15, ignore_index=255):
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    total_loss = 0.0
    total_correct = 0
    total_labeled = 0
    total_hist = torch.zeros((num_classes, num_classes), device=device)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)['out'] 
        loss = ce(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1) 
        mask = labels != ignore_index
        total_correct += (preds[mask] == labels[mask]).sum().item()
        total_labeled += mask.sum().item()
        hist = torch.bincount(
            (labels[mask] * num_classes + preds[mask]).view(-1), 
            minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes).to(torch.float32)
        total_hist += hist

    avg_loss = total_loss / max(1, len(loader.dataset))
    pixel_acc = total_correct / max(1, total_labeled)
    diag = torch.diag(total_hist)
    denom = total_hist.sum(1) + total_hist.sum(0) - diag
    iou = diag / torch.clamp(denom, min=1e-6)
    miou = torch.nanmean(iou).item()
    
    return {'loss': avg_loss, 'acc': pixel_acc, 'miou': miou, 'iou': iou}

def main():
    data_path = 'D:\\Deeplabv3-Segmentation\\61541v003\\data\\WildScenes\\WildScenes2d'
    lr=3e-4
    weight_decay=1e-4
    epochs = 80
    patience = 12
    
    image_list = []
    for d in os.listdir(data_path):
        d_path = os.path.join(data_path, d)
        i_path = os.path.join(d_path, "image")
        image_list += glob.glob(f"{i_path}\\*.png")
    image_list = random.sample(image_list, 2000) ## as device limitation, we choose part of the dataset to train ##
    
    train_image_list, test_image_list = train_test_split(image_list, test_size=0.2, random_state=42)
    train_train_list, train_valid_list = train_test_split(train_image_list, test_size=0.2, random_state=42)

    train_train_dataset = dataset.WildscenesDataset(train_train_list, 1)
    train_valid_dataset = dataset.WildscenesDataset(train_valid_list, 0)
    test_dataset = dataset.WildscenesDataset(test_image_list, 0)

    train_loader = DataLoader(
        train_train_dataset, batch_size=8, shuffle=True, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        train_valid_dataset, batch_size=8, shuffle=False, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, pin_memory=True, drop_last=False
    )

    dml_device = torch_directml.device()
    model = build_model().to(dml_device)
    optimizer = AdamW_DML(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_miou = -1.0
    epochs_since_improve = 0

    history = []

    for epoch in range(1, epochs+1):
        train_stats = train_one_epoch(model, train_loader, dml_device, optimizer)
        val_stats = evaluate(model, val_loader, dml_device)
        print(f"[Epoch {epoch:03d}] "
              f"Train Loss: {train_stats['loss']:.4f} | Train Acc: {train_stats['acc']:.4f} || "
              f"Val Loss: {val_stats['loss']:.4f} | Val Acc: {val_stats['acc']:.4f} | Val mIoU:{val_stats['miou']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_stats['loss'],
            "train_acc": train_stats['acc'],
            "val_loss": val_stats['loss'],
            "val_acc": val_stats['acc'],
            "val_miou": val_stats['miou']
        })

        if val_stats['miou'] > best_miou + 1e-6:
            best_miou = val_stats['miou']
            epochs_since_improve = 0
            torch.save({'model': model.state_dict(), 'num_classes': 15}, "best_model.pth")
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    ## save history as .csv
    df = pd.DataFrame(history)
    df.to_csv("training_log.csv", index=False)

    ## test
    ckpt = torch.load("best_model.pth", map_location=dml_device)
    model.load_state_dict(ckpt['model'])
    test_stats = evaluate(model, test_loader, dml_device)
    print(f"[Test] Loss: {test_stats['loss']:.4f} | Acc: {test_stats['acc']:.4f} | mIoU: {test_stats['miou']:.4f}")
    
    ## plot test iou
    iou = test_stats['iou']
    iou = iou.detach().float().cpu().numpy()
    x = np.arange(len(iou))
    heights = np.nan_to_num(iou, nan=0.0)
    plt.figure(figsize=(10, 4))
    bars = plt.bar(x, heights)
    plt.xticks(x, [str(i) for i in x], fontsize=8)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Class ID')
    plt.ylabel('IoU')
    plt.title('Pre-class IoU')
    for i, b in enumerate(bars):
        val = iou[i]
        label = f'{val:.2f}' if not np.isnan(val) else 'NaN'
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, label,
                 ha='center', va='bottom', fontsize=8)  
    plt.tight_layout()
    plt.savefig('iou_bar.png', dpi=150)
    

## train.py main
if __name__ == "__main__":
    main()