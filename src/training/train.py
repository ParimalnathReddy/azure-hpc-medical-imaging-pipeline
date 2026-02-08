import os, json, time, argparse, random
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(name: str, num_classes: int):
    if name.lower() == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown model: {name}")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_prob.extend(probs.tolist())
        y_pred.extend(pred.tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": acc, "f1": f1, "auc": auc, "confusion_matrix": cm}

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    set_seed(int(cfg["seed"]))

    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{cfg['model']}"
    run_root = Path(cfg["paths"]["run_root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(cfg["paths"]["local_data_root"])
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    tfm_train = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=tfm_train)
    val_ds   = datasets.ImageFolder(str(val_dir), transform=tfm_eval)
    test_ds  = datasets.ImageFolder(str(test_dir), transform=tfm_eval)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(cfg["model"], cfg["num_classes"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": []}

    for epoch in range(int(cfg["epochs"])):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg['epochs']}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history["train_loss"].append(float(np.mean(losses)))

        val_metrics = evaluate(model, val_loader, device)
        print(f"[epoch {epoch+1}] loss={history['train_loss'][-1]:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}")

    test_metrics = evaluate(model, test_loader, device)

    # save
    torch.save(model.state_dict(), run_root / "best_model.pt")
    with open(run_root / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    with open(run_root / "metrics.json", "w") as f:
        json.dump({"val_last": val_metrics, "test": test_metrics, "history": history}, f, indent=2)

    print("Run saved at:", run_root)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
