import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, out_path: Path, labels=("NORMAL", "PNEUMONIA")):
    cm = np.array(cm)
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0,1], labels)
    plt.yticks([0,1], labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_loss_curve(losses, out_path: Path):
    fig = plt.figure()
    plt.plot(range(1, len(losses)+1), losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main(run_dir: str):
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # loss curve
    losses = metrics.get("history", {}).get("train_loss", [])
    if losses:
        plot_loss_curve(losses, plots_dir / "loss.png")

    # confusion matrix
    cm = metrics.get("test", {}).get("confusion_matrix", None)
    if cm is not None:
        plot_confusion_matrix(cm, plots_dir / "confusion_matrix.png")

    # compact report
    report = {
        "run_id": run_dir.name,
        "val_last": metrics.get("val_last", {}),
        "test": metrics.get("test", {}),
        "notes": {
            "classes": {"0": "NORMAL", "1": "PNEUMONIA"},
            "dataset": "Chest X-ray Pneumonia (Kaggle)",
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("Wrote:", run_dir / "report.json")
    print("Plots:", plots_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    main(args.run_dir)
