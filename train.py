"""
train.py - reproduce the whole project with four commands:

    python train.py prepare      download Defactify and build the compact dataset in data/
    python train.py train        transfer learning + augmentation  -> models/detector.pt
    python train.py evaluate     test metrics, robustness, cost    -> results/detector/
    python train.py compare      table + chart of every run in results/

Extra runs for the report:
    python train.py train --no-aug --name no_aug
    python train.py train --model mobilenet_v3_small --name mobilenet_v3_small
    python train.py train --model efficientnet_b0 --name efficientnet_b0
"""
import argparse
import io
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from detector import (BACKBONES, IMG_SIZE, SOURCE_KEYS, SOURCES, DeepfakeDetector,
                      eval_transform, load_detector, train_transform)

ROOT = Path(__file__).resolve().parent
DATA, MODELS, RESULTS, EXAMPLES = ROOT / "data", ROOT / "models", ROOT / "results", ROOT / "examples"
SEED = 490
SOURCE_LOSS_WEIGHT = 0.3  # weight of the auxiliary "which generator" loss

# Fixed degradations used to measure robustness on the test set.
ROBUSTNESS_TESTS = {
    "clean": {},
    "JPEG 70": {"jpeg": 70},
    "JPEG 40": {"jpeg": 40},
    "blur": {"blur": 1.0},
    "half size": {"scale": 0.5},
    "feed (half size + JPEG 50)": {"scale": 0.5, "jpeg": 50},
}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================= 1. dataset
HF_REPO = "Rajarshi-Roy-research/Defactify_Image_Dataset"
HF_SHARDS = {"train": 7, "validation": 2}
# Images kept per split as (real, AI-generated); AI images are split evenly over the 5 generators.
SPLIT_SIZES = {"train": (7000, 7000), "val": (500, 500), "test": (1000, 1000)}
N_EXAMPLE_PAIRS = 6


def _save_resized(job):
    image_bytes, path, short_side = job
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scale = short_side / min(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    img.save(path, "JPEG", quality=92)


def download_shards(cache):
    """Fetch the Defactify train / validation parquet files, skipping any already on disk."""
    from huggingface_hub import hf_hub_download
    shards = {}
    for split, n in HF_SHARDS.items():
        names = [f"data/{split}-{i:05d}-of-{n:05d}.parquet" for i in range(n)]
        shards[split] = [str(cache / name) if (cache / name).exists() else
                         hf_hub_download(HF_REPO, name, repo_type="dataset", local_dir=cache) for name in names]
    return shards


def prepare(args):
    """Download the public Defactify dataset and write a small balanced copy to data/."""
    import pyarrow.parquet as pq

    rng = random.Random(SEED)
    shards = download_shards(DATA / "_download")

    def read_index(files):
        """(file, row, source id, caption) for every row; source id 0 = real, 1..5 = generator."""
        rows = []
        for f in files:
            table = pq.read_table(f, columns=["Label_B", "Caption"]).to_pydict()
            rows += [(f, i, b, c) for i, (b, c) in enumerate(zip(table["Label_B"], table["Caption"]))]
        return rows

    def sample(rows, n_real, n_fake, taken=()):
        """Pick n_real real rows and n_fake AI rows (balanced over generators)."""
        by_source = defaultdict(list)
        for row in rows:
            if row[:2] not in taken:
                by_source[row[2]].append(row)
        picked = []
        for source, group in sorted(by_source.items()):
            rng.shuffle(group)
            picked += group[:n_real if source == 0 else n_fake // len(SOURCES)]
        return picked

    train_rows, heldout_rows = read_index(shards["train"]), read_index(shards["validation"])
    selection = {"train": sample(train_rows, *SPLIT_SIZES["train"])}
    selection["val"] = sample(heldout_rows, *SPLIT_SIZES["val"])
    val_keys = {row[:2] for row in selection["val"]}
    selection["test"] = sample(heldout_rows, *SPLIT_SIZES["test"], taken=val_keys)

    # Demo pairs: the same caption as a real photo and as an AI image, never seen in train / val.
    by_caption = defaultdict(dict)
    for row in heldout_rows:
        if row[:2] not in val_keys:
            by_caption[row[3]][row[2]] = row
    pairs = [(c, v) for c, v in sorted(by_caption.items()) if 0 in v and len(v) > 1]
    rng.shuffle(pairs)
    example_rows, used = {}, defaultdict(int)
    for k, (caption, versions) in enumerate(pairs[:N_EXAMPLE_PAIRS], start=1):
        source = min((s for s in versions if s != 0), key=lambda s: (used[s], s))  # cover every generator
        used[source] += 1
        fake = versions[source]
        example_rows[versions[0][:2]] = EXAMPLES / f"pair{k}_real.jpg"
        example_rows[fake[:2]] = EXAMPLES / f"pair{k}_{SOURCE_KEYS[fake[2] - 1]}.jpg"

    # Map every selected (file, row) to its output path, then extract in one pass per shard.
    targets = defaultdict(list)
    counts = {}
    for split, rows in selection.items():
        counter = defaultdict(int)
        for f, i, source, _ in rows:
            key = "real" if source == 0 else SOURCE_KEYS[source - 1]
            counter[key] += 1
            folder = DATA / split / ("real" if source == 0 else "fake")
            targets[(f, i)].append((folder / f"{key}_{counter[key]:05d}.jpg", 256))
        counts[split] = dict(counter)
    for key, path in example_rows.items():
        targets[key].append((path, 512))
    for paths in targets.values():
        for path, _ in paths:
            path.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    with ThreadPoolExecutor() as pool:
        for f in shards["train"] + shards["validation"]:
            row = 0
            for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["Image"]):
                jobs = []
                for image in batch.column("Image").to_pylist():
                    jobs += [(image["bytes"], p, s) for p, s in targets.get((f, row), [])]
                    row += 1
                list(pool.map(_save_resized, jobs))
                done += len(jobs)
            print(f"  {Path(f).name}: {done} images written", flush=True)

    info = {"source": f"https://huggingface.co/datasets/{HF_REPO}",
            "paper": "A Comprehensive Dataset for Human vs. AI Generated Image Detection (Defactify)",
            "real_images": "MS COCO photographs", "generators": SOURCES, "seed": SEED,
            "preprocessing": "short side resized to 256 px, JPEG quality 92 (identical for both classes)",
            "images_per_split": counts}
    (DATA / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(counts, indent=2))
    print(f"Dataset ready in {DATA}. The raw download in {DATA / '_download'} can be deleted.")


class RealFakeFolder(Dataset):
    """data/<split>/real/*.jpg and data/<split>/fake/<generator>_*.jpg"""

    def __init__(self, split, transform):
        self.transform = transform
        self.items = [(p, 0, -1) for p in sorted((DATA / split / "real").glob("*.jpg"))]
        for p in sorted((DATA / split / "fake").glob("*.jpg")):
            self.items.append((p, 1, SOURCE_KEYS.index(p.name.split("_")[0])))
        if not self.items:
            raise SystemExit(f"No images in {DATA / split}. Run:  python train.py prepare")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, source = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label, source


def numpy_collate(batch):
    """Collate to NumPy arrays: batches then travel from the workers by value. (Torch's shared-memory
    tensors make every worker take 5 s to shut down on macOS.)"""
    return tuple(np.stack([np.asarray(item) for item in column]) for column in zip(*batch))


def make_loader(split, transform, args, shuffle=False, batch_size=None):
    return DataLoader(RealFakeFolder(split, transform), batch_size=batch_size or args.batch_size, shuffle=shuffle,
                      drop_last=shuffle, num_workers=args.workers, persistent_workers=args.workers > 0,
                      collate_fn=numpy_collate)


# ============================================================================= 2. training
def run_epoch(model, loader, device, optimizer=None, scheduler=None, scaler=None):
    """One pass over a loader; trains if an optimizer is given. Returns (mean loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    if training and not next(model.features.parameters()).requires_grad:
        model.features.eval()  # frozen backbone: keep its BatchNorm statistics untouched
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels, sources in loader:
        images, labels, sources = [torch.from_numpy(a).to(device) for a in (images, labels, sources)]
        is_fake = labels == 1
        with torch.set_grad_enabled(training), torch.autocast("cuda", enabled=device.type == "cuda"):
            fake_logit, source_logits = model(images)
            loss = F.binary_cross_entropy_with_logits(fake_logit.float(), labels.float())
            if is_fake.any():
                loss = loss + SOURCE_LOSS_WEIGHT * F.cross_entropy(source_logits[is_fake].float(), sources[is_fake])
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
        total_loss += loss.item() * len(labels)
        correct += ((fake_logit > 0) == is_fake).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


def train(args):
    set_seed()
    device = pick_device()
    name = args.name
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    augment = not args.no_aug
    print(f"Run '{name}': {args.model}, augmentation={'on' if augment else 'off'}, device={device}")

    train_loader = make_loader("train", train_transform(augment), args, shuffle=True)
    val_loader = make_loader("val", eval_transform(), args)
    model = DeepfakeDetector(args.model, pretrained=True).to(device)

    # Transfer learning in two phases: (1) heads only on frozen ImageNet features, (2) fine-tune everything.
    phases = [("heads only", args.head_epochs, 1e-3, True), ("fine-tune", args.epochs, args.lr, False)]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")  # mixed precision on Colab GPUs
    history, best_acc, start = [], 0.0, time.time()
    for phase, epochs, lr, frozen in phases:
        model.freeze_backbone(frozen)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(train_loader),
                                                        pct_start=0.15)
        for _ in range(epochs):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, scheduler, scaler)
            val_loss, val_acc = run_epoch(model, val_loader, device)
            history.append({"epoch": len(history) + 1, "phase": phase, "train_loss": train_loss,
                            "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
                            "seconds": time.time() - t0})
            h = history[-1]
            print(f"  epoch {h['epoch']:2d} [{phase:10s}] train loss {h['train_loss']:.3f} acc {h['train_acc']:.3f} | "
                  f"val loss {h['val_loss']:.3f} acc {h['val_acc']:.3f} | {h['seconds']:.0f}s", flush=True)
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save({"state_dict": model.state_dict(), "backbone": args.model, "augment": augment,
                            "epoch": h["epoch"], "val_acc": best_acc}, MODELS / f"{name}.pt")
    minutes = (time.time() - start) / 60
    (out_dir / "history.json").write_text(json.dumps({"train_minutes": minutes, "device": str(device),
                                                       "epochs": history}, indent=2))
    print(f"Best validation accuracy {best_acc:.4f} in {minutes:.1f} min -> {MODELS / f'{name}.pt'}")
    del train_loader, val_loader
    calibrate(name, args, device)
    evaluate(args)


# ============================================================================= 3. evaluation
class RobustnessViews:
    """Applies each degradation to the same image and stacks the results: [degradations, 3, H, W]."""

    def __init__(self, degradations):
        self.views = [eval_transform(**degradation) for degradation in degradations]

    def __call__(self, img):
        return torch.stack([view(img) for view in self.views])


@torch.no_grad()
def predict(model, split, degradations, args, device):
    """Returns labels [N], generator ids [N], P(AI) under each degradation [N, degradations],
    and the predicted generator on the first view [N]."""
    loader = make_loader(split, RobustnessViews(degradations), args, batch_size=max(8, args.batch_size // 4))
    out = defaultdict(list)
    for views, labels, sources in loader:
        fake_logit, source_logits = model(torch.from_numpy(views).flatten(0, 1).to(device))
        out["p_fake"].append(torch.sigmoid(fake_logit).view(len(labels), -1).cpu().numpy())
        out["source_pred"].append(source_logits.view(len(labels), views.shape[1], -1)[:, 0].argmax(1).cpu().numpy())
        out["label"].append(labels)
        out["source"].append(sources)
    return {k: np.concatenate(v) for k, v in out.items()}


def calibrate(name, args, device):
    """Find the decision threshold that is best on the validation set, then shift the output bias of the
    saved model so that this threshold becomes 0.5 (the rest of the code can keep using 0.5)."""
    path = MODELS / f"{name}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    out = predict(load_detector(path, device), "val", [{}], args, device)
    p, y = out["p_fake"][:, 0], out["label"]
    grid = np.linspace(0.05, 0.95, 91)
    accuracy = np.array([((p >= t) == y).mean() for t in grid])
    best = grid[accuracy == accuracy.max()]
    threshold = float(best[np.abs(best - 0.5).argmin()])  # among equally good thresholds, the closest to 0.5
    checkpoint["state_dict"]["fake_head.bias"] -= float(np.log(threshold / (1 - threshold)))
    checkpoint["calibration_threshold"] = threshold
    torch.save(checkpoint, path)
    print(f"Calibration on the validation set: threshold {threshold:.2f} moved to 0.50 "
          f"(validation accuracy {accuracy.max():.4f})")


def measure_cost(model):
    """Parameters, weight size, GFLOPs (multiply-accumulates per image) and single-image CPU latency."""
    model = model.cpu().eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    params = sum(p.numel() for p in model.parameters())
    macs = [0]

    def count(layer, _, out):
        if isinstance(layer, torch.nn.Conv2d):
            macs[0] += out.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        else:
            macs[0] += layer.in_features * layer.out_features

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model(x)
        for hook in hooks:
            hook.remove()
        for _ in range(10):
            model(x)
        t0 = time.perf_counter()
        for _ in range(50):
            model(x)
    return {"params_millions": params / 1e6, "size_mb": params * 4 / 1e6, "gflops": macs[0] / 1e9,
            "cpu_ms_per_image": (time.perf_counter() - t0) / 50 * 1000}


def evaluate(args):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name, device = args.name, pick_device()
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=True)
    model = load_detector(MODELS / f"{name}.pt", device)

    clean = predict(model, "test", list(ROBUSTNESS_TESTS.values()), args, device)
    y, p = clean["label"], clean["p_fake"][:, 0]  # column 0 = the clean, undegraded test
    robustness = {test: float(((clean["p_fake"][:, k] >= 0.5) == y).mean()) for k, test in enumerate(ROBUSTNESS_TESTS)}
    pred = (p >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary")
    cm = confusion_matrix(y, pred)
    fake = y == 1
    per_class = {"Real photos": float((pred[~fake] == 0).mean())}
    for i, label in enumerate(SOURCES.values()):
        per_class[label] = float(pred[clean["source"] == i].mean())

    history_file = out_dir / "history.json"
    history = json.loads(history_file.read_text()) if history_file.exists() else None
    metrics = {
        "name": name, "backbone": checkpoint["backbone"], "augmentation": checkpoint["augment"],
        "calibration_threshold": checkpoint.get("calibration_threshold"), "test_images": int(len(y)),
        "accuracy": float((pred == y).mean()), "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": {"real_as_real": int(cm[0, 0]), "real_as_ai": int(cm[0, 1]),
                             "ai_as_real": int(cm[1, 0]), "ai_as_ai": int(cm[1, 1])},
        "correct_by_class": per_class,
        "generator_identification_accuracy": float((clean["source_pred"][fake] == clean["source"][fake]).mean()),
        "robustness_accuracy": robustness,
        "cost": {**measure_cost(model), "train_minutes": history["train_minutes"] if history else None,
                 "train_device": history["device"] if history else None},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- figures
    fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
    ax[0, 0].imshow(cm, cmap="Blues")
    for (r, c), value in np.ndenumerate(cm):
        ax[0, 0].text(c, r, value, ha="center", va="center", fontsize=16,
                      color="white" if value > cm.max() / 2 else "black")
    ax[0, 0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "AI"], yticklabels=["Real", "AI"],
                 xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (accuracy {metrics['accuracy']:.1%})")
    fpr, tpr, _ = roc_curve(y, p)
    ax[0, 1].plot(fpr, tpr, lw=2)
    ax[0, 1].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax[0, 1].set(xlabel="False positive rate", ylabel="True positive rate",
                 title=f"ROC curve (AUC {metrics['roc_auc']:.3f})")
    for axis, values, title, colour in [(ax[1, 0], per_class, "Correctly classified, by image source", "#3b7dd8"),
                                        (ax[1, 1], robustness, "Accuracy under feed-style degradation", "#2a9d8f")]:
        bars = axis.barh(list(values)[::-1], [v * 100 for v in values.values()][::-1], color=colour)
        axis.bar_label(bars, fmt="%.1f%%", padding=3)
        axis.set(xlim=(0, 112), xlabel="%", title=title)
    fig.suptitle(f"{name}: {checkpoint['backbone']}, augmentation {'on' if checkpoint['augment'] else 'off'}")
    fig.tight_layout()
    fig.savefig(out_dir / "evaluation.png", dpi=150)
    plt.close(fig)

    if history:
        epochs = history["epochs"]
        xs = [e["epoch"] for e in epochs]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for axis, key, title in [(ax[0], "loss", "Loss"), (ax[1], "acc", "Accuracy")]:
            axis.plot(xs, [e[f"train_{key}"] for e in epochs], "o-", label="train")
            axis.plot(xs, [e[f"val_{key}"] for e in epochs], "o-", label="validation")
            unfreeze = sum(e["phase"] == "heads only" for e in epochs) + 0.5
            axis.axvline(unfreeze, color="gray", ls="--", lw=1)
            axis.set(xlabel="Epoch", title=f"{title}  (dashed line: backbone unfrozen)")
            axis.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close(fig)

    print(f"\nTest results for '{name}' ({metrics['test_images']} images)")
    print(f"  accuracy {metrics['accuracy']:.4f} | precision {precision:.4f} | recall {recall:.4f} | "
          f"F1 {f1:.4f} | AUC {metrics['roc_auc']:.4f}")
    print("  correct by class: " + ", ".join(f"{k} {v:.1%}" for k, v in per_class.items()))
    print(f"  generator identification accuracy: {metrics['generator_identification_accuracy']:.1%}")
    print("  robustness: " + ", ".join(f"{k} {v:.1%}" for k, v in robustness.items()))
    c = metrics["cost"]
    print(f"  cost: {c['params_millions']:.2f} M params, {c['size_mb']:.1f} MB, "
          f"{c['gflops']:.2f} GFLOPs, {c['cpu_ms_per_image']:.1f} ms/image on CPU")
    print(f"  saved to {out_dir}")


# ============================================================================= 4. comparison
def compare(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [json.loads(f.read_text()) for f in sorted(RESULTS.glob("*/metrics.json"))]
    if not runs:
        raise SystemExit("No results yet. Run:  python train.py train")
    feed = list(ROBUSTNESS_TESTS)[-1]
    lines = ["| Run | Backbone | Augmentation | Accuracy | AUC | Feed accuracy | Params (M) | GFLOPs | CPU ms/img | Train min |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in runs:
        c = m["cost"]
        lines.append(f"| {m['name']} | {m['backbone']} | {'yes' if m['augmentation'] else 'no'} | "
                     f"{m['accuracy']:.2%} | {m['roc_auc']:.4f} | {m['robustness_accuracy'][feed]:.2%} | "
                     f"{c['params_millions']:.2f} | {c['gflops']:.2f} | {c['cpu_ms_per_image']:.1f} | "
                     f"{c['train_minutes'] or float('nan'):.1f} |")
    table = "\n".join(lines)
    (RESULTS / "comparison.md").write_text(f"# Run comparison\n\n'Feed accuracy' = test accuracy after {feed}.\n\n{table}\n")
    print(table)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for m in runs:
        ax[0].scatter(m["cost"]["gflops"], m["accuracy"] * 100, s=90)
        ax[0].annotate(m["name"], (m["cost"]["gflops"], m["accuracy"] * 100),
                       textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax[0].set(xlabel="Compute per image (GFLOPs)", ylabel="Test accuracy (%)", title="Accuracy vs computational cost")
    ax[0].margins(0.25)
    tests = list(ROBUSTNESS_TESTS)
    width = 0.8 / len(runs)
    for k, m in enumerate(runs):
        ax[1].bar(np.arange(len(tests)) + k * width, [m["robustness_accuracy"][t] * 100 for t in tests], width,
                  label=m["name"])
    ax[1].set_xticks(np.arange(len(tests)) + 0.4 - width / 2, [t.replace(" (", "\n(") for t in tests], fontsize=8)
    ax[1].set(ylabel="Test accuracy (%)", title="Robustness to feed-style degradation", ylim=(40, 112))
    ax[1].legend(fontsize=8, loc="upper center", ncol=len(runs))
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=150)
    print(f"Saved {RESULTS / 'comparison.md'} and {RESULTS / 'comparison.png'}")


# ============================================================================= command line
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "compare"])
    parser.add_argument("--model", default="mobilenet_v3_large", choices=list(BACKBONES))
    parser.add_argument("--name", default="detector", help="run name: models/<name>.pt and results/<name>/")
    parser.add_argument("--no-aug", action="store_true", help="train without data augmentation (ablation)")
    parser.add_argument("--head-epochs", type=int, default=2, help="epochs with the backbone frozen")
    parser.add_argument("--epochs", type=int, default=8, help="fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="peak fine-tuning learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
