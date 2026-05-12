from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDED_SOURCE_DIRS = {
    "__pycache__",
    "runs",
    "split_dataset",
    "splits",
    "train",
    "val",
    "valid",
    "vali",
    "test",
}


class CustomMangoCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float,
        conv_channels: Sequence[int],
        kernel_size: int,
        use_batchnorm: bool,
        activation: str,
        classifier_hidden: Sequence[int],
    ) -> None:
        super().__init__()
        in_channels = 3
        feature_blocks: List[nn.Module] = []
        for out_channels in conv_channels:
            feature_blocks.append(
                self._block(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                )
            )
            in_channels = out_channels
        feature_blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(
            *feature_blocks,
        )

        classifier_layers: List[nn.Module] = [nn.Flatten()]
        in_features = conv_channels[-1]
        for hidden_units in classifier_hidden:
            classifier_layers.extend([
                nn.Dropout(dropout),
                nn.Linear(in_features, hidden_units),
                self._activation(activation),
            ])
            in_features = hidden_units
        classifier_layers.extend([
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        ])
        self.classifier = nn.Sequential(*classifier_layers)

    @staticmethod
    def _activation(name: str) -> nn.Module:
        if name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if name == "elu":
            return nn.ELU(inplace=True)
        return nn.ReLU(inplace=True)

    @classmethod
    def _block(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_batchnorm: bool,
        activation: str,
    ) -> nn.Sequential:
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([
            cls._activation(activation),
            nn.MaxPool2d(kernel_size=2),
        ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split mango image folders and train a CNN classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder with one subfolder per class.")
    parser.add_argument("--split-dir", type=Path, default=None, help="Output folder for train/val/test split.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Folder where training outputs are saved.")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run.")
    parser.add_argument("--rebuild-split", action="store_true", help="Recreate the split folder from scratch.")
    parser.add_argument("--split-only", action="store_true", help="Only create the train/val/test folders.")
    parser.add_argument("--split-unit", choices=["file", "source"], default="file",
                        help="Use file for exact image splits, or source to keep related crop files together.")
    parser.add_argument("--val-folder", type=str, default="val", help="Validation folder name inside split-dir.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--conv-channels", type=str, default="32,64,128,256",
                        help="Custom CNN filters per convolution block.")
    parser.add_argument("--kernel-size", type=int, default=3, help="Convolution kernel size. Use an odd number.")
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "elu"], default="relu")
    parser.add_argument("--no-batchnorm", action="store_true", help="Disable batch normalization in conv blocks.")
    parser.add_argument("--classifier-hidden", type=str, default="",
                        help="Optional dense layer sizes before output, e.g. 256 or 512,128.")
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="plateau")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience. Use 0 to disable.")
    parser.add_argument("--class-weights", action="store_true", help="Weight loss by class frequency.")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision when CUDA is available.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    args.data_dir = (args.data_dir or script_dir).resolve()
    args.split_dir = (args.split_dir or args.data_dir / "split_dataset").resolve()
    run_name = args.run_name or time.strftime("cnn_%Y%m%d_%H%M%S")
    args.run_dir = (args.run_dir or args.data_dir / "runs" / run_name).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_ratios(args: argparse.Namespace) -> None:
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must add to 1.0, got {total:.4f}")
    if min(args.train_ratio, args.val_ratio, args.test_ratio) <= 0:
        raise ValueError("All split ratios must be greater than zero.")


def coerce_int_list(
    value,
    default: Sequence[int] | str | None,
    name: str,
    allow_empty: bool = False,
) -> Tuple[int, ...]:
    if value is None:
        value = default

    if isinstance(value, str):
        if not value.strip():
            values: Tuple[int, ...] = ()
        else:
            values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        values = tuple(int(part) for part in value)

    if not values and not allow_empty:
        raise ValueError(f"{name} must contain at least one positive integer.")
    if any(number <= 0 for number in values):
        raise ValueError(f"{name} must contain only positive integers.")
    return values


def validate_model_hyperparameters(args: argparse.Namespace) -> None:
    args.conv_channels = coerce_int_list(args.conv_channels, "32,64,128,256", "conv_channels")
    args.classifier_hidden = coerce_int_list(
        args.classifier_hidden,
        "",
        "classifier_hidden",
        allow_empty=True,
    )
    if args.kernel_size <= 0 or args.kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be a positive odd number, such as 3 or 5.")
    args.use_batchnorm = not args.no_batchnorm


def iter_images(folder: Path) -> List[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_class_dirs(data_dir: Path, split_dir: Path) -> List[Path]:
    class_dirs: List[Path] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.resolve() == split_dir:
            continue
        if child.name.lower() in EXCLUDED_SOURCE_DIRS:
            continue
        if iter_images(child):
            class_dirs.append(child)

    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders with images were found in {data_dir}. "
            "Expected one folder per class."
        )
    return class_dirs


def split_counts(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def source_key(path: Path) -> str:
    return re.sub(r"_cls\d+_id\d+$", "", path.stem)


def split_files(files: Sequence[Path], args: argparse.Namespace) -> Dict[str, List[Path]]:
    rng = random.Random(args.seed)
    train_count, val_count, _ = split_counts(len(files), args.train_ratio, args.val_ratio)

    if args.split_unit == "file":
        shuffled = list(files)
        rng.shuffle(shuffled)
        return {
            "train": shuffled[:train_count],
            "val": shuffled[train_count:train_count + val_count],
            "test": shuffled[train_count + val_count:],
        }

    grouped: Dict[str, List[Path]] = defaultdict(list)
    for path in files:
        grouped[source_key(path)].append(path)

    groups = list(grouped.values())
    rng.shuffle(groups)
    targets = {"train": train_count, "val": val_count, "test": len(files) - train_count - val_count}
    splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

    for group in sorted(groups, key=len, reverse=True):
        split_name = max(targets, key=lambda name: targets[name] - len(splits[name]))
        splits[split_name].extend(group)
    return splits


def safe_remove_split_dir(split_dir: Path, data_dir: Path) -> None:
    resolved_split = split_dir.resolve()
    resolved_data = data_dir.resolve()
    if resolved_split == resolved_data or resolved_data not in resolved_split.parents:
        raise ValueError(f"Refusing to delete split dir outside data dir: {resolved_split}")
    shutil.rmtree(resolved_split)


def split_exists(split_dir: Path, val_folder: str) -> bool:
    return all((split_dir / name).is_dir() for name in ("train", val_folder, "test"))


def prepare_split(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    split_names = {"train": "train", "val": args.val_folder, "test": "test"}

    if args.split_dir.exists() and args.rebuild_split:
        safe_remove_split_dir(args.split_dir, args.data_dir)

    if split_exists(args.split_dir, args.val_folder):
        print(f"Using existing split: {args.split_dir}")
        return count_split_images(args.split_dir, args.val_folder)

    class_dirs = find_class_dirs(args.data_dir, args.split_dir)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, int]] = {}

    for class_dir in class_dirs:
        files = iter_images(class_dir)
        splits = split_files(files, args)
        summary[class_dir.name] = {}

        for split_key, split_folder in split_names.items():
            destination_class_dir = args.split_dir / split_folder / class_dir.name
            destination_class_dir.mkdir(parents=True, exist_ok=True)
            for source_path in splits[split_key]:
                relative_path = source_path.relative_to(class_dir)
                destination_path = destination_class_dir / relative_path
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
            summary[class_dir.name][split_key] = len(splits[split_key])

    manifest = {
        "data_dir": str(args.data_dir),
        "split_dir": str(args.split_dir),
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "split_unit": args.split_unit,
        "counts": summary,
    }
    (args.split_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created split: {args.split_dir}")
    return summary


def count_split_images(split_dir: Path, val_folder: str) -> Dict[str, Dict[str, int]]:
    split_names = {"train": "train", "val": val_folder, "test": "test"}
    class_names = sorted(path.name for path in (split_dir / "train").iterdir() if path.is_dir())
    summary: Dict[str, Dict[str, int]] = {}
    for class_name in class_names:
        summary[class_name] = {}
        for split_key, split_folder in split_names.items():
            summary[class_name][split_key] = len(iter_images(split_dir / split_folder / class_name))
    return summary


def print_split_summary(summary: Dict[str, Dict[str, int]]) -> None:
    print("\nSplit summary")
    print("-" * 52)
    print(f"{'Class':<20} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 52)
    totals = Counter()
    for class_name, counts in summary.items():
        print(f"{class_name:<20} {counts['train']:>8} {counts['val']:>8} {counts['test']:>8}")
        totals.update(counts)
    print("-" * 52)
    print(f"{'Total':<20} {totals['train']:>8} {totals['val']:>8} {totals['test']:>8}\n")


def build_transforms(args: argparse.Namespace) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.no_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            normalize,
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_transform, eval_transform = build_transforms(args)
    train_ds = datasets.ImageFolder(args.split_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(args.split_dir / args.val_folder, transform=eval_transform)
    test_ds = datasets.ImageFolder(args.split_dir / "test", transform=eval_transform)

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("Train, validation, and test class folders do not match.")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(
    num_classes: int,
    dropout: float,
    conv_channels: Sequence[int] | str = (32, 64, 128, 256),
    kernel_size: int = 3,
    use_batchnorm: bool = True,
    activation: str = "relu",
    classifier_hidden: Sequence[int] | str = (),
) -> nn.Module:
    return CustomMangoCNN(
        num_classes=num_classes,
        dropout=dropout,
        conv_channels=coerce_int_list(conv_channels, (32, 64, 128, 256), "conv_channels"),
        kernel_size=kernel_size,
        use_batchnorm=use_batchnorm,
        activation=activation,
        classifier_hidden=coerce_int_list(classifier_hidden, (), "classifier_hidden", allow_empty=True),
    )


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    return None


def make_class_weights(loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = [label for _, label in loader.dataset.samples]
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (num_classes * max(1, counts[index])) for index in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


@contextmanager
def autocast_context(use_amp: bool):
    try:
        with torch.amp.autocast("cuda", enabled=use_amp):
            yield
    except (AttributeError, TypeError):
        with torch.cuda.amp.autocast(enabled=use_amp):
            yield


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    use_amp: bool = False,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast_context(use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def save_metrics_csv(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_training_plot(run_dir: Path, rows: List[Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in rows], label="train")
    plt.plot(epochs, [row["val_loss"] for row in rows], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_acc"] for row in rows], label="train")
    plt.plot(epochs, [row["val_acc"] for row in rows], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close()


def save_confusion_matrix(run_dir: Path, matrix: np.ndarray, classes: List[str]) -> None:
    np.savetxt(
        run_dir / "confusion_matrix.csv",
        matrix,
        fmt="%d",
        delimiter=",",
        header=",".join(classes),
        comments="",
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=35, ha="right")
    plt.yticks(tick_marks, classes)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def train(args: argparse.Namespace) -> None:
    train_loader, val_loader, test_loader, classes = build_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    model = build_model(
        num_classes=len(classes),
        dropout=args.dropout,
        conv_channels=args.conv_channels,
        kernel_size=args.kernel_size,
        use_batchnorm=args.use_batchnorm,
        activation=args.activation,
        classifier_hidden=args.classifier_hidden,
    ).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    class_weights = make_class_weights(train_loader, len(classes), device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = make_grad_scaler(use_amp)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    (args.run_dir / "config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = args.run_dir / "best_model.pt"
    last_path = args.run_dir / "last_model.pt"
    history: List[Dict[str, float]] = []

    print(f"Training on {device} with classes: {', '.join(classes)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.6g}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_name": "custom_cnn",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": classes,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "val_acc": val_acc,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, last_path)

        improved = val_acc > best_val_acc or (np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1

        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement.")
            break

    save_metrics_csv(args.run_dir / "metrics.csv", history)
    save_training_plot(args.run_dir, history)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device, use_amp=use_amp)
    y_true, y_pred = predict(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    (args.run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(args.run_dir, matrix, classes)

    print("\nFinal test results")
    print("-" * 52)
    print(f"Best epoch: {best_epoch}")
    print(f"Test loss:  {test_loss:.4f}")
    print(f"Test acc:   {test_acc:.4f}")
    print("\nClassification report")
    print(report)
    print(f"Saved best model: {best_path}")
    print(f"Saved run outputs: {args.run_dir}")


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    validate_ratios(args)
    validate_model_hyperparameters(args)
    set_seed(args.seed)
    summary = prepare_split(args)
    print_split_summary(summary)

    if args.split_only:
        print("Split complete. Training skipped because --split-only was used.")
        return

    train(args)


if __name__ == "__main__":
    main()
