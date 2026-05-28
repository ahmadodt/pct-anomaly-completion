import argparse
import os
import shutil

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import ModelNet10
from model import Pct
from util import IOStream


def chamfer_distance(predicted, target):
    pred_to_target = torch.cdist(predicted, target).min(dim=2).values
    target_to_pred = torch.cdist(target, predicted).min(dim=2).values
    return pred_to_target.mean() + target_to_pred.mean()


def make_loader(args, partition, batch_size, shuffle):
    return DataLoader(
        ModelNet10(
            partition=partition,
            num_points=args.num_points,
            num_missing_points=args.num_missing_points,
            data_root=args.data_root,
        ),
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0

    for partial, missing, _full in loader:
        partial = partial.to(device).permute(0, 2, 1)
        missing = missing.to(device)

        if is_train:
            optimizer.zero_grad()

        predicted_missing = model(partial)
        loss = chamfer_distance(predicted_missing, missing)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        raise ValueError("DataLoader produced no batches")
    return total_loss / total_batches


def train(args, io):
    train_loader = make_loader(args, "train", args.batch_size, True)
    test_loader = make_loader(args, "test", args.test_batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    io.cprint(f"using: {device}")

    model = Pct(args.num_missing_points, args.num_attention_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_test_loss = np.inf

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        io.cprint(f"{epoch} / {args.epochs} | train_chamfer: {train_loss:.6f}")

        if epoch % args.eval_every == 0:
            with torch.no_grad():
                test_loss = run_epoch(model, test_loader, device)
            io.cprint(f"{epoch} / {args.epochs} | test_chamfer: {test_loss:.6f}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), f"checkpoints/{args.exp_name}/models/model.pct")
                io.cprint(f"saved checkpoint with test_chamfer: {best_test_loss:.6f}")


def test(args):
    test_loader = make_loader(args, "test", args.test_batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pct(args.num_missing_points, args.num_attention_layers).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    with torch.no_grad():
        test_loss = run_epoch(model, test_loader, device)

    print(f"test_chamfer: {test_loss:.6f}")


def backup_source_files(exp_name):
    checkpoint_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(os.path.join(checkpoint_dir, "models"), exist_ok=True)
    for filename in ("main.py", "model.py", "util.py", "data.py"):
        shutil.copy2(filename, os.path.join(checkpoint_dir, f"{filename}.backup"))


def parse_args():
    parser = argparse.ArgumentParser(description="Point cloud completion decoder")
    parser.add_argument("--exp_name", type=str, default="exp", help="Name of the experiment")
    parser.add_argument("--dataset", type=str, default="modelnet10", choices=["modelnet10"])
    parser.add_argument("--data_root", type=str, default=None, help="Path to extracted ModelNet10 folder")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Evaluate a saved model")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--num_points", type=int, default=512, help="Number of full-shape points")
    parser.add_argument("--num_missing_points", type=int, default=128, help="Number of missing points to predict")
    parser.add_argument("--model_path", type=str, default="", help="Pretrained model path")
    parser.add_argument("--num_attention_layers", type=int, default=4, help="Number of attention layers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    backup_source_files(args.exp_name)

    io = IOStream(f"checkpoints/{args.exp_name}/run.log")
    io.cprint(str(args))

    if not args.eval:
        train(args, io)
    else:
        test(args)
