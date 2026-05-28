import argparse
import os

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import ModelNet10
from model import Pct
from util import IOStream

# TODO:
# - [x] 4 value (sxyz) output decoder
# - [x] batch support
# - [x] loss_fn
# - [x] masking
# - [x] add additional layers
# - [x] add shuffling and normalization in each __getitem__ call
# - [ ] transfer learning to shape classifaction
# - [ ] ? add stop signal


def loss_fn(pcd, predicted_completion):
    bs, _, num_points = pcd.shape

    def point_cloud_distance(point_clouds, query_points):
        sqdists = ((point_clouds - query_points[..., None]) ** 2).sum(1)
        return sqdists.min(axis=1).values

    pcd_so_far = [pcd[..., i + 1 :] for i in range(num_points - 1)]
    dist = [point_cloud_distance(pcd_so_far[i], predicted_completion[..., i]).sum() for i in range(num_points - 1)]

    return sum(dist) / bs


def train(args, io):
    train_loader = DataLoader(
        ModelNet10(partition="train", num_points=args.num_points),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ModelNet10(partition="test", num_points=args.num_points),
        num_workers=args.num_workers,
        batch_size=args.test_batch_size,
        shuffle=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using: {device}")

    model = Pct(args.num_output_points, args.num_attention_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    best_test_loss = np.inf

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for pcd in train_loader:
            pcd = pcd.to(device)

            pcd = pcd.permute(0, 2, 1)
            predicted_completion = model(pcd, device=device)

            optimizer.zero_grad()
            loss = loss_fn(pcd, predicted_completion)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # scheduler.step()
        outstr = f"{epoch} / {args.epochs}  |  train_loss: {train_loss:.2f}"
        io.cprint(outstr)

        ####################
        # Test
        ####################

        if epoch % 5 == 0:
            model.eval()
            test_loss = 0.0
            for pcd in test_loader:
                pcd = pcd.to(device)
                pcd = pcd.permute(0, 2, 1)

                predicted_completion = model(pcd)

                loss = loss_fn(pcd, predicted_completion)
                test_loss += loss.item()

            outstr = f"test_loss: {test_loss:.2f}"
            io.cprint(outstr)

            if test_loss < best_test_loss:
                print("yey progress!")
                best_test_loss = test_loss
                torch.save(model.state_dict(), "checkpoints/%s/models/model.pct" % args.exp_name)


def test(args):
    test_loader = DataLoader(
        ModelNet10(partition="test", num_points=args.num_points),
        batch_size=args.test_batch_size,
        shuffle=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Pct(args.num_output_points, args.num_attention_layers).to(device)

    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    test_loss = 0.0
    for pcd in test_loader:
        pcd = pcd.to(device)
        pcd = pcd.permute(0, 2, 1)

        predicted_completion = model(pcd)

        loss = loss_fn(pcd, predicted_completion)
        test_loss += loss.item()

    outstr = f"test_loss: {test_loss:.2f}"
    print(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="Point Cloud Recognition")
    parser.add_argument("--exp_name", type=str, default="exp", help="Name of the experiment")
    parser.add_argument("--dataset", type=str, default="modelnet10", choices=["modelnet10"])
    parser.add_argument("--batch_size", type=int, default=32, help="Size of batch")
    parser.add_argument("--num_attention_layers", type=int, default=4, help="Num attention layers")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Size of batch")
    parser.add_argument("--num_workers", type=int, default=8, help="Num data loader workers")
    parser.add_argument("--epochs", type=int, default=1000, help="number of episode to train ")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--eval", type=bool, default=False, help="evaluate the model")
    parser.add_argument("--num_points", type=int, default=512, help="num of points to use")
    parser.add_argument("--num_output_points", type=int, default=1, help="predicted num points in single step")
    parser.add_argument("--model_path", type=str, default="", help="Pretrained model path")
    args = parser.parse_args()

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("checkpoints/" + args.exp_name):
        os.makedirs("checkpoints/" + args.exp_name)
    if not os.path.exists("checkpoints/" + args.exp_name + "/" + "models"):
        os.makedirs("checkpoints/" + args.exp_name + "/" + "models")
    os.system("cp main.py checkpoints" + "/" + args.exp_name + "/" + "main.py.backup")
    os.system("cp model.py checkpoints" + "/" + args.exp_name + "/" + "model.py.backup")
    os.system("cp util.py checkpoints" + "/" + args.exp_name + "/" + "util.py.backup")
    os.system("cp data.py checkpoints" + "/" + args.exp_name + "/" + "data.py.backup")

    io = IOStream("checkpoints/" + args.exp_name + "/run.log")
    io.cprint(str(args))

    torch.manual_seed(args.seed)
    if not args.eval:
        train(args, io)
    else:
        test(args)
