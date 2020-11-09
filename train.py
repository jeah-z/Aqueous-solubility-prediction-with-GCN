# -*- coding:utf-8 -*-
"""Sample training code
"""
import numpy as np
import pandas as pd
import argparse
import torch as th
import torch.nn as nn
from sch import SchNetModel
from mgcn import MGCNModel
from torch.utils.data import DataLoader
from Alchemy_dataset import TencentAlchemyDataset, batcher


def dataset_split(file):
    delaney = pd.read_csv("delaney.csv")
    test_set = delaney.sample(frac=0.1, random_state=0)
    train_set = delaney.drop(test_set.index)
    test_set.to_csv("delaney_test.csv", index=False)
    train_set.to_csv("delaney_train.csv", index=False)


def train(model="sch", epochs=80, device=th.device("cpu"), dataset=''):
    print("start")
    train_dir = "./"
    train_file = dataset+"_train.csv"
    alchemy_dataset = TencentAlchemyDataset()
    alchemy_dataset.mode = "Train"
    alchemy_dataset.transform = None
    alchemy_dataset.file_path = train_file
    alchemy_dataset._load()

    test_dataset = TencentAlchemyDataset()
    test_dir = train_dir
    test_file = dataset+"_valid.csv"
    test_dataset.mode = "Train"
    test_dataset.transform = None
    test_dataset.file_path = test_file
    test_dataset._load()

    alchemy_loader = DataLoader(
        dataset=alchemy_dataset,
        batch_size=10,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=10,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=0,
    )

    if model == "sch":
        model = SchNetModel(norm=False, output_dim=1)
    elif model == "mgcn":
        model = MGCNModel(norm=False, output_dim=1)
    print(model)
    # if model.name in ["MGCN", "SchNet"]:
    #     model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std, device)
    model.to(device)
    # print("test_dataset.mean= %s" % (alchemy_dataset.mean))
    # print("test_dataset.std= %s" % (alchemy_dataset.std))

    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):

        w_loss, w_mae = 0, 0
        model.train()

        for idx, batch in enumerate(alchemy_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)

            res = model(batch.graph)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            w_mae += mae.detach().item()
            w_loss += loss.detach().item()
        w_mae /= idx + 1
        w_loss /= idx + 1

        print("Epoch {:2d}, loss: {:.7f}, mae: {:.7f}".format(
            epoch, w_loss, w_mae))

        val_loss, val_mae = 0, 0
        for jdx, batch in enumerate(test_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)

            res = model(batch.graph)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            # optimizer.zero_grad()
            # mae.backward()
            # optimizer.step()

            val_mae += mae.detach().item()
            val_loss += loss.detach().item()
        val_mae /= jdx + 1
        val_loss /= jdx + 1
        print(
            "Epoch {:2d}, val_loss: {:.7f}, val_mae: {:.7f}".format(
                epoch, val_loss, val_mae
            ))

        if epoch % 200 == 0:
            th.save(model.state_dict(), './'+dataset+"/model_"+str(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        "--model",
                        help="model name (sch, mgcn)",
                        default="sch")
    parser.add_argument("--epochs", help="number of epochs", default=10000)
    parser.add_argument("--dataset", help="dataset to train", default="")
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    assert args.model in ["sch", "mgcn"]
    # dataset_split("delaney.csv")
    train(args.model, int(args.epochs), device, args.dataset)
