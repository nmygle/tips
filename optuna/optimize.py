from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

import optuna

class Unit(nn.Module):
    def __init__(self, n_in, n_out, flg_bn, act):
        """
        ニューラルネットワークの活性化関数を含む全結合層を定義するモデル
        Args:
            n_in (int) : 入力のノード数
            n_out (int) : 出力のノード数
            flg_bn (bool) : batch normalizationを入れるかどうか
            act (str) : 活性化関数の種類
        """
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        
        self.flg_bn = flg_bn
        if self.flg_bn:
            self.bn = nn.BatchNorm1d(n_out)
        
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "prelu":
            self.act = nn.PReLU()
        elif act == "mish":
            self.act = nn.Mish()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act is None:
            self.act = None
        else:
            assert False

    def forward(self, x):
        x = self.linear(x)
        if self.flg_bn:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        else:
            x
        return x


# モデルの定義
class Net(nn.Module):
    def __init__(self, n_in, n_out, n_hidden_node, n_hidden_layer, flg_bn, act):
        super().__init__()
        self.first = Unit(n_in, n_hidden_node, flg_bn, act)
        
        self.middle = nn.ModuleList()
        for k in range(n_hidden_layer):
            self.middle.append(Unit(n_hidden_node, n_hidden_node, flg_bn, act))
        
        self.last = Unit(n_hidden_node, n_out, flg_bn=False, act=None)

    def forward(self, x):
        x = self.first(x)
        for k in range(len(self.middle)):
            x = self.middle[k](x)
        return self.last(x)


# 訓練関数
def train(model, optimizer, criterion, data_loader, device):
    model.train()
    losses = []
    for pos, (x_data, y_data) in enumerate(data_loader):
        x_data, y_data = x_data.to(device), y_data.to(device)
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"\r[in train loop] {pos}, {loss.item():.5f}", end="")
    return losses

def train_lbfgs(model, optimizer, criterion, data_loader, device):
    model.train()
    losses = []
    for pos, (x_data, y_data) in enumerate(data_loader):
        x_data, y_data = x_data.to(device), y_data.to(device)
        def closure():
            optimizer.zero_grad()
            output = model(x_data)
            loss = criterion(output, y_data)
            loss.backward()
            return loss
        optimizer.step(closure)
        loss = closure()
        losses.append(loss.item())
        print(f"\r[in train loop] {pos}, {loss.item():.5f}", end="")
    return losses


# 評価関数
@torch.no_grad()
def valid(model, criterion, data_loader, device):
    model.eval()
    losses = []
    for x_data, y_data in data_loader:
        x_data, y_data = x_data.to(device), y_data.to(device)
        output = model(x_data)
        loss = criterion(output, y_data)
        losses.append(loss.item())
    return losses


def run():
    device = torch.device("cuda") # ("cpu" | "cuda" | "cuda:0")
    
    # 各学習のエポック数
    n_epoch = 100
    th_early_stopping = 5
    n_worker = 4
    
    n_trials = 1000

    # データのロード
    df = pd.read_csv("Output2_total_Conv_SDN_max_10_exp_1.csv")
    x_stat = pd.read_csv("med/x_stat.csv", index_col=0)
    y_stat = pd.read_csv("med/y_stat.csv", index_col=0)
    feature_col = list(x_stat.index)
    target_col = list(y_stat.index)
    train_index = np.loadtxt("med/train_index.txt", delimiter=",")
    test_index = np.loadtxt("med/test_index.txt", delimiter=",")
    
    x_data = (df.loc[train_index, feature_col] - x_stat["mean"]) / x_stat["std"]
    y_data = (df.loc[train_index, target_col] - y_stat["mean"]) / y_stat["std"]
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    # x_test = (df.loc[test_index, target_col] - x_stat["mean"]) / x_stat["std"]
    # y_test = (df.loc[test_index, target_col] - y_stat["mean"]) / y_stat["std"]
    

    # データセットの作成
    x_train_tensor = torch.Tensor(x_train.values.astype(np.float32))
    y_train_tensor = torch.Tensor(y_train.values.astype(np.float32))

    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)

    x_valid_tensor = torch.Tensor(x_valid.values.astype(np.float32))
    y_valid_tensor = torch.Tensor(y_valid.values.astype(np.float32))

    valid_dataset = torch.utils.data.TensorDataset(x_valid_tensor, y_valid_tensor)
    
    # 目標関数
    def objective(trial):
        # バッチサイズ
        batch_size = 2 ** trial.suggest_int("batch_size", 4, 12)
        # 隠れ層のノードサイズ
        n_hidden_node = 2 ** trial.suggest_int("n_hidden_node", 2, 6)
        # 隠れ層のレイヤー数
        n_hidden_layer = trial.suggest_int("n_hidden_layer", 1, 6)
        # 活性化関数
        act = trial.suggest_categorical("act", ["relu", "prelu", "sigmoid", "mish"])
        # batch normalizationを入れるか
        flg_bn = trial.suggest_categorical("flg_bn", [True, False])
        # 最適化手法
        opt_name = trial.suggest_categorical("opt_name", ["SGD", "Adam", "AdamW", "RAdam", "LBFGS"])
        # 学習係数
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        if opt_name != "LBFGS":
            # 正則化係数
            weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)

        # データローダーの作成
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_worker,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_worker,
            pin_memory=True,
        )

        # モデルの定義
        model = Net(
            n_in=len(feature_col),
            n_out=len(target_col),
            n_hidden_node=n_hidden_node,
            n_hidden_layer=n_hidden_layer,
            flg_bn=flg_bn,
            act=act,
        ).to(device)
        
        # 損失関数の定義
        criterion = nn.MSELoss()
        
        # 最適化の定義
        if opt_name == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif opt_name == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif opt_name == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif opt_name == "RAdam":
            optimizer = torch.optim.RAdam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif opt_name == "LBFGS":
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=lr
            )
        else:
            assert False

        history = []
        best_loss = np.inf
        best_epoch = -1
        count_not_update_best_loss_step = 0
        for epoch in range(n_epoch):
            if opt_name == "LBFGS":
                train_losses = train_lbfgs(model, optimizer, criterion, train_loader, device)
            else:
                train_losses = train(model, optimizer, criterion, train_loader, device)
            valid_losses = valid(model, criterion, valid_loader, device)

            train_losses_mean = np.mean(train_losses)
            valid_losses_mean = np.mean(valid_losses)
            if valid_losses_mean < best_loss:
                best_loss = valid_losses_mean
                best_epoch = epoch
            else:
                count_not_update_best_loss_step += 1

            print(f"\r{epoch+1} / {n_epoch}, {train_losses_mean:.5f}, {valid_losses_mean:.5f}, [best] {best_loss:.5f} ({best_epoch})", end="")
            print()
            history.append({
                "epoch": epoch,
                "train_loss": train_losses_mean,
                "valid_loss": valid_losses_mean,
            })
            if count_not_update_best_loss_step > th_early_stopping:
                break
        return best_loss
    
    study_name = 'SDN_max_10_exp_1'
    # 結果をSQLiteのデータベースに保存
    # study_nameを統一しておけば途中で中断してもロードして始める
    # 逆にstudy_nameを変えないと前回の最適化結果をひきずってしまうので新規に始める際にはstudy_nameもしくはstorage名を変更すること
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage='sqlite:///optuna_nn_study.db',
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)

    with open("best_param.json", "wb") as fp:
        json.dump(study.best_trial, fp, indent=2)

if __name__ == "__main__":
    run()

    
