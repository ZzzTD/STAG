import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model_1 import *

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def plot_loss_data(data):
    # 使用Matplotlib绘制线图
    fig2 = plt.figure()
    plt.semilogy(range(len(data)), data)
    plt.savefig("../result/gru/loss.png")
    plt.close(fig2)

    # plt.plot(data, marker='o')

    # # 添加标题
    # plt.title("loss results Plot")
    #
    # # 显示图例
    # plt.legend(["Loss"])

    plt.show()


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, config):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == 'MS':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def calculate_mae(labels, results):
    # 平均绝对误差
    mae = np.mean(np.abs(labels - results))
    return mae


def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_csv(config.data_path)  # 填你自己的数据地址,自动选取你最后一列数据为特征列 # 添加你想要预测的特征列
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口

    # 将特征列移到末尾
    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = df.columns[1:]
    df_data = df[cols_data]

    # 这里加一些数据的预处理, 最后需要的格式是pd.series
    true_data = df_data.values

    # 定义标准化优化器
    # 定义标准化优化器
    scaler = StandardScaler()
    scaler.fit(true_data)
    train_timesteps = int(true_data.shape[0] * 0.8)
    train_data = true_data[:train_timesteps]
    test_data = true_data[train_timesteps-config.window_size:]
    # train_data = true_data[int(0.3 * len(true_data)):]
    # valid_data = true_data[int(0.15 * len(true_data)):int(0.30 * len(true_data))]
    # test_data = true_data[:int(0.15 * len(true_data))]

    # 进行标准化处理
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)

    # 转化为深度学习模型需要的类型Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)

    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)


    return train_loader, test_loader,  scaler


def train(model, args, scaler, device):
    start_time = time.time()  # 计算起始时间
    model = model
    loss_function = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)  ## lr?
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    epochs = args.epochs
    model.train()  # 训练模式
    results_loss = []
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    for i in tqdm(range(epochs)):
        losss = []
        for seq, labels in train_loader:
            optimizer.zero_grad()

            results = model(seq)

            single_loss = loss_function(results, labels)

            single_loss.backward()

            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())
        tqdm.write(f"\t Epoch {i + 1} / {epochs}, Loss: {sum(losss) / len(losss)}")
        results_loss.append(sum(losss) / len(losss))
        scheduler.step()
        torch.save(model.state_dict(), '../result/gru/prediction.pth')
        time.sleep(0.1)

    # valid_loss = valid(model, args, scaler, valid_loader)
    # 尚未引入学习率计划后期补上
    # 保存模型

    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")
    plot_loss_data(results_loss)
    result = pd.DataFrame(data={'loss': results_loss})
    filepath = '../result/gru/'
    result.to_csv(filepath + 'loss.csv')


def test(model, args, test_loader, scaler):
    # 加载模型进行预测
    losss = []
    model = model
    model.load_state_dict(torch.load('../result/gru/prediction.pth'))
    model.eval()  # 评估模式
    results = []
    labels = []
    for seq, label in test_loader:
        pred = model(seq)
        # mae = calculate_mae(pred.detach().cpu().numpy(),
        #                     np.array(label.detach().cpu()))  # MAE误差计算绝对值(预测值  - 真实值)
        # losss.append(mae)
        pred = pred[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])
    labels = np.array(labels)
    results = np.array(results)
    rmse = np.sqrt(mean_squared_error(labels, results))
    mae = mean_absolute_error(labels, results)
    mape = np.mean(np.abs((labels - results) / labels)) * 100
    r2 = r2_score(labels, results)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("R2:", r2)

    filename = f"../result/gru/error.txt"
    with open(filename, "w") as file:
        # 输出评估值到文件
        file.write(f"\nRMSE for emd: {rmse}\n")
        file.write(f"MAE for emd : {mae}\n")
        file.write(f"MAPE for emd : {mape}\n")
        file.write(f"R^2 for emd : {r2}\n")
    # print("测试集误差MAE:", losss)
    # 绘制历史数据
    result = pd.DataFrame(data={'Q(t+1)': results, 'Q(t+1)truth': labels})
    filepath = '../result/gru/'
    result.to_csv(filepath + 'prediction.csv')

    fig3 = plt.figure()
    plt.plot(results, label='Predicted')
    plt.plot(labels, label="True")
    plt.legend(loc='upper left')
    plt.savefig(filepath + "/predict.png")
    plt.close(fig3)
    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='TCN', help="模型持续更新")
    parser.add_argument('-window_size', type=int, default=12, help="时间窗口大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=1, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
    parser.add_argument('-data_path', type=str, default='../data/L_river_waterlevel.csv', help="你的数据数据地址")
    parser.add_argument('-target', type=str, default='LZ_Z', help='你需要预测的特征列，这个值会最后保存在csv文件里')
    parser.add_argument('-input_size', type=int, default=11, help='你的特征个数不算时间那一列')
    parser.add_argument('-feature', type=str, default='MS', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
    parser.add_argument('-model_dim', type=list, default=[64, 64, 64], help='这个地方是这个TCN卷积的关键部分,它代表了TCN的层数我这里输'
                                                                              '入list中包含三个元素那么我的TCN就是三层，这个根据你的数据复杂度来设置'
                                                                              '层数越多对应数据越复杂但是不要超过5层')
    parser.add_argument('-hidden_size', type=int, default=64, help="隐藏层单元数")
    parser.add_argument('-attention_hiddensize', type=int, default=512, help="注意力隐藏层单元数")
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    # parser.add_argument('-drop_out', type=float, default=0.05, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=100, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=64, help="批次大小")
    parser.add_argument('-save_path', type=str, default='models')

    # model
    parser.add_argument('-kernel_sizes', type=int, default=3)
    parser.add_argument('-laryer_num', type=int, default=1)
    parser.add_argument('-bigru_layers', type=int, default=1)

    parser.add_argument('-bigru_hidden_size', type=int, default=2)

    # device
    parser.add_argument('-use_gpu', type=bool, default=True)
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

    # option
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)
    parser.add_argument('-lr-scheduler', type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备:", device)
    train_loader, test_loader,  scaler = create_dataloader(args, device)

    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size

    # 实例化模型
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = TemporalConvNet(
            args.input_size,
            args.output_size,
            args.pre_len,
            args.model_dim,
            args.window_size,
            args.hidden_size,
            args.kernel_sizes).to(
            device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    # 训练模型
    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, scaler, device)
    if args.test:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        test(model, args, test_loader, scaler)
