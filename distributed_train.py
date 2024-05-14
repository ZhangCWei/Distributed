import math
import os
import torch
import simpleCNN
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# 常量
WORKERS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 10


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(rank, world):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 根据节点数重新划分batch
    batch_size = math.ceil(BATCH_SIZE / float(world))

    # 计算每个rank负责的样本数量
    num_samples = len(train_data) // world
    start_index = rank * num_samples
    end_index = start_index + num_samples if rank < world - 1 else len(train_data)

    # 根据索引划分子数据集
    subset_indices = list(range(start_index, end_index))
    subset = Subset(train_data, subset_indices)

    # 创建对应的DataLoader
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return train_loader, batch_size


def average_gradients(model):
    world = float(dist.get_world_size())
    for param in model.parameters():
        # param.grad.data是每个参数的梯度
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)  # 求和
        param.grad.data /= world                                # 取平均



def train(rank, world):
    model = simpleCNN.SimpleCNN().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    torch.manual_seed(1234)
    train_loader, batch_size = load_data(rank, world)

    # 对于该进程，一共有多少个batch
    num_batches = math.ceil(len(list(train_loader)) / float(batch_size))

    train_accs = []

    # epoch的数目与单进程一样
    for epoch in range(NUM_EPOCHS+1):
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()             # 反向传播求梯度
            average_gradients(model)    # Synchronous All-Reduce SGD
            optimizer.step()            # 更新参数

            _, predicted = output.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()

        train_acc = correct_train / total_train
        train_accs.append(train_acc)
        print(f'Rank{rank} epoch {epoch} Loss: { train_loss / num_batches:.6f}  Accuracy:{train_acc:.6f}')



def init_processes(rank, world, func, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=world)
    func(rank, world)

    if torch.distributed.is_initialized():
        print("当前处于分布式训练环境")
    else:
        print("当前未处于分布式训练环境")
