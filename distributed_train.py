import os
import torch
import random
import simpleCNN
from krum import krum
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 训练参数
MOMENTUM = 0.5
NUM_EPOCHS = 80
BATCH_SIZE = 128
LEARNING_RATE = 0.01

# 攻击参数
ATTACK_TYPE = 2     # 0: 不攻击; 1: Label-Flipping; 2: Data-Flipping
ATTACK_RATE = 1.0   # 攻击成功比率
ATTACK_NUM = 5      # 攻击节点数量

# 防御参数
DEFENSE_TYPE = 1    # 0: 不启用防御; 1: 启用防御
MULTI = True        # True: 启用Multi Krum; False: 启用Krum

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_processes(rank, world, func, q, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=world)

    if torch.distributed.is_initialized():
        print("Rank", rank, "处于分布式训练环境")
    else:
        print("Rank", rank, "未处于分布式训练环境")

    q.put(func(rank, world, (world - rank <= ATTACK_NUM)))


def all_reduce_average(model):
    world = float(dist.get_world_size())
    for param in model.parameters():
        # param.grad.data是每个参数的梯度
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)  # 求和
        param.grad.data /= world  # 取平均


def train(rank, world, IS_ATTACK=False):
    model = simpleCNN.SimpleCNN().to(device)
    model = nn.parallel.DistributedDataParallel(model)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    torch.manual_seed(1234)

    # 导入数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 分布式数据
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world, rank=rank)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False,
                                               pin_memory=True, sampler=train_sampler)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    train_accs = []
    test_accs = []

    # epoch的数目与单进程一样
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            # Label-Flipping
            if IS_ATTACK and ATTACK_TYPE == 1 and random.random() < ATTACK_RATE:
                target = 9 - target

            # 求损失
            loss = criterion(output, target)
            train_loss += loss.item()

            # 反向传播求梯度
            loss.backward()

            # Bit-Flipping
            if IS_ATTACK and ATTACK_TYPE == 2:
                for param in model.parameters():
                    if param.grad is not None:
                        grad = param.grad
                        mask = torch.rand_like(grad) < ATTACK_RATE
                        grad[mask] = -grad[mask]

            if DEFENSE_TYPE == 0 or 2 * ATTACK_NUM + 2 >= world:
                # 不采用 Krum, 执行 Synchronous All-Reduce SGD
                all_reduce_average(model)
            else:
                # 采用 Krum
                if rank == 0:
                    for p in model.parameters():
                        # 新建list存储各节点的梯度
                        grad_list = [torch.zeros_like(p.grad) for _ in range(world)]
                        # 获取所有节点的梯度
                        dist.gather(p.grad, grad_list, async_op=False)
                        # 计算所有节点的krum梯度
                        new_grad, _ = krum(grad_list, ATTACK_NUM, MULTI)
                        # 新建一个list存储将要发到各个节点的平均梯度
                        scatter_list = [new_grad for _ in range(world)]
                        # 将所有的值发送到各个节点
                        dist.scatter(p.grad, scatter_list, src=0, async_op=False)
                else:
                    for p in model.parameters():
                        # 将grad值发送到node0
                        dist.gather(p.grad, async_op=False)
                        # 接收node0发送过来的grad值
                        dist.scatter(p.grad, src=0, async_op=False)

            # 更新参数
            optimizer.step()

            _, predicted = output.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()

        train_acc = correct_train / total_train
        train_accs.append(train_acc)

        print(
            f'Rank{rank}\tEpoch {epoch}\tLoss: {train_loss / len(train_loader):.6f}  Accuracy:{100 * train_acc:.2f}% ')

        if rank == 0:
            test(model, criterion, test_loader, test_accs)

    return train_accs, test_accs


def test(model, criterion, test_loader, test_accs):
    # 测试模型
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predicted = output.max(1)
            total_test += target.size(0)
            correct_test += predicted.eq(target).sum().item()

    test_acc = correct_test / total_test
    test_accs.append(test_acc)

    print(f'-----\nTest Accuracy:{100 * test_acc:.2f}%\n-----')
