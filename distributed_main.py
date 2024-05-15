import time
import torch.distributed as dist
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from distributed_train import init_processes, train


# 常量
WORKERS = 10


if __name__ == "__main__":

    start_time = time.time()

    q = mp.Queue()
    processes = []
    # mp.set_start_method("spawn")

    for rank in range(WORKERS):
        process = mp.Process(target=init_processes, args=(rank, WORKERS, train, q))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    end_time = time.time()
    print("Total time: %s s" % (end_time-start_time))

    accs_all = [q.get() for p in processes]

    # 绘制学习曲线
    for i, (train_accs, test_accs) in enumerate(accs_all):
        plt.plot(range(1, len(accs_all[0]) + 1), train_accs, label=f'Worker {i+1}')
        plt.plot(range(1, len(accs_all[0]) + 1), test_accs, label=f'Test Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

    # 释放资源
    # dist.destroy_process_group()
