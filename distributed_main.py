from distributed_train import init_processes, train
import torch.distributed as dist
import torch.multiprocessing as mp

# 常量
WORKERS = 2


if __name__ == "__main__":

    processes = []
    mp.set_start_method("spawn")

    for rank in range(WORKERS):
        process = mp.Process(target=init_processes, args=(rank, WORKERS, train))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    # 释放资源
    dist.destroy_process_group()
