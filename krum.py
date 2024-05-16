import torch
import numpy as np


def krum(grads, ATTACK_NUM, MULTI=False):

    new_grads = []                              # 存储选择的梯度值
    grads_indices = []                          # 所选梯度值在原始梯度列表的索引
    remaining = grads                           # 剩余列表, 用于计算去重
    all_indices = np.arange(len(grads))         # 原始梯度列表索引

    # 2f + 2 < n
    while len(remaining) > 2 * ATTACK_NUM + 2:

        torch.cuda.empty_cache()  # 清空GPU缓存

        # 计算各个节点间欧氏距离平方
        dists = []
        for x1 in remaining:
            dist = []
            for x2 in remaining:
                dist.append(torch.norm((x1 - x2)) ** 2)
            dist = torch.Tensor(dist).float()
            # 将 x1 的所有计算 dist 值加入 dists 的新维度中
            dists = dist[None, :] if not len(dists) else torch.cat((dists, dist[None, :]), 0)

        # 对距离进行排序
        dists = torch.sort(dists, dim=1)[0]

        # 选择与其最近的 n-f-1 个计算得分
        scores = torch.sum(dists[:, :len(remaining) - 2 - ATTACK_NUM], dim=1)

        # 返回排序后的值所对应的下标
        indices = torch.argsort(scores)[:len(remaining) - 2 - ATTACK_NUM]

        # 添加选中的梯度的索引
        grads_indices.append(all_indices[indices[0].cpu().numpy()])

        # 在所有索引中删除已经选中的索引
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())

        # 添加选择的梯度值
        new_grads = remaining[indices[0]][None, :] if not len(new_grads) else torch.cat((new_grads, remaining[indices[0]][None, :]), 0)

        # 在剩余中删除已选中的梯度
        remaining = torch.cat((remaining[:indices[0]], remaining[indices[0] + 1:]), 0)

        # 若不是 multi-krum 算法, 只取一个分数
        if not MULTI:
            break

    # 计算聚合后的梯度 (取平均)
    new_grad = torch.mean(new_grads, dim=0)

    return new_grad, np.array(grads_indices)
