import torch
import numpy as np


def krum(grads, ATTACK_NUM, MULTI=False):

    new_grads = []                              # 存储选择的梯度值
    grads_indices = []                          # 所选梯度值在原始梯度列表的索引
    left_grads = grads                          # 剩余梯度列表
    org_indices = np.arange(len(grads))         # 原始梯度列表索引

    # 2f + 2 < n
    while len(left_grads) > 2 * ATTACK_NUM + 2:

        torch.cuda.empty_cache()  # 清空GPU缓存

        # 计算各个节点间欧氏距离平方
        dists = []
        for x1 in left_grads:
            dist = []
            for x2 in left_grads:
                dist.append(torch.norm((x1 - x2)) ** 2)
            dist = torch.Tensor(dist).float()
            # 将 x1 的所有计算 dist 值加入 dists 的新维度中
            dists = dist[None, :] if not len(dists) else torch.cat((dists, dist[None, :]), 0)

        # 对距离进行排序
        dists = torch.sort(dists, dim=1)[0]

        # 选择与其最近的 n-f-1 个计算得分
        length = len(left_grads) - 2 - ATTACK_NUM

        scores = torch.sum(dists[:, :length], dim=1)

        # 返回排序后的值所对应的下标
        indices = torch.argsort(scores)[:length]

        # 添加选中的梯度的索引
        grads_indices.append(org_indices[indices[0].cpu().numpy()])

        # 在所有索引中删除已经选中的索引
        org_indices = np.delete(org_indices, indices[0].cpu().numpy())

        # 添加选择的梯度值
        new_grads = left_grads[indices[0]][None, :] \
            if not len(new_grads) else torch.cat((new_grads, left_grads[indices[0]][None, :]), 0)

        # 在剩余中删除已选中的梯度
        left_grads = torch.cat((left_grads[:indices[0]], left_grads[indices[0] + 1:]), 0)

        # 若不是 multi-krum 算法, 只取一个分数
        if not MULTI:
            break

    # 计算聚合后的梯度 (取平均)
    new_grad = torch.mean(new_grads, dim=0)

    return new_grad, np.array(grads_indices)
