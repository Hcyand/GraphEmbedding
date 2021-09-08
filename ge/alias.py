"""
Alias采样方法：时间复杂度是O(1)
常见的采样方法：顺序查找为O(N)、二分查找为O(logN)
Alias分为两步：做表、根据表采样
做表：O(N), 根据表采样：O(1)
1。每个概率乘以N，总面积变为N
2。再将单位长度内面积大于1的进行切割，保证每个单位长度内面积为1，同时至多两个事件面积
"""
import numpy as np


def create_alias_table(area_ratio):
    """

    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    # 返回大于等于1和小于1的index
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    # 修改柱状图，获得单位长度内面积都为1
    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                 (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """

    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


if __name__ == '__main__':
    area_ratio = [0.1, 0.15, 0.05, 0.2, 0.3, 0.2]
    a, b = create_alias_table(area_ratio)
    index = alias_sample(a, b)

