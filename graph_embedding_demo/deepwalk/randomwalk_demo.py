import itertools
import random

from joblib import Parallel, delayed


class RandomWalker:
    def __init__(self, G):
        self.G = G

    def deepwalk_walk(self, walk_length, start_node):
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))  # cur节点的邻居节点
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))  # walk加入邻居节点
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        G = self.G

        nodes = list(G.nodes())  # 节点

        # Parallel函数是并行执行多个函数，每个函数都是立即执行
        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in self.partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))
        return walks

    def partition_num(self, num, workers):
        if num % workers == 0:
            return [num // workers] * workers
        else:
            return [num // workers] * workers + [num % workers]
