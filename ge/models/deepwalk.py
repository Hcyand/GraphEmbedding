# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.(http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)



"""
from gensim.models import Word2Vec

from ..walker import RandomWalker


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph  # 有向图数据
        self.w2v_model = None  # word2vec模型
        self._embeddings = {}  # w2v结果

        self.walker = RandomWalker(
            graph, p=1, q=1, )
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)  # 返回deepwalk后结果

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences  # list
        kwargs["min_count"] = kwargs.get("min_count", 0)  # 词频小于min_count的词会被丢弃
        kwargs["size"] = embed_size  # 特征向量维度
        kwargs["sg"] = 1  # 0为CBOW算法，1为skip-gram算法
        kwargs["hs"] = 1  # 1则会采用hierarchica·softmax技巧，0（default）则negative sampling会被使用
        kwargs["workers"] = workers  # 控制训练的并行数
        kwargs["window"] = window_size  # 当前词与预测词在一个句子中最大的距离
        kwargs["iter"] = iter  # 迭代次数

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]  # 读取出向量

        return self._embeddings
