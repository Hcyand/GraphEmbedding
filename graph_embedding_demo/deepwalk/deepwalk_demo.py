# 输出embedding

from gensim.models import Word2Vec

from randomwalk_demo import RandomWalker


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):
        self.graph = graph  # 有向图数据
        self.w2v_model = None  # word2vec model
        self._embedding = {}  # w2v embedding

        self.walker = RandomWalker(graph)
        print('make sentences...')
        self.sentences = self.walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=workers,
                                                    verbose=1)  # 随机游走产生的sentence

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)  # 词频小于min_count的词会被丢弃
        kwargs["size"] = embed_size  # 特征向量维度
        kwargs["sg"] = 1  # 0为CBOW算法，1为skip-gram算法
        kwargs["hs"] = 1  # 1则会采用hierarchica·softmax技巧，0（default）则negative sampling会被使用
        kwargs["workers"] = workers  # 控制训练的并行数
        kwargs["window"] = window_size  # 当前词与预测词在一个句子中最大的距离
        kwargs["iter"] = iter  # 迭代次数

        print('train model start...')
        model = Word2Vec(**kwargs)
        print('train model end!')
        self.w2v_model = model
        return model

    def get_embedding(self):
        if self.w2v_model is None:
            print('model not train')
            return {}
        self._embedding = {}
        print('out embedding...')
        for word in self.graph.nodes():
            self._embedding[word] = self.w2v_model.wv[word]
        return self._embedding
