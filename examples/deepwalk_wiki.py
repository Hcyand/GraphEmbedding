import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from ge import DeepWalk
from ge.classify import read_node_label, Classifier


def evaluate_embeddings(embeddings):  # 验证embedding效果
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings):  # 绘制embedding分布图
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # read_edgelist从txt文件中读取有向图函数，从边列表中读取图形
    # create_using要创建的图表类型，DiGraph有向图
    # nodetype节点类型
    # data为边缘数据制定字典键名和类型
    G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    evaluate_embeddings(embeddings)
    plot_embeddings(embeddings)
