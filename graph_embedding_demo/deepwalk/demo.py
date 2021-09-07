import networkx as nx
from deepwalk_demo import DeepWalk

if __name__ == '__main__':
    print('load data to DiGraph')
    G = nx.read_edgelist('../../data/wiki/Wiki_edgelist.txt', create_using=nx.DiGraph(), nodetype=None,
                         data=[('weight', int)])
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embedding = model.get_embedding()
