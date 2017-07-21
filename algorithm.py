import numpy as np
import matplotlib.pyplot as plt
import logging
from math import inf
from random import randint
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA

logging.basicConfig(filename='runtime.log',
                    format='%(levelname)s:%(message)s',
                    level=logging.INFO)


class GraphDrawing:

    def __init__(self, dimension=50):
        self.dimension = dimension

    def transform(self, graph_file, first_node=None):
        logging.info('loading graph')
        self.graph = self.load_graph(graph_file)
        self.node_count = self.find_node_count(self.graph)
        self.node_range = range(1, self.node_count + 1)

        logging.info('computing distance matrix')
        self.distance_matrix = self.compute_distance_matrix(self.graph,
                                                            self.node_count)

        if first_node is None:
            self.first_node = randint(0, self.node_count) + 1
        else:
            self.first_node = first_node

        logging.info('finding pivots')
        self.pivot_nodes = self.choose_pivot_points(self.graph, self.dimension)

        logging.info('drawing graph in high dimensional space')
        self.points = list(map(
            lambda i: tuple(self.distance_matrix[i - 1, p - 1]
                            for p in self.pivot_nodes),
            self.node_range
        ))

        logging.info('project into a low dimension use PCA')
        pca = PCA(n_components=2, copy=True)
        self.transformed_points = pca.fit_transform(self.points)

    @staticmethod
    def find_node_count(graph):
        return max([
            *list(max(graph, key=lambda i: i[0]))[0:2],
            *list(max(graph, key=lambda i: i[1]))[0:2]])

    @staticmethod
    def load_graph(graph_file):
        return np.genfromtxt(graph_file, delimiter=',', dtype=[
            ('from', np.intp),
            ('to', np.intp),
            ('weight', np.float)
        ])

    def choose_pivot_points(self, graph, dimension):
        pivots = []
        coordinates = []
        pivots = [self.first_node]

        # find next by k-center problem
        for _ in range(0, dimension - 1):
            next_pivot = self.k_center(self.distance_matrix,
                                       pivots,
                                       self.node_count)
            pivots.append(next_pivot)

        return pivots

    @staticmethod
    def compute_distance_matrix(graph, node_count):
        graph_matrix = lil_matrix((node_count, node_count))

        for node_from, node_to, weight in graph:
            graph_matrix[node_from - 1, node_to - 1] = weight

        return shortest_path(graph_matrix,
                             method='D',
                             directed=False,
                             unweighted=False)

    def k_center(self, distance_matrix, pivots, node_count):

        distances = []
        for i in self.node_range:
            if i not in pivots:
                matrix_i = i - 1
                nearest_pivot = min(
                    pivots,
                    key=lambda p: distance_matrix[matrix_i, p - 1])
                nearest_dis = distance_matrix[matrix_i, nearest_pivot - 1]
                distances.append((i, nearest_dis))

        return max(distances, key=lambda d: d[1])[0]

    def plot(self, filename):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, frame_on=False)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        logging.info('plot start')
        ax.scatter(self.transformed_points[:,0],
                   self.transformed_points[:,1],
                   s=1,
                   facecolor="black",
                   linewidth=0)

        for node_from, node_to, weight in self.graph:
            coord_from = list(self.transformed_points[node_from - 1])
            coord_to = list(self.transformed_points[node_to - 1])
            ax.plot([coord_from[0], coord_to[0]],
                    [coord_from[1], coord_to[1]],
                    linewidth=0.5,
                    color='black')

        logging.info('plot end')
        plt.savefig(filename, dpi=600)
        plt.close()
