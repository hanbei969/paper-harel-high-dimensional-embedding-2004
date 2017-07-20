import numpy as np
from math import inf
from random import randint
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA


class GraphDrawing:

    def __init__(self, graph_file, dimension=5, first=None):
        self.dimension = dimension
        self.first = first

        self.graph = self.load_graph(graph_file)
        self.max_node = self.find_max_node(self.graph)
        self.distance_matrix = self.compute_distance_matrix(self.graph, self.max_node)

    def draw(self):
        self.pivot_nodes = self.choose_pivot_points(self.graph, self.dimension)

        node_id_list = range(1, self.max_node + 1)
        self.coordinates = list(map(
            lambda i: tuple(self.compute_coordinates(i, self.pivot_nodes)),
            node_id_list
        ))

        self.projected = self.project_into_low_dimension(self.coordinates)

    @staticmethod
    def find_max_node(graph):
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

        # choose the first one randomly from the graph
        if self.first is None:
            first = randint(0, max_node) + 1
        else:
            first = self.first

        pivots = [first]

        # find next by k-center problem
        for _ in range(0, dimension - 1):
            next_pivot = self.k_center(self.distance_matrix,
                                       pivots,
                                       self.max_node)
            pivots.append(next_pivot)

        return pivots

    @classmethod
    def compute_distance_matrix(cls, graph, max_node):
        graph_matrix = lil_matrix((max_node, max_node))

        for node_from, node_to, weight in graph:
            graph_matrix[node_from - 1, node_to - 1] = weight

        return shortest_path(graph_matrix,
                             method='D',
                             directed=False,
                             unweighted=False)

    @classmethod
    def k_center(cls, distance_matrix, pivots, max_node):
        distances = []
        for i in range(1, max_node + 1):
            if i not in pivots:
                matrix_i = i - 1
                nearest_pivot = min(
                    pivots,
                    key=lambda p: distance_matrix[matrix_i, p - 1])
                nearest_dis = distance_matrix[matrix_i, nearest_pivot - 1]
                distances.append((i, nearest_dis))

        return max(distances, key=lambda d: d[1])[0]

    def compute_coordinates(self, node_id, pivot_nodes):
        return (self.distance_matrix[node_id - 1, p - 1] for p in pivot_nodes)

    def project_into_low_dimension(self, points):
        X = np.matrix(points)
