import numpy as np
from math import inf
from random import randint
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path


class GraphDrawing:

    def __init__(self, graph_file, dimension=50):
        self.graph_file = graph_file
        self.dimension = dimension

    def draw(self):
        graph = self.load_graph(self.graph_file)
        pivots = self.choose_pivot_points(graph, self.dimension)
        #self.draw_in_high_dimension(pivots, graph)
        #self.project_into_low_dimension()

    @staticmethod
    def load_graph(graph_file):
        return np.genfromtxt(graph_file, delimiter=',', dtype=[
            ('from', np.intp),
            ('to', np.intp),
            ('weight', np.float)
        ])

    @classmethod
    def choose_pivot_points(cls, graph, dimension):
        pivots = []
        coordinates = []

        max_node_id = max([
            *list(max(graph, key=lambda i: i[0]))[0:2],
            *list(max(graph, key=lambda i: i[1]))[0:2]])

        distance_matrix = cls.distance_matrix(graph, max_node_id)

        # choose the first one randomly from the graph
        first = randint(0, max_node_id) + 1
        pivots = [first]

        # find next by k-center problem
        for _ in range(0, dimension):
            next_pivot = cls.k_center(distance_matrix, pivots, max_node_id)
            pivots.append(next_pivot)

        return None

    @classmethod
    def distance_matrix(cls, graph, max_node_id):
        graph_matrix = lil_matrix((max_node_id, max_node_id))

        for node_from, node_to, weight in graph:
            graph_matrix[node_from - 1, node_to - 1] = weight

        return shortest_path(graph_matrix,
                             method='D',
                             directed=False,
                             unweighted=False)

    @classmethod
    def k_center(cls, distance_matrix, pivots, max_node_id):
        distances = []
        for i in range(1, max_node_id + 1):
            if i not in pivots:
                nearest_pivot = min(pivots, key=lambda p: distance_matrix[i-1,p-1])
                nearest_dis = distance_matrix[i-1, nearest_pivot-1]
                distances.append((i, nearest_dis))

        return max(distances, key=lambda d: d[1])[0]
