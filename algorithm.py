import numpy as np
from random import randint


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
        return np.genfromtxt(graph_file, delimiter=',', dtype=[('from', np.intp), ('to', np.intp), ('weight', np.float)])

    @staticmethod
    def choose_pivot_points(graph, dimension):
        points = []

        # choose the first one randomly from the graph
        first = graph[randint(0, len(graph))]

        # compute the i-th coordinate
        coordinate = []
        for _ in range(0, dimension):
            pass

        return None
