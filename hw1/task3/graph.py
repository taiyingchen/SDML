from collections import defaultdict
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_file')
    return parser.parse_args()


class Graph(object):
    def __init__(self, graph):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.init_graph(graph)

    def init_graph(self, graph):
        max_index = graph.max() + 1
        self.V = max_index

        self.V_seen = set()
        for u, v in graph:
            self.add_edge(u, v)
            self.V_seen.update([u, v])

    # function to add an edge to graph
    def add_edge(self, u, v):
        self.graph[u].append(v)

    # A recursive function used by topologicalSort
    def topological_sort_util(self, v, visited, stack):
        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topological_sort_util(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    # The function to do Topological Sort. It uses recursive
    # topological_sort_util()
    def topological_sort(self):
        # Mark all the vertices as not visited
        visited = [False]*self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topological_sort_util(i, visited, stack)

        # Remove unseen vertices
        V_unseen = []
        for v in stack:
            if v not in self.V_seen:
                V_unseen.append(v)
        for v in V_unseen:
            stack.remove(v)

        return stack


def main(args):
    g = Graph(args.graph_file)
    print(g.graph)
    print(g.topological_sort())


if __name__ == '__main__':
    args = parse_args()
    main(args)
