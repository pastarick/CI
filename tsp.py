# Riccardo Sepe - 287760

import logging
from math import sqrt
from typing import Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

NUM_CITIES = 33
STEADY_STATE = 1000


class Tsp:

    def __init__(self, num_cities: int, seed: Any = None) -> None:
        if seed is None:
            seed = num_cities
        self._num_cities = num_cities
        self._graph = nx.DiGraph()

        np.random.seed(seed)
        for c in range(num_cities):
            self._graph.add_node(c, pos=(np.random.random(), np.random.random()))

        # my attribute np.array([[f(i,j) for i in range(n)] for j in range(n)])
        self._matrix = np.array([[self._distance(n1, n2) for n1 in range(num_cities)] for n2 in range(num_cities)],
                                dtype=float)

        # for utility purposes, it's useful to put zeros as NaNs
        for i in range(num_cities):
            self._matrix[i, i] = np.nan

    def _distance(self, n1, n2) -> int:
        pos1 = self._graph.nodes[n1]['pos']
        pos2 = self._graph.nodes[n2]['pos']
        return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0])**2 +
                                                         (pos1[1] - pos2[1])**2))

    def distance(self, n1, n2) -> int:
        return self._matrix[n1, n2]

    def evaluate_solution(self, solution: np.array) -> float:
        total_cost = 0
        tmp = solution.tolist() + [solution[0]]
        for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
            total_cost += self._matrix[n1, n2]
        return total_cost

    def plot(self, path: np.array = None) -> None:
        if path is not None:
            self._graph.remove_edges_from(list(self._graph.edges))
            tmp = path.tolist() + [path[0]]
            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
                self._graph.add_edge(n1, n2)
        plt.figure(figsize=(12, 5))
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink')
        if path is not None:
            plt.title(f"Current path: {self.evaluate_solution(path):,}")
        plt.show()

    def evaluate_path(self, path: np.array) -> list:
        # path_costs[0] will be the distance between 0th and 1st element of path
        # path_costs[-1] will be the distance between last and 0th element of path
        path_costs = [self.distance(path[i], path[i+1]) for i in range(NUM_CITIES-1)]
        path_costs.append(self.distance(path[-1], path[0]))
        return path_costs

    def print_matrix(self) -> None:
        print(self._matrix)
        print(self._matrix.shape)

    def clean_matrix(self, quadrants: dict) -> None:
        # ne and sw
        for n1, n2 in product(quadrants['ne'], quadrants['sw']):
            self._matrix[n1, n2] = np.inf
        for n1, n2 in product(quadrants['nw'], quadrants['se']):
            self._matrix[n1, n2] = np.inf

    @property
    def graph(self) -> nx.digraph:
        return self._graph

    @property
    def dist_matrix(self) -> np.array:
        return self._matrix


def plot_history(history):
    x, y = [i for i, j in history], [j for i, j in history]
    plt.plot(x, y)
    plt.xlabel('Steps')
    plt.ylabel('Cost')
    plt.show()


def tweak_inversion(solution: np.array, *, pm: float = .1) -> np.array:
    new_solution = solution.copy()
    p = 0
    while p < pm:
        i1 = np.random.randint(0, solution.shape[0])
        i2 = np.random.randint(0, solution.shape[0])

        if i1 == i2:
            continue
        if i1 > i2:
            i1, i2 = i2, i1

        new_solution[i1:i2] = np.flip(new_solution[i1:i2])

        p = np.random.random()
    return new_solution


def tweak(solution: np.array, path_costs: list, problem: Tsp):
    # search for the 2 longest paths between 4 distinct nodes: if we're lucky, they form kind of an X in the graph:
    # the aim is to flip the 2 other paths between these nodes in 2 different children, hoping to remove the X
    # and finally pick the best from the offspring
    size = len(solution)
    m1 = int(np.argmax(path_costs))
    # retrieve the first 2 of the 4 nodes: the ending points of the m1 edge
    nodes = [solution[m1], solution[(m1 + 1) % size]]
    v = path_costs[m1]
    path_costs[m1] = 0
    m2 = np.argmax(path_costs)
    path_costs[m1] = v
    nodes += [solution[m2], solution[(m2 + 1) % size]]

    # There are 2 cases:
    # 1. if the 4 nodes are not distinct, they form a 'triangle' (because the 2 edges are surely distinct)
    #   >In such case, remove the repeated node from the sequence, as it's in the wrong place (triangular inequality)
    #    and put it somewhere else (randomly?)
    # 2. if the 4 nodes are distinct, they form a 'square'
    #   >if they form kind of an X (which means that the 2 edges cover the 'diagonals' of the 'square'):
    #     *-* <- ceiling
    #      V
    #      Λ
    #     *-* <- floor
    #       >flip the floor
    #   >if they don't form kind of an X:
    #     *-* <- ceiling
    #     | |
    #     | |
    #     *-* <- floor
    #       >generate 1 children, in which the floor and ceiling subsequences are randomly moved somewhere else
    sol = list(solution)
    if nodes == list(np.unique(nodes)):

        # square
        dm = problem.dist_matrix
        critical = dm[nodes[0], nodes[1]]
        if dm[nodes[0], nodes[2]] > critical and dm[nodes[0], nodes[3]] > critical:
            # there's a cross
            # flip the floor
            i1 = sol.index(nodes[0])
            i2 = sol.index(nodes[1])
            if i1 > i2:
                i1, i2 = i2, i1
            solution[i1:i2+1] = np.flip(solution[i1:i2+1])
        else:
            # there isn't a cross
            # pick a random node and put it between the 2 couples of nodes
            pickable = [n for n in range(size) if n not in nodes]
            np.random.shuffle(pickable)
            n1 = pickable.pop()
            n2 = pickable.pop()
            i1 = sol.index(nodes[0])
            i2 = sol.index(nodes[2])
            sol.remove(n1)
            sol.remove(n2)
            sol.insert(i1, n1)
            sol.insert(i2, n2)
            solution = np.array(sol)
    else:
        # triangle
        # extract the repeated node and put it somewhere else
        if nodes[0] in nodes[1:]:
            n = nodes[0]
        elif nodes[1] in nodes[2:]:
            n = nodes[1]
        else:
            n = nodes[2]

        i = sol.index(n)

        sol.remove(n)
        i2 = np.random.choice(list(range(0, i)) + list(range(i+1, size)))
        sol.insert(i2, n)
        solution = np.array(sol)

    return solution


def main():

    problem = Tsp(NUM_CITIES)

    solution = np.array(range(NUM_CITIES))
    np.random.shuffle(solution)
    original_cost = solution_cost = problem.evaluate_solution(solution)
    path_costs = problem.evaluate_path(solution)
    problem.plot(solution)
    print(f"Original cost: {solution_cost:.0f}")

    history = [(0, solution_cost)]
    steady_state = 0
    step = 0
    improvement_factor = 0
    while steady_state < STEADY_STATE*50:
    # while improvement_factor < 3:
        step += 1
        steady_state += 1
        # new_solution = tweak_inversion(solution, pm=.55)
        new_solution = tweak(solution, path_costs, problem)

        new_solution_cost = problem.evaluate_solution(new_solution)
        path_costs = problem.evaluate_path(new_solution)
        if new_solution_cost < solution_cost:
            solution = new_solution
            solution_cost = new_solution_cost
            history.append((step, solution_cost))
            steady_state = 0
            improvement_factor = original_cost/solution_cost

    problem.plot(solution)
    print(f"New cost: {solution_cost:.0f}")
    print(f"Improvement factor: {improvement_factor*100:.0f}%")
    print(f"Number of steps: {step}")
    plot_history(history)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    main()
