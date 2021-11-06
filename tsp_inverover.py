# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

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


def tweak_generations(solution: np.array, problem: Tsp, ngens=2):
    # the idea is to generate up to ngens generations of offspring and select the best
    # of the last generation
    ngens = max(ngens, 1)

    # first generation done by hand
    best = solution.copy()
    worst = np.flip(solution.copy())

    best_cost = problem.evaluate_solution(best)

    while ngens > 0:
        ngens -= 1
        # it's called worst just for algo purposes
        worst = mate(best, worst)
        worst_cost = problem.evaluate_solution(worst)

        if worst_cost < best_cost:
            best, worst = worst, best
            best_cost, worst_cost = worst_cost, best_cost

    return best, best_cost


def mate(first: np.array, second: np.array) -> np.array:
    size = first.shape[0]
    child = first.copy()

    c1 = np.random.choice(first)
    pm = .15

    while True:
        p = np.random.random()
        if p < pm:
            c2 = np.random.choice(child)
        else:
            i = int(np.where(second == c1)[0][0])
            c2 = second[(i+1) % size]

        j = int(np.where(child == c1)[0][0])
        if child[(j-1) % size] == c2 or child[(j+1) % size] == c2:
            break
        else:
            k = int(np.where(child == c2)[0][0])
            if j > k:
                child[j+1:] = np.flip(child[j+1:])
                child[:k+1] = np.flip(child[:k+1])
            else:
                child[j+1:k+1] = np.flip(child[j+1:k+1])
        c1 = c2

    return child


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
    while steady_state < STEADY_STATE:
    # while improvement_factor < 3:
        step += 1
        steady_state += 1
        # new_solution = tweak_inversion(solution, pm=.55)
        # new_solution, new_solution_cost, path_costs = tweak_double_inversion_2(solution, path_costs, problem, pm=.2)

        ngens = int((steady_state / STEADY_STATE) * NUM_CITIES**(9/8))
        new_solution, new_solution_cost = tweak_generations(solution, problem, ngens)

        # new_solution_cost = problem.evaluate_solution(new_solution)
        # path_costs = problem.evaluate_path(new_solution)
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
