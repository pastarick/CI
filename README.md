# CI
My personal Computational Intelligence course repo


## Description

### /TSP:
 - *__tsp.py__*: tsp solver with custom 'heuristic' permutation (not working very well)
 - *__tsp_inverover.py__*: tsp solver with pure ga (generations + inver-over + elitism) (**works definitely better**))

### /Connect4:
 - *__main.py__*: just the calls to the methods for playing the game
 - *__Connect4.py__*: connect-4 wrapper class + minmax (with alpha-beta pruning + hard cut-off) + montecarlo solver
 - *__TreeNode.py__*: implementation of a recursive data structure to represent and explore the game tree
