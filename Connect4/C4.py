import numpy as np
from collections import deque, Counter
import logging
from itertools import product

NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
FOUR = 4

MAGIC_SQUARE = np.array([[16, 2, 3, 13],
                         [5, 11, 10, 8],
                         [9, 7, 6, 12],
                         [4, 14, 15, 1]])

AI_PLAYER = -1
USER_PLAYER = 1


def valid_moves(board):
    """Returns columns where a disc may be played"""
    return [n for n in range(NUM_COLUMNS) if board[n, COLUMN_HEIGHT - 1] == 0]


def play(board, column, player):
    """Updates `board` as `player` drops a disc in `column`"""
    (index,) = next((i for i, v in np.ndenumerate(board[column]) if v == 0))
    board[column, index] = player


def take_back(board, column):
    """Updates `board` removing top disc from `column`"""
    (index,) = [i for i, v in np.ndenumerate(board[column]) if v != 0][-1]
    board[column, index] = 0


def four_in_a_row(board, player):
    """Checks if `player` has a 4-piece line"""
    return (
        any(
            np.all(board[c, r] == player)
            for c in range(NUM_COLUMNS)
            for r in (list(range(n, n + FOUR)) for n in range(COLUMN_HEIGHT - FOUR + 1))
        )
        or any(
            np.all(board[c, r] == player)
            for r in range(COLUMN_HEIGHT)
            for c in (list(range(n, n + FOUR)) for n in range(NUM_COLUMNS - FOUR + 1))
        )
        or any(
            np.all(board[diag] == player)
            for diag in (
                (range(ro, ro + FOUR), range(co, co + FOUR))
                for ro in range(0, NUM_COLUMNS - FOUR + 1)
                for co in range(0, COLUMN_HEIGHT - FOUR + 1)
            )
        )
        or any(
            np.all(board[diag] == player)
            for diag in (
                (range(ro, ro + FOUR), range(co + FOUR - 1, co - 1, -1))
                for ro in range(0, NUM_COLUMNS - FOUR + 1)
                for co in range(0, COLUMN_HEIGHT - FOUR + 1)
            )
        )
    )


def print_board(board):
    """
    This function just prints the human-readable version of the board, i.e. with the downward oriented gravity
    """
    # print(np.rot90(board))
    b = np.rot90(board)
    for i in range(COLUMN_HEIGHT):
        for j in range(NUM_COLUMNS):
            if b[i][j] == USER_PLAYER:
                print("| ● ", end='')
            elif b[i][j] == AI_PLAYER:
                print("| ○ ", end='')
            else:
                print("|   ", end='')
        print('|')

    print("-", end='')
    for i in range(NUM_COLUMNS):
        print("-"*4, end='')
    print('')

    for i in range(NUM_COLUMNS):
        print(f"| {i+1} ", end='')
    print('|')


def best_move(board):
    columns = valid_moves(board)
    # look for cells where the disk will go
    cells = []
    for c in columns:
        cells.append((c, np.where(board[c, :] == 0)[0][0]))
    print(cells)


def _mc(board, player):
    p = -player
    while valid_moves(board):
        p = -p
        c = np.random.choice(valid_moves(board))
        play(board, c, p)
        if four_in_a_row(board, p):
            return p
    return 0


def montecarlo(board, player):
    montecarlo_samples = 100
    cnt = Counter(_mc(np.copy(board), player) for _ in range(montecarlo_samples))
    return (cnt[1] - cnt[-1]) / montecarlo_samples


def eval_board(board, player):
    if four_in_a_row(board, 1):
        # Alice won
        return 1
    elif four_in_a_row(board, -1):
        # Bob won
        return -1
    else:
        # Not terminal, let's simulate...
        return montecarlo(board, player)


def is_terminal(board):
    return four_in_a_row(board, USER_PLAYER) or four_in_a_row(board, AI_PLAYER) or not valid_moves(board)


def utility(board, cell):
    player = board[cell[0], cell[1]]
    # tmp_board is the board with only the moves played by ‘player‘
    tmp_board = (board.copy() + player*np.absolute(board))//2

    # i, j are the coordinates of the top-leftmost cell in sub square
    # OBS: j won't change over the loop
    i = max(0, cell[0]-FOUR+1)
    j = max(0, cell[1]-FOUR+1)

    # x, y are the coordinates of current cell in sub square
    # OBS: also y won't change over the loop
    y = cell[1] - j

    m = 0
    while i <= NUM_COLUMNS-FOUR:
        sub = tmp_board[i:i+4, j:j+4]

        # tmp_sub is the result of applying the MAGIC_SQUARE mask to sub
        tmp_sub = sub * MAGIC_SQUARE

        x = cell[0] - i
        # y = cell[1] - j

        if player == USER_PLAYER:
            m = max(m, np.sum(tmp_sub[x, :]), np.sum(tmp_sub[:, y]))
        else:
            m = min(m, np.sum(tmp_sub[x, :]), np.sum(tmp_sub[:, y]))

        # if cell is on the diagonal, count also the diagonal
        if x == y:
            if player == USER_PLAYER:
                m = max(m, np.trace(tmp_sub))
            else:
                m = min(m, np.trace(tmp_sub))

        # if cell is on the antidiagonal (even size -> cell cannot be on both diag and antidiag)
        if x+y == FOUR-1:
            if player == USER_PLAYER:
                m = max(m, np.trace(np.flip(tmp_sub)))
            else:
                m = min(m, np.trace(np.flip(tmp_sub)))
        i += 1

    return m


def minmax(board, move, depth, player, alpha, beta):
    if is_terminal(board) or depth == 0:
        (index,) = [i for i, v in np.ndenumerate(board[move]) if v != 0][-1]
        cell = (move, index)
        return move, player * utility(board, cell)

    if player == AI_PLAYER:
        v1 = (None, np.inf)
        for m in valid_moves(board):
            # alpha = min(alpha, minmax(board, m, depth-1, -player))
            play(board, m, player)
            v2 = minmax(board, m, depth-1, -player, alpha, beta)
            take_back(board, m)
            if v2[1] < v1[1]:
                v1 = v2
            if v2[1] <= alpha:
                return v1
            if v2[1] < beta:
                beta = v1[1]
    else:
        v1 = (None, -np.inf)
        for m in valid_moves(board):
            # alpha = max(alpha, minmax(board, m, depth-1, -player))
            play(board, m, player)
            v2 = minmax(board, m, depth-1, -player, alpha, beta)
            take_back(board, m)
            if v2[1] > v1[1]:
                v1 = v2
            if v2[1] >= beta:
                return v1
            if v2[1] > alpha:
                alpha = v2[1]

    return v1


def ai_move(board, user_move):
    depth = 3
    return minmax(board, user_move, depth, USER_PLAYER, -np.inf, np.inf)[0]


def main():
    board = np.zeros((NUM_COLUMNS, COLUMN_HEIGHT), dtype=np.byte)
    # player = np.random.choice([1, -1])
    #
    # while True:
    #     if player == 1:
    #         ply = int(input("Please chose the column of your move: "))
    #     else:
    #         # first move should always be in ‘real‘ central column, i.e. in 3
    #         ply = 3
    #
    #     play(board, ply, player)
    #     print_board(board)
    #     player = -player
    #     best_move(board)

    player = -1
    ply = 3
    play(board, ply, player)
    # ply = minmax(board, 3, 3, -player, -np.inf, np.inf)
    print(f"AI playing move {ply}...")
    print_board(board)

    player = -player
    while True:
        if player == AI_PLAYER:
            ply = minmax(board, 3, 3, player, -np.inf, np.inf)
            print(f"AI playing column {ply}...")
            play(board, ply[0], player)
            print_board(board)
            if four_in_a_row(board, player):
                print(f"AI won the game.")
                break
        else:
            ply = int(input("Please chose the column of your move: "))-1
            print(f"You played column {ply}...")
            play(board, ply, player)
            print_board(board)
            if four_in_a_row(board, player):
                print(f"You won the game!")
                break




        player = -player


if __name__ == '__main__':
    main()

