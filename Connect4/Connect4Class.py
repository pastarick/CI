import numpy as np


class Connect4:
    __NUM_COLUMNS = 7
    __COLUMN_HEIGHT = 6
    __FOUR = 4
    __EVALUATION_TABLE = np.rot90(np.array([[3, 4, 5, 7, 5, 4, 3],
                                            [4, 6, 8, 10, 8, 6, 4],
                                            [5, 8, 11, 13, 11, 8, 5],
                                            [5, 8, 11, 13, 11, 8, 5],
                                            [4, 6, 8, 10, 8, 6, 4],
                                            [3, 4, 5, 7, 5, 4, 3]]), 3)
    __DIRECTIONS = {
        "tl": (-1, -1),
        "t":  (-1, 0),
        "tr": (-1, 1),
        "r":  (0, 1),  # OBS: right direction should never be useful
        "dr": (1, 1),
        "d":  (1, 0),
        "dl": (1, -1),
        "l":  (0, -1),
    }

    __DIRECTIONS_ARRAYS = {
        "tl": lambda i, j, board: np.flip(np.diagonal(board[i - 3:i, j - 3:j])),
        "t":  lambda i, j, board: np.flip(board[i - 3:i, j]),
        "tr": lambda i, j, board: np.diagonal(np.flip(board[i - 3:i, j + 1:j + 4], 0)),
        "r":  lambda i, j, board: board[i, j + 1:j + 4],  # OBS: right direction should never be useful
        "dr": lambda i, j, board: np.diagonal(board[i + 1:i + 4, j + 1:j + 4]),
        "d":  lambda i, j, board: board[i + 1:i + 4, j],
        "dl": lambda i, j, board: np.diagonal(np.flip(board[i + 1:i + 4, j - 3:j], 1)),
        "l":  lambda i, j, board: np.flip(board[i, j - 3:j]),
    }

    def __init__(self, num_players):
        self.__board = np.zeros((self.__NUM_COLUMNS, self.__COLUMN_HEIGHT), dtype=np.byte)
        self.__player1 = 1
        self.__player2 = -1
        self.__num_players = num_players

    def play_game(self):
        """
        Only exposed method, creates and handles a game
        """
        if self.__num_players == 1:
            ai = self.__player2
            player = ai
            ply = 3
            print(f"AI playing move {ply}...")
            self.__play(ply, player)
            print(self)

            player = -player
            while True:
                if player == ai:

                    ply = self.__ai_move()
                    print(f"AI playing column ({ply[0] + 1}, {ply[1]})...")
                    self.__play(ply[0], player)
                    print(self)
                    if self.__four_in_a_row(player):
                        print(f"AI won the game.")
                        break
                else:
                    ply = int(input("Please chose the column of your move: ")) - 1
                    print(f"You played column {ply + 1}.")
                    self.__play(ply, player)
                    print(self)
                    if self.__four_in_a_row(player):
                        print(f"You won the game!")
                        break

                player = -player

            print("Terminating game")

    def __valid_moves(self):
        """Returns columns where a disc may be played"""
        return [n for n in range(self.__NUM_COLUMNS) if self.__board[n, self.__COLUMN_HEIGHT - 1] == 0]

    def __play(self, column, player):
        """Updates `board` as `player` drops a disc in `column`"""
        (index,) = next((i for i, v in np.ndenumerate(self.__board[column]) if v == 0))
        self.__board[column, index] = player

    def __take_back(self, column):
        """Updates `board` removing top disc from `column`"""
        (index,) = [i for i, v in np.ndenumerate(self.__board[column]) if v != 0][-1]
        self.__board[column, index] = 0

    def __four_in_a_row(self, player):
        """Checks if `player` has a 4-piece line"""
        return (
                any(
                    np.all(self.__board[c, r] == player)
                    for c in range(self.__NUM_COLUMNS)
                    for r in (list(range(n, n + self.__FOUR)) for n in range(self.__COLUMN_HEIGHT - self.__FOUR + 1))
                )
                or any(
            np.all(self.__board[c, r] == player)
            for r in range(self.__COLUMN_HEIGHT)
            for c in (list(range(n, n + self.__FOUR)) for n in range(self.__NUM_COLUMNS - self.__FOUR + 1))
        )
                or any(
            np.all(self.__board[diag] == player)
            for diag in (
                (range(ro, ro + self.__FOUR), range(co, co + self.__FOUR))
                for ro in range(0, self.__NUM_COLUMNS - self.__FOUR + 1)
                for co in range(0, self.__COLUMN_HEIGHT - self.__FOUR + 1)
            )
        )
                or any(
            np.all(self.__board[diag] == player)
            for diag in (
                (range(ro, ro + self.__FOUR), range(co + self.__FOUR - 1, co - 1, -1))
                for ro in range(0, self.__NUM_COLUMNS - self.__FOUR + 1)
                for co in range(0, self.__COLUMN_HEIGHT - self.__FOUR + 1)
            )
        )
        )

    def __is_terminal(self):
        return self.__four_in_a_row(self.__player1) \
               or self.__four_in_a_row(self.__player2) \
               or not self.__valid_moves()

    def __valid_directions(self, tmp_board, i, j):
        return [d for d in self.__DIRECTIONS.keys()
                if tmp_board[i+self.__DIRECTIONS[d][0], j+self.__DIRECTIONS[d][1]] != 0]

    """
    Utility function section
    """

    @staticmethod
    def __count_vec(vec, player):
        # len(vec) is always 3
        if vec[0] == 0:
            return 0

        k = 1
        while k < 3 and vec[k] == vec[0]:
            k += 1
        if vec[0] == player:
            k += 1
        else:
            k += 0.5

        if k == 4:
            return 25
        return k

    def __cell_value(self, cell):
        tmp_board = np.pad(self.__board, ((3, 3), (3, 3)), mode='constant', constant_values=(0, 0))

        i = cell[0]
        j = cell[1]

        player = self.__board[i, j]

        i += 3
        j += 3

        value = 0
        # value = 0  # can init to 0 because value is always >= 0
        for d in self.__valid_directions(tmp_board, i, j):
            # value = max(value, self.__count_vec(self.__DIRECTIONS_ARRAYS[d](i, j, tmp_board), player))
            value += self.__count_vec(self.__DIRECTIONS_ARRAYS[d](i, j, tmp_board), player)

        return value  # + self.__EVALUATION_TABLE[i-3, j-3]

    """
    MinMax algorithm
    """

    def __ai_move(self):
        depth = 2
        # in the first level of recursion, the move parameter of minmax won't be used
        return self.__minmax(None, depth, self.__player2, -np.inf, np.inf)

    def __minmax(self, move, depth, player, alpha, beta):
        """
        MinMax algorithm implementation with alpha-beta pruning and hard cut-off to @param depth
        """
        if move is not None and (self.__is_terminal() or depth == 0):
            (index,) = [i for i, v in np.ndenumerate(self.__board[move]) if v != 0][-1]
            cell = (move, index)
            # if current player is -1, this means that the top-of-the-stack call was made by player 1
            # -> return score with opposite sign
            return move, -player * self.__cell_value(cell)  # int(player * utility(board, cell))

        # print(f"depth = {depth}")
        if player == self.__player2:
            v1 = (None, np.inf)
            for m in self.__valid_moves():
                # alpha = min(alpha, minmax(board, m, depth-1, -player))
                self.__play(m, player)
                _, v2 = self.__minmax(m, depth - 1, -player, alpha, beta)
                self.__take_back(m)
                if v2 < v1[1]:
                    v1 = m, v2
                if v2 <= alpha:
                    return v1
                if v2 < beta:
                    beta = v2
        else:
            v1 = (None, -np.inf)
            for m in self.__valid_moves():
                # alpha = max(alpha, minmax(board, m, depth-1, -player))
                self.__play(m, player)
                _, v2 = self.__minmax(m, depth - 1, -player, alpha, beta)
                self.__take_back(m)
                if v2 > v1[1]:
                    v1 = m, v2
                if v2 >= beta:
                    return v1
                if v2 > alpha:
                    alpha = v2

        return v1

    """
    Representation
    """

    def __repr__(self):
        """
        This function just prints the human-readable version of the board, i.e. with the downward oriented gravity
        """
        rep = ""
        b = np.rot90(self.__board)
        for i in range(self.__COLUMN_HEIGHT):
            for j in range(self.__NUM_COLUMNS):
                if b[i][j] == self.__player1:
                    rep += "| ● "
                elif b[i][j] == self.__player2:
                    rep += "| ○ "
                else:
                    rep += "|   "
            rep += '|\n'

        rep += "-"
        for i in range(self.__NUM_COLUMNS):
            rep += "----"
        rep += '\n'

        for i in range(self.__NUM_COLUMNS):
            rep += f"| {i + 1} "
        rep += '|'

        return rep
