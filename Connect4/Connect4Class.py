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

    f = lambda k: max(k)

    def __init__(self):
        self.__board = np.zeros((self.__NUM_COLUMNS, self.__COLUMN_HEIGHT), dtype=np.byte)
        self.__player1 = 1
        self.__player2 = -1

    def play_game(self, num_players):
        """
        Only exposed method, creates and handles a game
        """
        if num_players == 1:
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
                    print(f"AI playing column {ply + 1}...")
                    self.__play(ply, player)
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

        return k

    def __cell_value(self, cell):
        # OBS: IT MIGHT BE USEFUL TO ALWAYS HAVE THIS BOARD (REMEMBERING THE OFFSET FOR INDICES)
        tmp_board = np.pad(self.__board, ((3, 3), (3, 3)), mode='constant', constant_values=(0, 0))
        # with this padding, I avoid controls on corner values

        i = cell[0]
        j = cell[1]

        player = self.__board[i, j]

        i += 3
        j += 3

        value = -1
        base_value = self.__EVALUATION_TABLE[cell[0], cell[1]]

        # top-left
        vec = np.flip(np.diagonal(tmp_board[i - 3:i, j - 3:j]))
        value = max(value, self.__count_vec(vec, player))

        # top
        vec = np.flip(tmp_board[i - 3:i, j])
        value = max(value, self.__count_vec(vec, player))

        # top-right
        vec = np.diagonal(np.flip(tmp_board[i - 3:i, j + 1:j + 4]))
        value = max(value, self.__count_vec(vec, player))

        # right
        vec = tmp_board[i, j + 1:j + 4]
        value = max(value, self.__count_vec(vec, player))

        # down-right
        vec = np.diagonal(tmp_board[i + 1:i + 4, j + 1:j + 4])
        value = max(value, self.__count_vec(vec, player))

        # down
        vec = tmp_board[i + 1:i + 4, j]
        value = max(value, self.__count_vec(vec, player))

        # down-left
        vec = np.diagonal(np.flip(tmp_board[i + 1:i + 4, j - 3:j]))
        value = max(value, self.__count_vec(vec, player))

        # left
        vec = np.flip(tmp_board[i, j - 3:j])
        value = max(value, self.__count_vec(vec, player))

        return base_value + value

    """
    MinMax algorithm
    """

    def __ai_move(self):
        depth = 5
        # in the first level of recursion, the move parameter of minmax won't be used
        return self.__minmax(None, depth, self.__player2, -np.inf, np.inf)[0]

    def __minmax(self, move, depth, player, alpha, beta):
        """
        MinMax algorithm implementation with alpha-beta pruning and hard cut-off to @param depth
        """
        if move is not None and (self.__is_terminal() or depth == 0):
            (index,) = [i for i, v in np.ndenumerate(self.__board[move]) if v != 0][-1]
            cell = (move, index)
            return move, self.__cell_value(cell)  # int(player * utility(board, cell))

        # print(f"depth = {depth}")
        if player == self.__player2:
            v1 = (None, np.inf)
            for m in self.__valid_moves():
                # alpha = min(alpha, minmax(board, m, depth-1, -player))
                self.__play(m, player)
                v2 = self.__minmax(m, depth - 1, -player, alpha, beta)
                self.__take_back(m)
                if v2[1] < v1[1]:
                    v1 = v2
                if v2[1] <= alpha:
                    return v1
                if v2[1] < beta:
                    beta = v1[1]
        else:
            v1 = (None, -np.inf)
            for m in self.__valid_moves():
                # alpha = max(alpha, minmax(board, m, depth-1, -player))
                self.__play(m, player)
                v2 = self.__minmax(m, depth - 1, -player, alpha, beta)
                self.__take_back(m)
                if v2[1] > v1[1]:
                    v1 = v2
                if v2[1] >= beta:
                    return v1
                if v2[1] > alpha:
                    alpha = v2[1]

        return v1

    """
    Representation
    """

    def __repr__(self):
        """
        This function just prints the human-readable version of the board, i.e. with the downward oriented gravity
        """
        b = np.rot90(self.__board)
        for i in range(self.__COLUMN_HEIGHT):
            for j in range(self.__NUM_COLUMNS):
                if b[i][j] == self.__player1:
                    print("| ● ", end='')
                elif b[i][j] == self.__player2:
                    print("| ○ ", end='')
                else:
                    print("|   ", end='')
            print('|')

        print("-", end='')
        for i in range(self.__NUM_COLUMNS):
            print("-" * 4, end='')
        print('')

        for i in range(self.__NUM_COLUMNS):
            print(f"| {i + 1} ", end='')
        print('|')
