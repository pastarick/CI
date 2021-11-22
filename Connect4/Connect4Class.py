import numpy as np
from TreeNode import TreeNode


class Connect4:
    """
    This class encapsulates a connect-4 minmax solver (against user or possibly against itself)
    and a Monte Carlo Tree Search solver
    """
    _NUM_COLUMNS = 7
    _COLUMN_HEIGHT = 6
    _FOUR = 4

    _EVALUATION_GRID = {
        (0, 0, 0):    (0, 0),
        (0, 0, 1):    (1, 2),
        (0, 0, -1):   (1, 2),
        (0, 1, 0):    (1, 1),
        (0, 1, 1):    (2, 1),
        (0, 1, -1):   (1, 1),
        (0, -1, 0):   (1, 1),
        (0, -1, 1):   (1, 1),
        (0, -1, -1):  (2, 1),
        (1, 0, 0):    (1, 0),
        (1, 0, 1):    (2, 0),
        (1, 0, -1):   (1, 0),
        (1, 1, 0):    (2, 0),
        (1, 1, 1):    (3, 0),
        (1, 1, -1):   (2, 0),
        (1, -1, 0):   (1, 0),
        (1, -1, 1):   (1, 0),
        (1, -1, -1):  (1, 0),
        (-1, 0, 0):   (1, 0),
        (-1, 0, 1):   (1, 0),
        (-1, 0, -1):  (1, 0),
        (-1, 1, 0):   (1, 0),
        (-1, 1, 1):   (1, 0),
        (-1, 1, -1):  (1, 0),
        (-1, -1, 0):  (2, 0),
        (-1, -1, 1):  (2, 0),
        (-1, -1, -1): (3, 0),
    }
    
    _DIRECTIONS = {
        "tl": (-1, -1),
        "t":  (-1, 0),
        "tr": (-1, 1),
        # "r":  (0, 1),  # OBS: right direction should never be useful
        "dr": (1, 1),
        "d":  (1, 0),
        "dl": (1, -1),
        "l":  (0, -1),
    }

    _OPPOSITE_DIRECTIONS = {
        "tl": "dr",
        "t":  "d",
        "tr": "dl",
    }

    _DIRECTIONS_ARRAYS = {
        "tl": lambda i, j, board: np.flip(np.diagonal(board[i - 3:i, j - 3:j])),
        "t":  lambda i, j, board: np.flip(board[i - 3:i, j]),
        "tr": lambda i, j, board: np.diagonal(np.flip(board[i - 3:i, j + 1:j + 4], 0)),
        # "r":  lambda i, j, board: board[i, j + 1:j + 4],  # OBS: right direction should never be useful
        "dr": lambda i, j, board: np.diagonal(board[i + 1:i + 4, j + 1:j + 4]),
        "d":  lambda i, j, board: board[i + 1:i + 4, j],
        "dl": lambda i, j, board: np.diagonal(np.flip(board[i + 1:i + 4, j - 3:j], 1)),
        "l":  lambda i, j, board: np.flip(board[i, j - 3:j]),
    }

    def __init__(self, num_players):
        self._board = np.zeros((Connect4._NUM_COLUMNS, Connect4._COLUMN_HEIGHT), dtype=np.byte)
        self._player1 = 1
        self._player2 = -1
        self._num_players = num_players

    def _valid_moves(self):
        """Returns columns where a disc may be played"""
        return [n for n in range(Connect4._NUM_COLUMNS) if self._board[n, Connect4._COLUMN_HEIGHT - 1] == 0]

    def _play(self, column, player):
        """Updates `board` as `player` drops a disc in `column`"""
        (index,) = next((i for i, v in np.ndenumerate(self._board[column]) if v == 0))
        self._board[column, index] = player

    def _take_back(self, column):
        """Updates `board` removing top disc from `column`"""
        (index,) = [i for i, v in np.ndenumerate(self._board[column]) if v != 0][-1]
        self._board[column, index] = 0

    def _four_in_a_row(self, player):
        """Checks if `player` has a 4-piece line"""
        return (
                any(
                    np.all(self._board[c, r] == player)
                    for c in range(Connect4._NUM_COLUMNS)
                    for r in (list(range(n, n + Connect4._FOUR)) for n in range(Connect4._COLUMN_HEIGHT - Connect4._FOUR + 1))
                )
                or any(
            np.all(self._board[c, r] == player)
            for r in range(Connect4._COLUMN_HEIGHT)
            for c in (list(range(n, n + Connect4._FOUR)) for n in range(Connect4._NUM_COLUMNS - Connect4._FOUR + 1))
        )
                or any(
            np.all(self._board[diag] == player)
            for diag in (
                (range(ro, ro + Connect4._FOUR), range(co, co + Connect4._FOUR))
                for ro in range(0, Connect4._NUM_COLUMNS - Connect4._FOUR + 1)
                for co in range(0, Connect4._COLUMN_HEIGHT - Connect4._FOUR + 1)
            )
        )
                or any(
            np.all(self._board[diag] == player)
            for diag in (
                (range(ro, ro + Connect4._FOUR), range(co + Connect4._FOUR - 1, co - 1, -1))
                for ro in range(0, Connect4._NUM_COLUMNS - Connect4._FOUR + 1)
                for co in range(0, Connect4._COLUMN_HEIGHT - Connect4._FOUR + 1)
            )
        )
        )

    @staticmethod
    def cleaned_input():
        while True:
            s = input("Please chose the column of your move: ")
            if s.isnumeric():
                if 1 <= int(s) <= Connect4._NUM_COLUMNS:
                    return int(s)-1
                else:
                    print(f"Input column can only be an integer between 1 and {Connect4._NUM_COLUMNS}.")
            else:
                print(f"Input column can only be an integer between 1 and {Connect4._NUM_COLUMNS}.")

    def _is_terminal(self):
        return self._four_in_a_row(self._player1) \
               or self._four_in_a_row(self._player2) \
               or not self._valid_moves()

    @staticmethod
    def _valid_directions(tmp_board, i, j):
        return [d for d in Connect4._DIRECTIONS.keys()
                if tmp_board[i+Connect4._DIRECTIONS[d][0], j+Connect4._DIRECTIONS[d][1]] != -2]

    """
    Utility function section
    """

    @staticmethod
    def _count_vec(vec, player):
        # len(vec) is always 3
        vec = [v if v != -2 else 0 for v in vec]
        score, p = Connect4._EVALUATION_GRID[tuple(vec)]

        if vec[p] == player:
            score += 1
        elif vec[p] == -player:
            score += 0.5

        return score, p

    def _cell_value(self, cell):
        # OBS: for further optimization, I could also use always the padded grid
        tmp_board = np.pad(self._board, ((3, 3), (3, 3)), mode='constant', constant_values=(-2, -2))

        i = cell[0]
        j = cell[1]

        player = self._board[i, j]

        i += 3
        j += 3

        values = dict.fromkeys(Connect4._DIRECTIONS.keys(), (0, 0))

        for d in self._valid_directions(tmp_board, i, j):
            v = self._count_vec(Connect4._DIRECTIONS_ARRAYS[d](i, j, tmp_board), player)
            if v[0] == 4:
                return 25
            values[d] = v

        value = values["l"][0]
        for d, od in Connect4._OPPOSITE_DIRECTIONS.items():
            if player == values[d][1] == values[od][1]:
                if (values[d][0] + values[od][0] - 1) >= 4:
                    return 25
                else:
                    value += values[d][0] + values[od][0] - 1
            value += values[d][0] + values[od][0]

        return value

    """
    MinMax algorithm
    """

    def play_game_minmax(self):
        if self._num_players == 1:
            ai = self._player2
            player = ai
            ply = 3
            print(f"AI playing move {ply}...")
            self._play(ply, player)
            print(self)

            player = -player
            depth = 1
            turn = 0
            while True:
                turn += 1
                if player == ai:

                    ply = self._ai_move(depth)
                    print(f"AI playing column {ply+1}...")
                    self._play(ply, player)
                    print(self)
                    if self._four_in_a_row(player):
                        print(f"AI won the game.")
                        break
                else:
                    ply = Connect4.cleaned_input()
                    print(f"You played column {ply + 1}.")
                    self._play(ply, player)
                    print(self)
                    if self._four_in_a_row(player):
                        print(f"You won the game!")
                        break

                player = -player
                if (turn % 4 == 0) and depth < 4:
                    depth *= 2

            print("Terminating game")

    def _ai_move(self, depth):
        # in the first level of recursion, the move parameter of minmax won't be used
        return self._minmax(None, depth, self._player2, -np.inf, np.inf)[0]

    def _minmax(self, move, depth, player, alpha, beta):
        """
        MinMax algorithm implementation with alpha-beta pruning and hard cut-off to @param depth
        """
        if move is not None and (self._is_terminal() or depth == 0):
            (index,) = [i for i, v in np.ndenumerate(self._board[move]) if v != 0][-1]
            cell = (move, index)
            # if current player is -1, this means that the top-of-the-stack call was made by player 1
            # -> return score with opposite sign
            return move, -player * self._cell_value(cell)

        if player == self._player2:
            v1 = (None, np.inf)
            for m in self._valid_moves():
                self._play(m, player)
                _, v2 = self._minmax(m, depth - 1, -player, alpha, beta)
                self._take_back(m)
                if v2 < v1[1]:
                    v1 = m, v2
                if v2 <= alpha:
                    return v1
                if v2 < beta:
                    beta = v2
        else:
            v1 = (None, -np.inf)
            for m in self._valid_moves():
                self._play(m, player)
                _, v2 = self._minmax(m, depth - 1, -player, alpha, beta)
                self._take_back(m)
                if v2 > v1[1]:
                    v1 = m, v2
                if v2 >= beta:
                    return v1
                if v2 > alpha:
                    alpha = v2

        return v1

    """
    Monte Carlo Tree Search
    """

    def play_game_montecarlo(self):
        if self._num_players == 1:
            ai = self._player2
            player = ai
            root = TreeNode(player, parent=None, num_columns=Connect4._NUM_COLUMNS, column_height=Connect4._COLUMN_HEIGHT,
                            four=Connect4._FOUR, board=self._board)
            ply = 3
            print(f"AI playing move {ply}...")
            self._play(ply, player)
            print(self)

            player = -player
            ply = Connect4.cleaned_input()
            print(f"You played column {ply + 1}.")
            self._play(ply, player)
            print(self)
            player = -player

            while True:
                if player == -1:

                    ply = root.next_action()
                    print(f"AI playing column {ply + 1}...")
                    self._play(ply, player)
                    print(self)
                    if self._four_in_a_row(player):
                        print(f"AI won the game.")
                        break
                else:
                    ply = Connect4.cleaned_input()
                    print(f"You played column {ply + 1}.")
                    self._play(ply, player)
                    print(self)
                    if self._four_in_a_row(player):
                        print(f"You won the game!")
                        break

                root = root.trim(ply)
                player = -player

            print("Terminating game")

    """
    Representation
    """

    def __repr__(self):
        """
        This function just prints the human-readable version of the board, i.e. with the downward oriented gravity
        """
        rep = ""
        b = np.rot90(self._board)
        for i in range(Connect4._COLUMN_HEIGHT):
            for j in range(Connect4._NUM_COLUMNS):
                if b[i][j] == self._player1:
                    rep += "| ● "
                elif b[i][j] == self._player2:
                    rep += "| ○ "
                else:
                    rep += "|   "
            rep += '|\n'

        rep += "-"
        for i in range(Connect4._NUM_COLUMNS):
            rep += "----"
        rep += '\n'

        for i in range(Connect4._NUM_COLUMNS):
            rep += f"| {i + 1} "
        rep += '|'

        return rep
