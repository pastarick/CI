import numpy as np


NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
FOUR = 4


class TreeNode:

    _board = None
    _column_height = 0
    _num_columns = 0
    _four = 0

    def __init__(self, player, parent=None, num_columns=None, column_height=None, four=None, board=None):
        # one time static attributes initialization (just on root initialization)
        if num_columns:
            TreeNode._num_columns = num_columns
        if column_height:
            TreeNode._column_height = column_height
        if four:
            TreeNode._four = four
        if board is not None:
            TreeNode._board = board

        self.parent = parent
        self._visits = 0
        self._wins = 0
        self._children = {}
        # self._possible_moves = TreeNode._valid_moves()
        self._player = player

    @staticmethod
    def _valid_moves():
        """Returns columns where a disc may be played"""
        return [n for n in range(TreeNode._num_columns) if TreeNode._board[n, TreeNode._column_height - 1] == 0]

    @staticmethod
    def _play(column, player):
        """Updates `board` as `player` drops a disc in `column`"""
        (index,) = next((i for i, v in np.ndenumerate(TreeNode._board[column]) if v == 0))
        TreeNode._board[column, index] = player

    @staticmethod
    def _take_back(column):
        """Updates `board` removing top disc from `column`"""
        (index,) = [i for i, v in np.ndenumerate(TreeNode._board[column]) if v != 0][-1]
        TreeNode._board[column, index] = 0

    @staticmethod
    def _four_in_a_row(player):
        """Checks if `player` has a 4-piece line"""
        return (
                any(
                    np.all(TreeNode._board[c, r] == player)
                    for c in range(TreeNode._num_columns)
                    for r in (list(range(n, n + TreeNode._four)) for n in range(TreeNode._column_height - TreeNode._four + 1))
                )
                or any(
            np.all(TreeNode._board[c, r] == player)
            for r in range(TreeNode._column_height)
            for c in (list(range(n, n + TreeNode._four)) for n in range(TreeNode._num_columns - TreeNode._four + 1))
        )
                or any(
            np.all(TreeNode._board[diag] == player)
            for diag in (
                (range(ro, ro + TreeNode._four), range(co, co + TreeNode._four))
                for ro in range(0, TreeNode._num_columns - TreeNode._four + 1)
                for co in range(0, TreeNode._column_height - TreeNode._four + 1)
            )
        )
                or any(
            np.all(TreeNode._board[diag] == player)
            for diag in (
                (range(ro, ro + TreeNode._four), range(co + TreeNode._four - 1, co - 1, -1))
                for ro in range(0, TreeNode._num_columns - TreeNode._four + 1)
                for co in range(0, TreeNode._column_height - TreeNode._four + 1)
            )
        )
        )

    @staticmethod
    def _terminal(player):
        # if not self._possible_moves:  # terminal: draw
        if not TreeNode._valid_moves():
            return 0
        elif TreeNode._four_in_a_row(player):  # terminal: ‘player‘ wins
            return player
        else:  # not terminal: return sentinel value
            return None

    def _is_leaf(self):
        return not self._children

    def visits(self):
        return self._visits

    def wins(self):
        return self._wins

    def select_child(self):
        # in self._children there are only the children visited at least once
        c_param = 0.
        log_p = np.log(self._visits)
        rank = {c: (n.visits()-n.wins())/n.visits() + np.sqrt(c_param*log_p/n.visits()) for c, n in self._children.items()}
        return max(rank, key=rank.get)

    def tree_walk(self):
        # obs: with this algorithm, at every level either a node is leaf or all of its children have been simulated (at
        # least) once
        if not self._is_leaf():
            # use tree policy to go down until a leaf
            # choose best child: SELECTION phase
            child = self._children[self.select_child()]
            child.tree_walk()
        else:  # leaf: breadth exploration of next generation of nodes
            # obs: in theory, it's possible to have a situation in which current node arrives here, since it's leaf
            # of the explored tree, and the node is also leaf of the game tree (i.e. it doesn't have possible moves)
            r = self._terminal(self._player)
            if r is not None:
                # if there is a win/draw/loss
                self.backpropagate(r)

            for move in TreeNode._valid_moves():
                self._children[move] = TreeNode(-self._player, parent=self)
                TreeNode._play(move, self._player)
                # self._possible_moves = TreeNode._valid_moves()
                # don't need to store the returned value as the
                # function backpropagate already has updated counters until root
                self._children[move].random_walk()
                # after this call, ch is a discovered node and all its counters and its predecessors' counters
                # have been updated
                # backtrack last played move and try the next one (i.e. try a new child of current (leaf) node)
                TreeNode._take_back(move)
                # update possible moves pool to keep consistency
                # self._possible_moves = TreeNode._valid_moves()

        # don't need to return anything since at this point all of the children of a leaf of explored tree have been
        # explored

    def random_walk(self):
        """
        Default policy: for unexpanded nodes, go random until terminal state
        """
        # this method is called on newly created node: r is the value for this specific node in the tree
        # Enter this method after self._node_player played a move ->
        # -> first move played by this function is by -self._node_player
        history = []
        player = -self._player
        while True:
            # m = np.random.choice(self._possible_moves)
            m = np.random.choice(TreeNode._valid_moves())
            # insert in head for easy reverse looping
            history.insert(0, m)
            TreeNode._play(m, player)
            # self._possible_moves = TreeNode._valid_moves()
            r = self._terminal(player)
            if r is not None:
                break
            player = -player

        # backtrack every random move
        for move in history:
            TreeNode._take_back(move)

        # self._possible_moves = TreeNode._valid_moves()
        self.backpropagate(r)
        return r

    def backpropagate(self, result):
        self._visits += 1
        if result == self._player:
            self._wins += 1
        # elif result == 0:
        # self._wins += 0.5
        if self.parent:
            self.parent.backpropagate(result)

    # exposed method: called only on current root (i.e. stable state)
    def next_action(self):
        simulations = 50
        for _ in range(simulations):
            self.tree_walk()
            # at the end of tree_walk, all counters are up to date and the first generation of child nodes of the most
            # promising nodes have been explored

        ch_visits = {c: n.visits() for c, n in self._children.items()}
        # return max(ch_visits, key=ch_visits.get)
        return self.select_child()

    def trim(self, action):
        """
        This function returns the new root. In fact, at a certain AI turn, after the AI has played its move, it's
        useless to keep track of the untaken moves for that turn in the explored tree. The same is for the user's turn
        """
        # action is actually a child of self
        for c, n in self._children.items():
            if c != action:
                del n
            else:
                n.parent = None
        return self._children[action]


def print_board(board):
    """
    This function just prints the human-readable version of the board, i.e. with the downward oriented gravity
    """
    b = np.rot90(board)
    for i in range(COLUMN_HEIGHT):
        for j in range(NUM_COLUMNS):
            if b[i][j] == 1:
                print("| ● ", end='')
            elif b[i][j] == -1:
                print("| ○ ", end='')
            else:
                print("|   ", end='')
        print('|')

    print("-", end='')
    for i in range(NUM_COLUMNS):
        print("-" * 4, end='')
    print('')

    for i in range(NUM_COLUMNS):
        print(f"| {i + 1} ", end='')
    print('|')


def play(board, column, player):
    """Updates `board` as `player` drops a disc in `column`"""
    (index,) = next((i for i, v in np.ndenumerate(board[column]) if v == 0))
    board[column, index] = player


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


def main():
    # in main, every time AI plays its move and the user replies to him, root changes and becomes the new stable state
    # after user move
    # -> root's player is always AI
    board = np.zeros((NUM_COLUMNS, COLUMN_HEIGHT), dtype=np.byte)
    print_board(board)

    player = -1
    root = TreeNode(player, parent=None, num_columns=NUM_COLUMNS, column_height=COLUMN_HEIGHT, four=FOUR, board=board)

    while True:
        if player == -1:

            ply = root.next_action()
            print(f"AI playing column {ply + 1}...")
            play(board, ply, player)
            print_board(board)
            if four_in_a_row(board, player):
                print(f"AI won the game.")
                break
        else:
            ply = int(input("Please chose the column of your move: ")) - 1
            print(f"You played column {ply + 1}.")
            play(board, ply, player)
            print_board(board)
            if four_in_a_row(board, player):
                print(f"You won the game!")
                break

        root = root.trim(ply)
        player = -player

    print("Terminating game")


if __name__ == '__main__':
    np.random.seed(27)
    main()
