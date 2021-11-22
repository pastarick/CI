from Connect4Class import Connect4


def main():
    game = Connect4(num_players=1)

    print(game)
    # game.play_game_montecarlo()
    game.play_game_minmax()


if __name__ == '__main__':
    main()
