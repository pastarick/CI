from Connect4Class import Connect4


def main():
    game = Connect4(num_players=1)

    print(game)
    game.play_game()


if __name__ == '__main__':
    main()
