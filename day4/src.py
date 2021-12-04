from pathlib import Path

import numpy as np


def load_data():
    root = Path(__file__).parent
    with open(root / "input.txt") as in_file:
        data = in_file.readlines()
    return data


class BingoBoard:
    def __init__(self, fields):
        self.fields = np.array(fields)
        self.marked = np.zeros((5, 5))

    def check(self):
        winning_row = np.any(np.all(self.marked == True, axis=0))
        winning_column = np.any(np.all(self.marked == True, axis=1))

        return winning_row or winning_column

    def mark(self, number):
        self.marked[self.fields == number] = True

    def score(self, number):
        return np.sum(self.fields[self.marked == False]) * number


def setup_game(data):
    boards = []
    board = []
    for row in data[2:]:
        if row == "\n":
            boards.append(BingoBoard(board))
            board = []
        else:
            board.append([int(elem) for elem in row.split()])
    rounds = [int(elem) for elem in data[0].split(",")]
    return rounds, boards


def task1():
    data = load_data()
    rounds, boards = setup_game(data)
    for round in rounds:
        for board in boards:
            board.mark(round)
        for board in boards:
            if board.check():
                print(board.score(round))
                return


def task2():
    data = load_data()
    rounds, boards = setup_game(data)
    for round in rounds:
        for board in boards:
            board.mark(round)
        for board in boards:
            if board.check():
                print(board.score(round))
                boards.remove(board)


if __name__ == "__main__":
    task1()
    task2()
