from pathlib import Path

import pytest

from My_Maze import Maze

# This needs you to be in the right directory. Adjust the path as you see fit.
MAZES = Path('mazes')

# Fixtures are very useful. This returned maze can be used in any testcase 
# by making `maze` a parameter of the corresponding function.
@pytest.fixture
def maze() -> Maze:
    maze_str = (MAZES / 'example_maze.txt').read_text()
    return Maze(maze_str)


def test_find_exits(maze):
    maze.find_exits(1, 1)
    assert maze.exits == [(5, 4)]


def test_correct_marking(maze):
    maze.find_exits(1, 1)
    assert str(maze) == (MAZES / 'example_maze_solution.txt').read_text()


def test_save_rec_depth(maze):
    maze.find_exits(1, 1)
    assert maze.max_recursion_depth != 0


print("HHH")