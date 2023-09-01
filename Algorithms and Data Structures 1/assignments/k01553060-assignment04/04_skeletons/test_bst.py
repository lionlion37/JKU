from random import seed, shuffle
from typing import List, Tuple

import pytest

from tree_node import TreeNode
from bst import BinarySearchTree

seed(8) # Change for different scenarios!

@pytest.fixture
def tuples() -> List[Tuple[int, str]]:
    words = "Hello this is a sentence and should be correct.".split()
    numbers = range(10)
    zipped = list(zip(numbers, words))
    shuffle(zipped)
    return zipped


@pytest.fixture
def tree(tuples) -> BinarySearchTree:
    tree = BinarySearchTree()

    for number, word in tuples:
        tree.insert(key=number, value=word)
    return tree

@pytest.fixture
def handcrafted() -> BinarySearchTree:
    tree = BinarySearchTree()
    
    root = TreeNode(1, "B")
    root.left = TreeNode(0, "A", parent=root)
    root.right = TreeNode(2, "C", parent=root)
    tree._root = root

    return tree


# This can be extended much more!
def test_is_valid(handcrafted):
    assert handcrafted.is_valid
    handcrafted._root.left.left = TreeNode(100, "I'm really huge and in the wrong place")
    assert not handcrafted.is_valid


def test_construction(tree):
    print(tree)
    return tree.is_valid


def test_removal(tree, tuples):
    for key, _ in tuples:
        tree.remove(key)
        assert tree.is_valid

print("HH")