import sys
import time
from typing import Any, Generator, Tuple

from tree_node import TreeNode


class BinarySearchTree:
    """Binary-Search-Tree implemented for didactic reasons."""

    def __init__(self, root: TreeNode = None):
        """Initialize BinarySearchTree.

        Args:
            root (TreeNode, optional): Root of the BST. Defaults to None.
        
        Raises:
            ValueError: root is not a TreeNode or not None.
        """
        self._root = root
        self._size = 0 if root is None else 1

    def insert(self, key: int, value: Any) -> None:
        """Insert a new node into BST.

        Args:
            key (int): Key which is used for placing the value into the tree.
            value (Any): Value to insert.

        Raises:
            ValueError: If key is not an integer.
            KeyError: If key is already present in the tree.
        """
        if type(key) != int:
            raise ValueError("given key is not an integer")

        # Case 0: no element in tree
        if self._size == 0:
            self._root = TreeNode(key=key, value=value)
            self._size += 1

        # Case 1: some elements already in tree
        else:
            # search for key
            search_result = self.find_helper(key, self._root)

            # key was found
            if search_result[0] == 1:
                raise KeyError("key is already present in tree")

            # key not already present --> ready to insert
            else:
                insert_node = search_result[1]
                if key < insert_node.key:
                    insert_node.left = TreeNode(key=key, value=value, parent=insert_node)
                else:
                    insert_node.right = TreeNode(key=key, value=value, parent=insert_node)
                self._size += 1

    def find(self, key: int) -> TreeNode:
        """Return node with given key.

        Args:
            key (int): Key of node.

        Raises:
            ValueError: If key is not an integer.
            KeyError: If key is not present in the tree.

        Returns:
            TreeNode: Node
        """
        if type(key) != int:
            raise ValueError("given key is not an integer")

        # initiate helper function with root node
        found_node = self.find_helper(key=key, cur_node=self._root)
        # key is not found
        if found_node[0] == 0:
            raise KeyError("key is not present in tree")
        # key is found
        else:
            return found_node[1]

    def find_helper(self, key: int, cur_node, n_comparisons: int = 0):
        """
        Searches recursively for given key and counts number of needed comparisons

        Args
            key (int): Key which is used for placing the value into the tree.
            cur_node (TreeNode): current node
            n_comparisons (int): number of comparisons until now

        Return
            0 or 1: key was not found / found
            cur_node: last current node
            n_comparisons: total number of comparisons
        """
        n_comparisons += 2
        if key == cur_node.key:
            n_comparisons -= 1
            return 1, cur_node, n_comparisons  # 1 = found

        elif cur_node.is_external:
            return 0, cur_node, n_comparisons  # 0 = not found

        elif key < cur_node.key:
            if cur_node.left:
                return self.find_helper(key, cur_node.left, n_comparisons)
            else:
                return 0, cur_node, n_comparisons
        else:
            if cur_node.right:
                return self.find_helper(key, cur_node.right, n_comparisons)
            else:
                return 0, cur_node, n_comparisons


    @property
    def size(self) -> int:
        """Return number of nodes contained in the tree."""
        return self._size

    # If users instead call `len(tree)`, this makes it return the same as `tree.size`
    __len__ = size

    # This is what gets called when you call e.g. `tree[5]`
    def __getitem__(self, key: int) -> Any:
        """Return value of node with given key.

        Args:
            key (int): Key to look for.

        Raises:
            ValueError: If key is not an integer.
            KeyError: If key is not present in the tree.

        Returns:
            Any: [description]
        """
        return self.find(key).value

    def remove(self, key: int) -> None:
        """Remove node with given key, maintaining BST-properties.

        Args:
            key (int): Key of node which should be deleted.

        Raises:
            ValueError: If key is not an integer.
            KeyError: If key is not present in the tree.
        """
        if type(key) != int:
            raise ValueError("given key is not an integer")

        # search for key
        search_result = self.find_helper(key, self._root)

        if search_result[0] == 0:
            raise KeyError("key is not present in tree")
        # key is present in tree
        else:
            remove_node = search_result[1]

            # node to remove has only left child
            if remove_node.right is None and remove_node.left:
                if remove_node.parent:
                    # change child
                    remove_node.left.parent = remove_node.parent
                    # change parent
                    if remove_node.parent.left == remove_node:
                        remove_node.parent.left = remove_node.left
                    elif remove_node.parent.right == remove_node:
                        remove_node.parent.right = remove_node.left

            # node to remove has only right child
            elif remove_node.left is None and remove_node.right:
                if remove_node.parent:
                    # change child
                    remove_node.right.parent = remove_node.parent
                    # change parent
                    if remove_node.parent.left == remove_node:
                        remove_node.parent.left = remove_node.right
                    elif remove_node.parent.right == remove_node:
                        remove_node.parent.right = remove_node.right

            # node to remove has no children (is external)
            elif remove_node.is_external:
                if remove_node.parent:
                    # change parent
                    if remove_node.parent.left == remove_node:
                        remove_node.parent.left = None
                    elif remove_node.parent.right == remove_node:
                        remove_node.parent.right = None

            # node to remove has two children
            else:
                # search for next largest key-value
                next_found = False
                for next_node in self.inorder():
                    if next_found:
                        break
                    if next_node == remove_node:
                        next_found = True

                # remove node with next larger key-value
                self.remove(next_node.key)

                # replace key-value of node to remove with next larger key-value
                if remove_node.left:
                    remove_node.left.parent.key = next_node.key
                if remove_node.right:
                    remove_node.right.parent.key = next_node.key
                if remove_node.parent:
                    if remove_node.parent.left == remove_node:
                        remove_node.parent.left.key = next_node.key
                    elif remove_node.parent.right == remove_node:
                        remove_node.parent.right.key = next_node.key
       
    # Hint: The following 3 methods can be implemented recursively, and 
    # the keyword `yield from` might be extremely useful here:
    # http://simeonvisser.com/posts/python-3-using-yield-from-in-generators-part-1.html

    # Also, we use a small syntactic sugar here: 
    # https://www.pythoninformer.com/python-language/intermediate-python/short-circuit-evaluation/

    def inorder(self, node: TreeNode = None) -> Generator[TreeNode, None, None]:
        """Yield nodes in inorder."""
        node = node or self._root
        
        # This is needed in the case that there are no nodes.
        if not node: return iter(()) 

        if node.is_internal and node.left:
            yield from self.inorder(node.left)
        yield node
        if node.is_internal and node.right:
            yield from self.inorder(node.right)

    def preorder(self, node: TreeNode = None) -> Generator[TreeNode, None, None]:
        """Yield nodes in preorder."""
        node = node or self._root
        if not node: return iter(())

        yield node
        if node.is_internal and node.left:
            yield from self.preorder(node.left)
        if node.is_internal and node.right:
            yield from self.preorder(node.right)

    def postorder(self, node: TreeNode = None) -> Generator[TreeNode, None, None]:
        """Yield nodes in postorder."""
        node = node or self._root
        if not node: return iter(())

        if node.is_internal and node.left:
            yield from self.postorder(node.left)
        if node.is_internal and node.right:
            yield from self.postorder(node.right)
        yield node

    _preorder = preorder
    _inorder = inorder
    _postorder = postorder

    # this allows for e.g. `for node in tree`, or `list(tree)`.
    def __iter__(self) -> Generator[TreeNode, None, None]: 
        yield from self._preorder()

    @property
    def is_valid(self) -> bool:
        """Return if the tree fulfills BST-criteria."""
        return self.is_node_valid_helper(self._root, (-sys.maxsize-1), sys.maxsize)

    def is_node_valid_helper(self, node: TreeNode, mini: int, maxi: int):
        # empty BSTs are valid
        if node is None:
            return True

        # check size of key-value
        if node.key < mini:
            return False
        if node.key > maxi:
            return False

        return self.is_node_valid_helper(node.right, node.key, maxi) and self.is_node_valid_helper(node.left, mini, node.key)

    def return_min_key(self) -> TreeNode:
        """Return the node with the smallest key."""
        for last_node in self.inorder():
            break
        return last_node
           
    def find_comparison(self, key: int) -> Tuple[int, int]:
        """Create an inbuilt python list of BST values in preorder and compute the number of comparisons needed for finding the key both in the list and in the BST.
        
           Return the numbers of comparisons for both, the list and the BST
        """

        python_list = list(node.key for node in self._preorder())
        print(python_list)

        # n_comparisons of find() method
        n_bst = self.find_helper(key, self._root)[2]

        # n_comparisons of list() iteration search
        n_list = 0
        for el in python_list:
            n_list += 1
            if el == key:
                break

        return n_list, n_bst

    @property
    def height(self) -> int:
        """Return height of the tree."""
        h = 0
        for n in self.postorder():
            if n.is_external:
                h = max(h, self.depth(n))
        return h

    def depth(self, node: TreeNode):
        if node == self._root:
            return 0
        return self.depth(node.parent) + 1

    @property  
    def is_complete(self) -> bool:
        """Return if the tree is complete."""
        # max number of nodes (therefore number of nodes of complete BST) = 2^(height+1) - 1
        if self._size == pow(2, self.height+1) - 1:
            return True
        return False

    def __repr__(self) -> str:
        return f"BinarySearchTree({list(self._inorder())})"

    # You can of course add your own methods and/or functions!
    # (A method is within a class, a function outside of it.)

print("HH")