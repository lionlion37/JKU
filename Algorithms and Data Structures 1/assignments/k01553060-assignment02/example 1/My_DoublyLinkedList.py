from typing import Union

from My_ListNode import My_ListNode


class My_DoublyLinkedList:
    """A base class providing a doubly linked list representation."""

    ## Do not change! ##

    def __init__(self, new_head: Union[None, 'My_ListNode'] = None, new_tail: Union[None, 'My_ListNode'] = None,
                 new_size=0):
        """Create a list and default values are None."""
        self._header = new_head
        self._tail = new_tail
        self._size = new_size

    def _len_(self) -> int:
        """Return the number of elements in the list."""
        return self._size

    def list_is_empty(self) -> bool:
        """Return True if list is empty."""
        return self._size == 0

    def _get_header(self) -> Union[None, 'My_ListNode']:
        return self._header

    def _get_tail(self) -> Union[None, 'My_ListNode']:
        return self._tail

    ## -- ##

    # EXAMPLE 1
    # The following methods are required for example 1.

    def insert_ordered(self, integer_val: int) -> None:
        """Add the element `integer_val` to the list, keeping the list in descending order.

        Args:
            integer_val (int): Integer value to be added.

        Raises:
            ValueError: If integer_val is not an integer.
        """
        if type(integer_val) != int:
            raise ValueError("Input is not a valid Integer!")

        # Case 0: No elements in list yet
        if self.list_is_empty():
            self._header = My_ListNode(data=integer_val)
            self._tail = self._header

        else:
            current_element = self._header
            for n in range(self._size):
                if current_element.get_data() <= integer_val:
                    # Case 1: integer_val is bigger than current header --> new header
                    if n == 0:
                        tmp = My_ListNode(data=integer_val, next_node=self._header)
                        self._header.set_prev_node(tmp)
                        self._header = tmp
                    # Case 2: integer_val neither has to be inserted at the very beginning or end of list
                    else:
                        tmp = My_ListNode(data=integer_val, prev_node=current_element.get_prev_node(),
                                          next_node=current_element)
                        current_element.get_prev_node().set_next_node(tmp)
                        current_element.set_prev_node(tmp)
                    break
                else:
                    current_element = current_element.get_next_node()
            # Case 3: integer_val is smaller than current tail --> new tail
            else:
                tmp = My_ListNode(data=integer_val, prev_node=self._tail)
                self._tail.set_next_node(tmp)
                self._tail = tmp

        self._size += 1

    def get_integer_value(self, index: int) -> int:
        """Return the value (data) at position `index`, without removing the node.

        Args:
            index (int): 0 <= index < Length of list

        Returns:
            (integer): Retrieved value.

        Raises:
            ValueError: If the passed index is not an integer or out of range.
        """
        if index >= self._size or index <= 0:
            raise ValueError("Index out of range!")
        if type(index) != int:
            raise ValueError("Input is not a valid integer!")
        if self.list_is_empty():
            raise ValueError("List is empty, no elements to retrieve!")

        # Circle through elements -index- times
        current_element = self._header
        for n in range(index):
            current_element = current_element.get_next_node()

        return current_element.get_data()

    def _remove(self, integer_val: int) -> bool:
        """Remove all occurences of given integer value `integer_val`.

         Args:
             integer_val (int): Value to remove.

         Returns:
             (bool): Whether an element was successfully removed or not.

         Raises:
             ValueError: If integer_val is not an integer.
        """
        if type(integer_val) != int:
            raise ValueError("Input is not a valid Integer!")

        removed = False
        # Circle through elements
        current_element = self._header
        for n in range(self._size):
            # Input Value found
            if current_element.get_data() == integer_val:
                removed = True
                # Case 0: Only 1 element in whole list
                if self._size == 1:
                    self._header = None
                    self._tail = None
                # Case 1: header matches integer_val
                elif current_element.get_prev_node() is None:
                    current_element.get_next_node().set_prev_node(None)
                    self._header = current_element.get_next_node()
                # Case 3: tail matches integer_val
                elif current_element.get_next_node() is None:
                    current_element.get_prev_node().set_next_node(None)
                    self._tail = current_element.get_prev_node()
                # Case 2: neither header or tail matches integer_val but any other node
                else:
                    current_element.get_prev_node().set_next_node(current_element.get_next_node())
                    current_element.get_next_node().set_prev_node(current_element.get_prev_node())
                self._size -= 1
            elif removed:
                break
            current_element = current_element.get_next_node()

        return removed

    def remove_duplicates(self) -> None:
        """Remove all duplicates from the list such that the remaining elements are all unique.

        Example:
            [3, 3, 2, 2, 1, 1] -> [3, 2, 1]
        """
        # Circle thorugh elements
        current_element = self._header
        for n in range(self._size):
            if n >= self._size - 1:
                break
            if current_element.get_data() == current_element.get_next_node().get_data():
                duplicate = current_element.get_data()
                # Get next distinct element
                current_element = current_element.get_next_node().get_next_node()
                try:
                    while current_element.get_next_node().get_data() == duplicate:
                        current_element = current_element.get_next_node()
                    current_element = current_element.get_next_node()
                except:
                    None
                # Remove duplicates
                self._remove(duplicate)
                self.insert_ordered(duplicate)
            else:
                current_element = current_element.get_next_node()

    def reorder_list(self) -> int:
        """Reorder list such that all odd numbers are in the front and the even numbers all 
        occur later than the odd numbers, while maintaining odd and even numbers ordered.
        
        Example: [7, 6, 4, 1] -> [7, 1, 6, 4].

        Returns:
            (int): index of first occuring even number. If there are no even numbers, return -1.
        """
        current_element = self._header
        tmp_odd = My_DoublyLinkedList()
        tmp_even = My_DoublyLinkedList()
        for n in range(self._size):
            # Get all even elements in one My_List
            if current_element.get_data() % 2 == 0:
                tmp_even.insert_ordered(current_element.get_data())
            # Get all odd elements in one My_List
            else:
                tmp_odd.insert_ordered(current_element.get_data())
            current_element = current_element.get_next_node()

        # whole list is empty or no even elements
        if self.list_is_empty() or tmp_even.list_is_empty():
            return -1
        # Only even elements
        elif tmp_odd.list_is_empty():
            return 0

        # Concatenate even list and odd list
        tmp_odd._tail.set_next_node(tmp_even._header)
        tmp_even._header.set_prev_node(tmp_odd._tail)
        self._header = tmp_odd._header
        self._tail = tmp_even._tail

        return self._size - tmp_even._size
