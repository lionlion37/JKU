from typing import Union

from My_ListNode import My_ListNode


class My_LinkedList:

    """A base class providing a single linked list representation."""

    # Do not modify this code section please!
    def __init__(self,new_head: Union[None, 'My_ListNode'] = None, new_tail: Union[None, 'My_ListNode']=None):
        """Create a list and default values are None."""
        self._header = new_head
        self._tail = new_tail
        self._size = int(bool(self._header)) + int(bool(self._tail))

    def _get_header(self) -> Union[None, 'My_ListNode']:
        return self._header

    def _get_tail(self) -> Union[None, 'My_ListNode']:
        return self._tail

    def get_size(self) -> int:
        """returns the number of nodes in the linked list in O(1)
        """
        return self._size

    def insert_ordered(self, integer_val: int) -> None:
        """Add the element `integer_val` to the list, keeping the list in descending order.

        Args:
            integer_val (int): Integer value to be added.

        Raises:
            ValueError: If integer_val is not an integer.
        """
        if type(integer_val) != int:
            raise ValueError("input_val is not an integer")

        # Case 0: list is empty
        if self._header is None:
            self._header = My_ListNode(data=integer_val)
            self._tail = self._header

        else:
            place_found = False
            inserted = False
            current_node = self._header

            # Case 1: input is larger than current header
            if self._header.get_data() < integer_val:
                tmp = My_ListNode(data=integer_val, next_node=self._header)
                self._header = tmp
                place_found = True
                inserted = True

            while not place_found:
                if current_node.get_next_node() is None:
                    place_found = True
                    current_node = None
                elif current_node.get_next_node().get_data() <= integer_val:
                    place_found = True
                else:
                    current_node = current_node.get_next_node()

            # Case 2: input is smaller than current tail
            if current_node is None and not inserted:
                tmp = My_ListNode(data=integer_val)
                self._tail.set_next_node(tmp)
                self._tail = tmp

            # Case 3: input is between current header and current tail
            elif not inserted:
                current_node.set_next_node(My_ListNode(data=integer_val, next_node=current_node.get_next_node()))

        self._size += 1

    def clear(self) -> None:
        """release the memory allocated for the list
        """
        while self._header:
            next_node = self._header.get_next_node()
            del self._header
            self._header = next_node
        self._size = 0
        del self._tail

    def remove_first(self) -> int:
        """removes the first node from the linked list and returns its value
        @return the value of the node that has been removed
        """
        removed_node = self._header
        if removed_node:
            self._header = self._header.get_next_node()
            return removed_node.get_data()
        else:
            return None

    def get_first(self) -> int:
        """returns the value of the first node in the linked list (without removing it)
        @return the value of the first node
        """
        return self._header.get_data() if self._header else None

    def contains(self, integer_val: int) -> bool:
        """returns true if integer_val is in the linked list; false otherwise
        @return True or False
        @raises ValueError if integer_val is None
        """
        if integer_val is None:
            raise ValueError("input is None")

        found = False
        finished = False
        current_node = self._header

        # Only check nodes if list is not empty
        if current_node:
            while not finished and not found:
                if current_node.get_data() == integer_val:
                    found = True
                if current_node == self._tail:
                    finished = True
                current_node = current_node.get_next_node()

        return found

    def to_list(self):
        """returns a python list representation of the linked list starting from _header
        @return a python list
        """
        my_list = list()
        current_node = self._header

        # Case 0: list is empty
        if current_node is None:
            return my_list

        # Case 1: list is not empty
        while True:
            my_list.append(current_node.get_data())
            if current_node == self._tail:
                break
            else:
                current_node = current_node.get_next_node()

        return my_list

    def to_string(self):
        """returns a comma-delimited string representation of the linked list:
        "[20]-> [8]-> [5]-> [1]"
        @return a string: "20,8,5,1"
        """
        if self._header:
            return str(self.to_list())[1:-1].replace(' ', '')
        else:
            return ''
