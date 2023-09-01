from typing import Union

from My_ListNode import My_ListNode

class My_SingleLinkedList:

    """A base class providing a single linked list representation."""

    # Do not modify this code section please!
    def __init__(self,new_head: Union[None, 'My_ListNode'] = None, new_tail: Union[None, 'My_ListNode']=None):
        """Create a list and default values are None."""
        self._header = new_head
        self._tail = new_tail

    def _get_header(self) -> Union[None, 'My_ListNode']:
        return self._header

    def _get_tail(self) -> Union[None, 'My_ListNode']:
        return self._tail

    # Example 2: Modify this as specified!
    def prepend(self, integer_val: int) -> None:
        """Prepend integer to list. Not sorted.

        Args:
            integer_val (int): Integer to add

        Raises:
            ValueError: If `integer_val` is not an integer.
        """
        if type(integer_val) != int:
            raise ValueError("Input is not a valid Integer!")

        # Case 0: No Element in List yet
        if self._header is None:
            self._header = My_ListNode(data=integer_val)
            self._tail = self._header

        else:
            tmp = My_ListNode(data=integer_val, next_node=self._header)
            self._header = tmp
