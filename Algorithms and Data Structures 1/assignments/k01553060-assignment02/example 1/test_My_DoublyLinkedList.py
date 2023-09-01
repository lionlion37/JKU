from My_DoublyLinkedList import My_DoublyLinkedList


class TestList:

    # requires that insert_ordered() and remove() works correctly
    def test_insert_ordered_remove(self):
        my_test_list = My_DoublyLinkedList()
        assert my_test_list._len_() == 0

        my_test_list.insert_ordered(1)
        assert my_test_list._len_() == 1

        my_test_list.insert_ordered(20)
        my_test_list.insert_ordered(13)
        assert my_test_list._len_() == 3

        my_test_list.insert_ordered(4)
        my_test_list.insert_ordered(9)
        assert my_test_list._len_() == 5

        my_test_list._remove(4)
        assert my_test_list._len_() == 4

    def test_list_is_empty(self):
        my_test_list = My_DoublyLinkedList()
        assert my_test_list.list_is_empty() == True

    def test_clear(self):
        my_test_list = My_DoublyLinkedList()

        assert my_test_list._get_header() is None
        assert my_test_list._get_tail() is None
        assert my_test_list._len_() == 0

        my_test_list.insert_ordered(2)
        my_test_list.insert_ordered(3)
        my_test_list.insert_ordered(1)
        assert my_test_list._len_() == 3

    def test_insert_ordered(self):
        my_test_list = My_DoublyLinkedList()

        # Insert 1 item
        my_test_list.insert_ordered(3)

        assert my_test_list._len_() == 1
        assert my_test_list._get_header().get_data() == 3
        assert my_test_list._get_tail().get_data() == 3

        # Insert several items
        my_test_list.insert_ordered(-1)
        my_test_list.insert_ordered(100)

        assert my_test_list._get_header().get_data() == 100
        assert my_test_list._get_header().get_next_node().get_data() == 3
        assert my_test_list._get_header().get_next_node().get_next_node().get_data() == -1
        assert my_test_list._get_tail().get_data() == -1
        assert my_test_list._get_header().get_next_node().get_next_node().get_next_node() is None

    def test_remove(self):
        my_test_list = My_DoublyLinkedList()

        # Remove from empty list
        assert my_test_list._remove(4) is False

        my_test_list.insert_ordered(3)
        my_test_list.insert_ordered(3)
        my_test_list.insert_ordered(10)
        my_test_list.insert_ordered(-2)
        my_test_list.insert_ordered(4)

        # Remove from non-empty list
        assert my_test_list._remove(4) is True
        assert my_test_list._remove(4) is False
        assert my_test_list._len_() == 4

        assert my_test_list._remove(10) is True
        assert my_test_list._get_header().get_data() == 3
        assert my_test_list._remove(3) is True
        assert my_test_list._get_tail().get_data() == -2
        assert my_test_list._remove(-2) is True
        assert my_test_list._len_() == 0
        assert my_test_list._get_header() is None
        assert my_test_list._get_tail() is None

    def test_remove_duplicates(self):
        my_test_list = My_DoublyLinkedList()

        my_test_list.insert_ordered(1)
        my_test_list.insert_ordered(1)
        my_test_list.insert_ordered(2)
        my_test_list.insert_ordered(5)
        my_test_list.insert_ordered(5)
        my_test_list.insert_ordered(5)
        my_test_list.insert_ordered(9)
        my_test_list.insert_ordered(10)
        my_test_list.insert_ordered(10)
        my_test_list.remove_duplicates()

        assert my_test_list._len_() == 5
        assert my_test_list._get_header().get_data() == 10
        assert my_test_list._get_tail().get_data() == 1

    def test_reorder_list(self):
        my_test_list = My_DoublyLinkedList()

        # Reorder empty list
        assert my_test_list.reorder_list() == -1

        my_test_list.insert_ordered(2)
        my_test_list.insert_ordered(1)
        my_test_list.insert_ordered(10)
        my_test_list.insert_ordered(-2)
        my_test_list.insert_ordered(-3)

        assert my_test_list.reorder_list() == 2
        assert my_test_list._get_header().get_data() == 1
        assert my_test_list._get_tail().get_data() == -2

        # Only even items
        my_test_list_even = My_DoublyLinkedList()
        my_test_list_even_copy = my_test_list_even
        my_test_list_even.insert_ordered(4)
        my_test_list_even.insert_ordered(8)
        my_test_list_even.insert_ordered(2)

        assert my_test_list_even.reorder_list() == 0
        assert my_test_list_even == my_test_list_even_copy

        # Only odd items
        my_test_list_odd = My_DoublyLinkedList()
        my_test_list_odd_copy = my_test_list_odd
        my_test_list_odd.insert_ordered(1)
        my_test_list_odd.insert_ordered(-3)
        my_test_list_odd.insert_ordered(11)

        assert my_test_list_odd.reorder_list() == -1
        assert my_test_list_odd == my_test_list_odd_copy

    def test_get_integer_value(self):
        my_test_list = My_DoublyLinkedList()

        e = False
        try:
            my_test_list.get_integer_value(1)
        except ValueError:
            e = True
        assert e
        
        my_test_list.insert_ordered(3)
        my_test_list.insert_ordered(1)

        e = False
        assert my_test_list.get_integer_value(1) == 1
        try:
            my_test_list.get_integer_value(3)
        except ValueError:
            e = True
        assert e
