import unittest

from My_LinkedList import My_LinkedList
from My_ListPriorityQueue import My_ListPriorityQueue


class UnitTests(unittest.TestCase):

    '''Test My_LinkedList'''

    def test_clear_list(self):
        list = My_LinkedList()

        list.insert_ordered(2)
        list.insert_ordered(3)
        list.insert_ordered(1)
        list.insert_ordered(8)

        self.assertEqual(4, list.get_size())

        list.clear()
        self.assertEqual(0, list.get_size())

    def test_contains(self):
        list = My_LinkedList()

        list.insert_ordered(2)
        list.insert_ordered(3)
        list.insert_ordered(1)
        list.insert_ordered(8)
        list.insert_ordered(9)
        list.insert_ordered(10)
        list.insert_ordered(5)
        list.insert_ordered(3)
        list.insert_ordered(15)

        self.assertTrue(list.contains(2))
        self.assertTrue(list.contains(3))
        self.assertTrue(list.contains(1))
        self.assertTrue(list.contains(8))
        self.assertTrue(list.contains(9))
        self.assertTrue(list.contains(10))
        self.assertTrue(list.contains(15))
        self.assertTrue(list.contains(3))
        self.assertFalse(list.contains(0))
        self.assertFalse(list.contains(11))

        ex = None
        try:
            list.contains(None)
        except ValueError as ve:
            ex = ve
        self.assertTrue(isinstance(ex, ValueError))

    '''Test PQ'''

    def test_PQ_is_empty(self):
        pq = My_ListPriorityQueue()
        self.assertTrue(pq.is_empty())

    def test_pq_max(self):
        pq = My_ListPriorityQueue()
        pq.insert(20)
        self.assertEqual(20, pq.get_max())
        pq.insert(21)
        self.assertEqual(21, pq.get_max())
        pq.insert(2)
        self.assertEqual(21, pq.get_max())
        pq.insert(3)
        self.assertEqual(21, pq.get_max())
        pq.remove_max()
        self.assertEqual(20, pq.get_max())

    def test_pq_to_list(self):
        pq = My_ListPriorityQueue()
        pq.insert(2)
        pq.insert(3)
        pq.insert(1)
        pq.insert(8)
        pq.insert(19)

        a = pq.to_list()
        self.assertEqual(pq.get_max(), a[0])
        self.assertEqual(pq.get_size(), len(a))
