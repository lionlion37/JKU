import unittest
from MaxHeap import MaxHeap

test_array = [3, 9, 17, 2, 23, 1, 5, 4, 19, 17, 7, 18, 8, 67, 6, 11, 0]

class Example1StudentsUnitTests(unittest.TestCase):
    def test_bottom_up_heap_construction(self):
        ex = None
        try:
            mh = MaxHeap(None)
        except ValueError as ve:
            ex = ve

        self.assertTrue(isinstance(ex, ValueError))

        mh = MaxHeap(test_array)
        heap_arr = mh.get_heap()
        self.assertEqual(mh.remove_max(), 67)
        self.assertEqual(mh.remove_max(), 23)
        self.assertEqual(mh.remove_max(), 19)
        self.assertEqual(mh.remove_max(), 18)
        self.assertEqual(mh.remove_max(), 17)
        self.assertEqual(mh.remove_max(), 17)
        self.assertEqual(mh.remove_max(), 11)
        self.assertEqual(mh.remove_max(), 9)
        self.assertEqual(mh.remove_max(), 8)
        self.assertEqual(mh.remove_max(), 7)
        self.assertEqual(mh.remove_max(), 6)
        self.assertEqual(mh.remove_max(), 5)
        self.assertEqual(mh.remove_max(), 4)
        self.assertEqual(mh.remove_max(), 3)
        self.assertEqual(mh.remove_max(), 2)
        self.assertEqual(mh.remove_max(), 1)
        self.assertEqual(mh.remove_max(), 0)

    def test_contains(self):
        mh = MaxHeap(test_array)
        self.assertTrue(mh.contains(3))
        self.assertTrue(mh.contains(0))
        self.assertTrue(mh.contains(19))
        self.assertTrue(mh.contains(4))
        self.assertFalse(mh.contains(100))

    def test_sort(self):
        mh = MaxHeap(test_array)
        mh.sort()
        sorted_arr = mh.get_heap()
        self.assertEqual(sorted_arr[0], 0)
        self.assertEqual(sorted_arr[1], 1)
        self.assertEqual(sorted_arr[2], 2)
        self.assertEqual(sorted_arr[3], 3)
        self.assertEqual(sorted_arr[4], 4)
        self.assertEqual(sorted_arr[5], 5)
        self.assertEqual(sorted_arr[6], 6)
        self.assertEqual(sorted_arr[7], 7)
        self.assertEqual(sorted_arr[8], 8)
        self.assertEqual(sorted_arr[9], 9)
        self.assertEqual(sorted_arr[10], 11)
        self.assertEqual(sorted_arr[11], 17)
        self.assertEqual(sorted_arr[12], 17)
        self.assertEqual(sorted_arr[13], 18)
        self.assertEqual(sorted_arr[14], 19)
        self.assertEqual(sorted_arr[15], 23)
        self.assertEqual(sorted_arr[16], 67)
