class MaxHeap:
    def __init__(self, list):
        """
        @param list from which the heap should be created
        @raises ValueError if list is None.
        Creates a bottom-up maxheap in place.
        """
        if list is None:
            raise ValueError("given list is None")

        self.heap = list
        self.size = len(list)

        start_id = self.size // 2 - 1  # index of last non-leave

        # down-heap all nodes from last non-leave to root
        for i in range(start_id, -1, -1):
            self.down_heap(i)

    def swap(self, index1, index2):
        tmp = self.heap[index2]
        self.heap[index2] = self.heap[index1]
        self.heap[index1] = tmp

    def compare_children(self, index):
        """
        :param index: index of element whose children should be compared
        :return: index of the smaller child
        """
        left_id = 2 * index + 2
        right_id = 2 * index + 1
        # only right child exists
        if left_id >= len(self.heap) > right_id:
            return right_id
        # only left child exists
        elif right_id >= len(self.heap) > left_id:
            return left_id
        # no child exists --> return length of heap
        elif right_id >= len(self.heap) and left_id >= len(self.heap):
            return len(self.heap)
        # left and right children exist
        elif right_id < len(self.heap) and left_id < len(self.heap):
            if self.heap[left_id] > self.heap[right_id]:
                return left_id
            else:
                return right_id

    def down_heap(self, index, break_point=None):
        if break_point is None:
            break_point = len(self.heap)
        compare_id = self.compare_children(index)
        if not (compare_id >= len(self.heap) or compare_id >= break_point):
            while self.heap[compare_id] > self.heap[index]:
                self.swap(compare_id, index)
                index = compare_id
                compare_id = self.compare_children(index)
                # End of heap is reached
                if compare_id >= len(self.heap) or compare_id >= break_point:
                    break

    def get_heap(self):
        # helper function for testing, do not change
        return self.heap

    def get_size(self):
        """
        @return size of the max heap
        """
        return self.size

    def contains(self, val):
        """
        @param val to check if it is contained in the max heap
        @return True if val is contained in the heap else False
        @raises ValueError if val is None.
        Tests if an item (val) is contained in the heap. Do not search the entire array sequentially, but use the properties of a heap
        """
        if val is None:
            raise ValueError("given value is None")

        return self.search(val, 0)

    def search(self, val, index):
        curr_el = self.heap[index]
        # value found
        if curr_el == val:
            return True
        # current node is smaller / lowest level is reached --> value not found
        elif curr_el < val or self.compare_children(index) >= self.size:
            return False
        else:
            # check if two children exist
            if 2*index+2 < self.size:
                return self.search(val, 2*index+2) or self.search(val, 2*index+1)
            else:
                return self.search(val, 2*index+1)

    def is_empty(self):
        """
        @return True if the heap is empty, False otherwise
        """
        return True if self.size == 0 else False

    def remove_max(self, break_point=None):
        """
        Removes and returns the maximum element of the heap
        @return maximum element of the heap or None if heap is empty
        """
        if not self.is_empty():
            removed_val = self.heap[0]
            if break_point is None:
                self.heap[0] = self.heap[-1]
            else:
                self.heap[0] = self.heap[break_point]
            if break_point is None:
                _ = self.heap.pop()
            if len(self.heap) > 1:
                self.down_heap(0, break_point)
            self.size -= 1
            return removed_val
        else:
            return None

    def sort(self):
        """
        This method sorts (ascending) the list in-place using HeapSort, e.g. [1,3,5,7,8,9]
        """
        num_el = len(self.heap)
        for n in range(num_el):
            # removes max while only minding elements until the id break_point
            curr_root = self.remove_max(break_point=num_el-n-1)
            self.heap[num_el-n-1] = curr_root
            self.size += 1
