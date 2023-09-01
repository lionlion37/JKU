
class My_HeapPriorityQueue:

    def __init__(self):
        self.heap = list()
        self.size = 0

    def get_heap(self):
        """for testing purposes only
        """
        return self.heap

    def parent(self, index):
        return int((index - 1) / 2)

    def compare_children(self, index):
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

    def swap(self, index1, index2):
        tmp = self.heap[index2]
        self.heap[index2] = self.heap[index1]
        self.heap[index1] = tmp

    def up_heap(self, index):
        while self.heap[self.parent(index)] < self.heap[index] and index != 0:
            self.swap(self.parent(index), index)
            index = self.parent(index)

    def down_heap(self, index):
        compare_id = self.compare_children(index)
        while self.heap[compare_id] > self.heap[index]:
            self.swap(compare_id, index)
            index = compare_id
            compare_id = self.compare_children(index)
            # End of heap is reached
            if compare_id >= len(self.heap):
                break

    def insert(self, integer_val: int) -> None:
        """inserts integer_val into the max heap
        @param integer_val: the value to be inserted
        @raises ValueError if integer_val is None
        """
        if integer_val is None:
            raise ValueError("input is None")

        self.heap.append(integer_val)
        self.up_heap(len(self.heap) - 1)

    def is_empty(self) -> bool:
        """returns True if the max heap is empty, False otherwise
        @return True or False
        """
        return True if len(self.heap) == 0 else False

    def get_max(self) -> int:
        """returns the value of the maximum element of the PQ without removing it
        @return the maximum value of the PQ or None if no element exists
        """
        return self.heap[0] if not self.is_empty() else None

    def remove_max(self) -> int:
        """removes the maximum element from the PQ and returns its value
        @return the value of the removed element or None if no element exists
        """
        if not self.is_empty():
            removed_val = self.heap[0]
            self.heap[0] = self.heap[-1]
            _ = self.heap.pop()
            if len(self.heap) > 1:
                self.down_heap(0)
            return removed_val
        else:
            return None

    def get_size(self) -> int:
        """returns the number of elements in the PQ
        @return number of elements
        """
        return len(self.heap)
