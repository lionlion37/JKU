import unittest
from RadixSort import RadixSort

test_array = [111, 32, 4, 50]


class Example1StudentsUnitTests(unittest.TestCase):
    def test_sort(self):
        rs = RadixSort()
        r = rs.sort(test_array)
        self.assertEqual(r, [4, 32, 50, 111])

    def test_sort_lists(self):
        rs = RadixSort()
        r = rs.sort(test_array)
        history = rs.get_bucket_list_history()
        self.print_bucket_list_history(history)

    def print_bucket_list_history(self, list):
        for i in range(0, len(list)):
            print("Iteration #" + str(i + 1))

            tmp = list[i]
            max_size = 0

            for j in range(0, len(tmp)):
                if len(tmp[j]) > max_size:
                    max_size = len(tmp[j])

            print("Bucket:", end='')

            for j in range(0, len(tmp)):
                print("\t" + str(j), end='')
            print("\n----------------------------------------------------------------------------------------\n",
                  end='')
            #print("\t", end='')
            for j in range(0, max_size):
                for k in range(0, len(tmp)):
                    if j < len(tmp[k]):
                        print("\t" + str(tmp[k][j]), end='')
                    else:
                        print("\t", end='')
                print("\n")
