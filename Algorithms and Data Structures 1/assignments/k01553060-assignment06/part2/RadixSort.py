## Runtime Complexity O(2 * max_len * N) ##
# since every element has to be added to a bucket, each # distribution phase # takes N operations
# each # collection phase # also takes N operations
# distribution phase and collection phase both have to be repeated max_len times, where max_len is the maximum number of
# digits a list element has
# Therefore, overall we have 2*max_len*N operations to perform which leads to a complexity of O(2 * max_len * N)

class RadixSort:
    def __init__(self):
        self.base = 7

        self.buckets = []
        for _ in range(self.base):  # | ________list of buckets -> list of buckets as array here
            self.buckets.append([])

        self.bucket_list_history = []  # n [[[]]] -> will look like this in the end

    def get_bucket_list_history(self):
        return self.bucket_list_history

    def sort(self, list):
        """
        Sorts a given list using radixsort in ascending order
        @param list to be sorted
        @returns a sorted list
        @raises ValueError if the list is None
        """
        self.bucket_list_history.clear()  # clear history list at beginning of sorting

        if list is None:
            raise ValueError("given list is None")

        list, max_len = self.str_converter(list)  # convert elements of list to str to make indexing possible

        for l in range(max_len):
            self.clear_buckets()

            for element in list:
                if len(element) > l:
                    self.buckets[int(element[len(element) - l - 1])].append(element)
                else:
                    self.buckets[0].append(element)

            self._add_bucket_list_to_history(self.str_converter(self.buckets, to_int=True))
            list = [j for sub in self.buckets for j in sub]  # merge buckets

        return self.str_converter(list, to_int=True)

    def clear_buckets(self):
        self.buckets = []
        for _ in range(self.base):
            self.buckets.append([])

    def str_converter(self, list, to_int=False):
        """
        Converts int elements of a list to str elements or the other way around
        """
        if not to_int:
            str_list = []
            lengths = []
            for element in list:
                str_list.append(str(element))
                lengths.append(len(str(element)))
            return str_list, max(lengths)

        else:
            int_list = []
            for element in list:
                # convert nested list
                if type(element) == type(list):
                    int_list.append(self.str_converter(element, to_int=True))
                else:
                    int_list.append(int(element))
            return int_list

    def _add_bucket_list_to_history(self, bucket_list):
        """
        This method creates a snapshot (clone) of the bucketlist and adds it to the bucketlistHistory.
        @param bucket_list is your current bucketlist, after assigning all elements to be sorted to the buckets.
        """
        arr_clone = []
        for i in range(0, len(bucket_list)):
            arr_clone.append([])
            for j in bucket_list[i]:
                arr_clone[i].append(j)

        self.bucket_list_history.append(arr_clone)
