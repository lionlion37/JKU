from My_SingleLinkedList import My_SingleLinkedList
import time


def compare_lists(num: int):
    if type(num) != int:
        raise ValueError("Input is not a valid Integer!")

    # Test 1: My_singleLinkedList()
    t = []
    for n in range(3):
        my_list = My_SingleLinkedList()
        start_time = time.time_ns()
        for _ in range(num):
            my_list.prepend(0)
        t.append(time.time_ns() - start_time)
    time_my_list = sum(t) / 3

    # Test 2: Built-In Python list()
    t = []
    for n in range(3):
        py_list = list()
        start_time = time.time_ns()
        for _ in range(num):
            py_list.insert(0, 0)
        t.append(time.time_ns() - start_time)
    time_py_list = sum(t) / 3

    if time_my_list < time_py_list:
        winner = "My_SingleLinkedList"
    else:
        winner = "Python Built-In list"
    print(f"{num}   :  {winner} is     {round(abs((time_my_list-time_py_list) / 1000), 2)}us faster")

    return time_my_list, time_py_list


def test_compare_lists():
    nums = [1000, 10000, 100000, 200000, 300000]
    for num in nums:
        _,_ = compare_lists(num)
