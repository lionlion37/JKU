import unittest

from KMP import KMP

test_array = [3, 9, 17, 2, 23, 1, 5, 4, 19, 17, 7, 18, 8, 67, 6, 11, 0]


class Example1StudentsUnitTests(unittest.TestCase):
    def test_kmp_failure_table(self):
        k = KMP()
        str_pattern = "ababac"
        failure_table = k.get_failure_table(str_pattern)

        self.assertEqual(0, failure_table[0])
        self.assertEqual(0, failure_table[1])
        self.assertEqual(1, failure_table[2])
        self.assertEqual(2, failure_table[3])
        self.assertEqual(3, failure_table[4])
        self.assertEqual(0, failure_table[5])

    def test_kmp_short(self):
        k = KMP()
        res = k.search("xxx", "abcdexxxunbxxxxke")
        print(res)
        print("\n")

        self.assertTrue(len(res) > 0)
        self.assertEqual(3, len(res))

        targets = [5, 11, 12]
        indices = res
        indices.sort()

        res_correct = True
        target_to_find = 0
        for i in range(0, len(indices)):
            if target_to_find >= len(targets):
                res_correct = False
            elif indices[i] == targets[target_to_find]:
                target_to_find += 1

        if target_to_find < len(targets):
            res_correct = False

        self.assertTrue(res_correct)

    def test_kmp_special(self):
        k = KMP()
        res = k.search("xxx", "xxxxxA")
        print(res)
        print("\n")

        self.assertTrue(len(res) > 0)
        self.assertEqual(3, len(res))

        targets = [0, 1, 2]
        indices = res
        indices.sort()

        res_correct = True
        target_to_find = 0
        for i in range(0, len(indices)):
            if target_to_find >= len(targets):
                res_correct = False
            elif indices[i] == targets[target_to_find]:
                target_to_find += 1

        if target_to_find < len(targets):
            res_correct = False

        self.assertTrue(res_correct)

    def test_advanced(self):
        k = KMP()
        res = k.search("xXxXx", "xXxABcdxXffxXxXxXxXxXxXasdf.,.")
        self.assertTrue(len(res) > 0)
        self.assertEqual(4,len(res))
        self.assertEqual([11, 13, 15, 17], res)

        res = k.search("Hi. Wie gehts dir heute, so.", "Hi.Wie.mir gehts   Hi wi, asgl ... Hi. Wie gehts dir heute, so. Asd. Hi, www....,")
        self.assertEqual(1, len(res))
        self.assertEqual([35], res)
