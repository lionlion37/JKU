import unittest

from RabinKarp import RabinKarp


class Example2StudentsUnitTests(unittest.TestCase):
    def test_kmp_short(self):
        r = RabinKarp()
        res = r.search("xxx", "abcdexxxunbxxxxke")
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
        r = RabinKarp()
        res = r.search("xxx", "xxxxxA")
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

        res = r.search("sdfadf", "bbbbbbb")
        self.assertTrue(res is None)

    def test_rabin_karp_hash_of_pattern(self):
        r = RabinKarp()
        str_pattern = "ef"
        hash_pattern = r.get_rolling_hash_value(str_pattern, '\0', 0)
        self.assertEqual(3031, hash_pattern)

    def test_rabin_karp_hash_of_text_sequences(self):
        r = RabinKarp()

        str_text = "abcdef"
        hash = 0

        hash = r.get_rolling_hash_value(str_text[0:2], '\0', 0)
        self.assertEqual(2911, hash)

        hash = r.get_rolling_hash_value(str_text[1:3], 'a', hash)
        self.assertEqual(2941, hash)

        hash = r.get_rolling_hash_value(str_text[2:4], 'b', hash)
        self.assertEqual(2971, hash)

        hash = r.get_rolling_hash_value(str_text[3:5], 'c', hash)
        self.assertEqual(3001, hash)

        hash = r.get_rolling_hash_value(str_text[4:6], 'd', hash)
        self.assertEqual(3031, hash)

    def test_modulo(self):
        r = RabinKarp(101)

        txt = "aaAaBcExxxxxxxdd.,.Ha"
        res = r.search("aAa", txt)
        self.assertTrue(len(res) == 1)
        self.assertEqual([1], res)

        res = r.search(".,.", txt)
        self.assertTrue(len(res) == 1)
        self.assertEqual([16], res)

        res = r.search("xx", txt)
        self.assertEqual(6, len(res))
        self.assertEqual([7, 8, 9, 10, 11, 12], res)