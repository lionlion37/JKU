class RabinKarp:
    """
        Constructor initialising the modulo value, if any.
        @ param mod_val - Modulo value to be used for hashing, if provided.
    """

    def __init__(self, mod_val=None):
        self.mod_val = mod_val

    """
        This method uses the RabinKarp algorithm to search a given pattern in a given input text.
        @ param pattern - The string pattern that is searched in the text.
        @ param text - The text string in which the pattern is searched.
        @ return a list with the starting indices of pattern occurrences in the text, or None if not found.
        @ raises ValueError if pattern or text is None or empty.
    """

    def search(self, pattern, text):
        if not pattern or not text:
            raise ValueError("None or empty strings in pattern or text")

        # initiate
        m = len(pattern)
        n = 0
        indices = []
        hash_p = self.get_rolling_hash_value(pattern, '\0', 0)
        hash_t = self.get_rolling_hash_value(text[n:n+m], '\0', 0)

        while True:
            if hash_t == hash_p:  # Brute Force if hash values are equal
                if self.brute_force(pattern, text[n:n+m]):
                    indices.append(n)
            n += 1
            if n+m > len(text):  # appearance of another pattern not possible anymore
                break
            hash_t = self.get_rolling_hash_value(text[n:n+m], text[n-1], hash_t)  # rolling hash for next position
        return indices if indices else None

    """
         This method calculates the (rolling) hash code for a given character sequence. For the calculation use the base b=29.
         @ param sequence - The char sequence for which the (rolling) hash shall be computed.
         @ param lastCharacter - The character to be removed from the hash when a new character is added.
         @ param previousHash - The most recent hash value to be reused in the new hash value.
         @ return hash value for the given character sequence using base 29.
    """

    def get_rolling_hash_value(self, sequence, last_character, previous_hash):
        base = 29
        # hash without modulo division
        if not self.mod_val:
            if previous_hash == 0:
                hash = 0
                for n, c in enumerate(sequence):
                    hash += ord(c) * pow(base, len(sequence) - (n + 1))
                return hash
            else:
                return previous_hash * base - ord(last_character) * pow(base, len(sequence)) + ord(sequence[-1])
        # hash with modulo division
        else:
            if previous_hash == 0:
                hash = 0
                for n, c in enumerate(sequence):
                    hash += (ord(c) * pow(base, len(sequence) - (n + 1))) % self.mod_val
                return hash % self.mod_val
            else:
                return (previous_hash * base - (ord(last_character) * pow(base, len(sequence))) % self.mod_val + \
                        (ord(sequence[-1])) % self.mod_val) % self.mod_val

    def brute_force(self, pattern, text):
        for n in range(len(text)):
            if text[n:n+len(pattern)] == pattern:
                return True
        else:
            return False
