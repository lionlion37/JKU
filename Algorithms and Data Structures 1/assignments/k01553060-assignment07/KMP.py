class KMP:

    def __init__(self):
        pass
    
    """
        This method uses the KMP algorithm to search a given pattern in a given input text.
        @ param pattern - The string pattern that is searched in the text.
        @ param text - The text string in which the pattern is searched.
        @ return a list with the starting indices of pattern occurrences in the text, or None if not found.
        @ raises ValueError if pattern or text is None or empty.
    """

    def search(self, pattern, text):
        if not pattern or not text:
            raise ValueError("None or empty strings in pattern or text")

        # initiate
        f = self.get_failure_table(pattern)
        indices = []
        i, j = 0, 0
        n = len(text)
        m = len(pattern)

        while i < n:
            if pattern[j] == text[i]:
                if j == m-1:  # Match
                    indices.append(i-m+1)
                    i = i-m+2
                    j = 0
                i += 1
                j += 1
            elif j > 0:  # No match, but advancement
                j = f[j-1]
            else:  # Mismatch at first character of pattern
                i += 1

        return indices if indices else None
    """
        This method calculates and returns the failure table for a given pattern.
        @ param pattern - The string pattern for which the failure table shall be calculated.
        @ return a list with the failure table values for the given pattern.
    """

    def get_failure_table(self, pattern):
        # initate
        m = len(pattern)
        f = [0] * m
        i, j = 1, 0

        while i <= m-1:
            if pattern[i] == pattern[j]:  # j+1 characters have matched
                f[i] = j+1
                i += 1
                j += 1
            elif j > 0:  # j indices just after a prefix of pattern match
                j = f[j-1]
            else:  # no match
                f[i] = 0
                i += 1
        return f
