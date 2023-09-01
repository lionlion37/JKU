from typing import List, Tuple

# Using constants might make this more readable.
START = 'S'
EXIT = 'X'
VISITED = '.'
OBSTACLE = '#'
PATH = ' '


class Maze:
    """Maze object, used for demonstrating recursive algorithms."""

    def __init__(self, maze_str: str):
        """Initialize Maze.

        Args:
            maze_str (str): Maze represented by a string, 
            where rows are separated by newlines (\\n).

        Raises:
            ValueError, if maze_str is invalid, i.e. if it is not the correct type, 
            if any of its dimensions is less than three, or if it contains 
            characters besides {'\\n', ' ', '#'}.
        """
        # We internally treat this as a List[List[str]], as it makes indexing easier.

        # Validate input-type
        if type(maze_str) != str:
            raise ValueError("Input has to be of type str!")

        self._maze = list(list(row) for row in maze_str.splitlines())

        # Validate input
        allowed_chars = ['\n', ' ', '#']
        for l in self._maze:
            # Check length of x-axis
            if len(l) < 3:
                raise ValueError("Input has one or more dimension(s) that are less than three")
            for e in l:
                # Check characters
                if e not in allowed_chars:
                    raise ValueError("input contains illegal character")
        # Check length of y-axis
        if len(self._maze) < 3:
            raise ValueError("input has one or more dimension(s) which are less than three")

        self._shape = [len(self._maze), len(self._maze[0])]  # [x (horizontal), y (vertical)]
        self._started = False
        self._start_is_exit = False

        self._exits: List[Tuple[int, int]] = []
        self._max_recursion_depth = 0

    def find_exits(self, start_x: int, start_y: int, depth: int = 0) -> None:
        """Find and save all exits into `self._exits` using recursion, save 
        the maximum recursion depth into 'self._max_recursion_depth' and mark the maze.

        An exit is an accessible from S empty cell on the outer rims of the maze.

        Args:
            start_x (int): x-coordinate to start from. 0 represents the topmost cell.
            start_y (int): y-coordinate to start from; 0 represents the leftmost cell.
            depth (int): Depth of current iteration.

        Raises:
            ValueError: If the starting position is out of range or not walkable path.
        """

        # Validate starting position
        try:
            s = self._maze[start_x][start_y]
        except IndexError:
            raise ValueError("starting position out of range")
        if s not in [PATH, EXIT]:
            raise ValueError("starting position is not a walkable path")

        # set current position to START if function is initiated for the first time; else: set position to VISITED
        if not self._started:
            self._maze[start_x][start_y] = START
            self._started = True
        elif s != EXIT:
            self._maze[start_x][start_y] = VISITED

        # Base Case
        if (start_x in [0, self._shape[0]-1] or start_y in [0, self._shape[1]-1]) and not self._start_is_exit:
            self._exits.append((start_x, start_y))
            self._maze[start_x][start_y] = EXIT
            # set _max_recursion_depth
            if depth > self._max_recursion_depth:
                self._max_recursion_depth = depth

            # Special Case START is an EXIT
            if depth == 0:
                # prevent from directly reentering Base Case with _start_is_exit
                self._start_is_exit = True
                # start again
                self.find_exits(start_x, start_y)

        # Recursion Case - check every neighbor and start function from there if walkable
        else:
            next_step_possible = False
            self._start_is_exit = False

            # try-except construction for the special case "START is an EXIT"
            try:
                if self._maze[start_x - 1][start_y - 1] == PATH:
                    self.find_exits(start_x-1, start_y-1, depth=depth+1)
                    next_step_possible = True
            except IndexError:
                None
            try:
                if self._maze[start_x - 1][start_y] == PATH:
                    self.find_exits(start_x-1, start_y, depth=depth+1)
                    next_step_possible = True
            except IndexError:
                None
            try:
                if self._maze[start_x - 1][start_y + 1] == PATH:
                    self.find_exits(start_x-1, start_y+1, depth=depth+1)
                    next_step_possible = True
            except IndexError:
                None
            try:
                if self._maze[start_x][start_y - 1] == PATH:
                    self.find_exits(start_x, start_y-1, depth=depth+1)
                    next_step_possible = True
            except IndexError:
                None
            try:
                if self._maze[start_x][start_y + 1] == PATH:
                    self.find_exits(start_x, start_y+1, depth=depth+1)
                    next_step_possible = True
            except IndexError:
                None
            try:
                if self._maze[start_x + 1][start_y - 1] == PATH:
                    self.find_exits(start_x+1, start_y-1, depth=depth+1)
                    next_step_possible = True
            except IndexError:
                None
            try:
                if self._maze[start_x + 1][start_y] == PATH:
                    self.find_exits(start_x+1, start_y, depth=depth+1)
                    next_step_possible = True
            except IndexError:
                None
            try:
                if self._maze[start_x + 1][start_y + 1] == PATH:
                    self.find_exits(start_x+1, start_y+1, depth=depth+1)
                    next_step_possible = True
            except IndexError:
                None

            # Set _max_recursion_depth if recursion terminates and no exit was found
            if not next_step_possible:
                if depth > self._max_recursion_depth:
                    self._max_recursion_depth = depth

    @property
    def exits(self) -> List[Tuple[int, int]]:
        """List of tuples of (x, y)-coordinates of currently found exits."""
        return self._exits

    @property
    def max_recursion_depth(self) -> int:
        """Return the maximum recursion depth after executing find_exits()."""
        return self._max_recursion_depth

    def __str__(self) -> str:
        return '\n'.join(''.join(row) for row in self._maze)

    __repr__ = __str__
