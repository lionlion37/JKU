# Assignment 3 - Recursion

Complete the class `Maze` in `My_Maze.py`. Specification on most behavior is specified
in the regarding docstrings.

**Please do not forget to submit the `time.txt` file!**

## Basic Idea

Your algorithm is given a multi-line string representing a maze, such as

```
######
#    #
## # #
#  # #
# ## #
#### #
```

where `#` represent walls/obstacles and empty spaces represent walkable path. 
Given some starting point S, which must be on the walkable path, 
your algorithm should find and save all the exits of the given maze and mark,

- exits with an `X`,
- the starting point with `S`.
- visited points with `.`

The example provided would be marked as

```
######
#S...#
##.#.#
#..#.#
#.##.#
####X#
```

and its exit(s) would be `[(5, 4)]`. 



## Running and writing Tests

Basic test functionality is provided, yet make sure to validate inputs 
and think about edge-cases, which are not tested by the given test.

You can run the tests by executing `pytest` in the given working directory, 
which first needs to be downloaded.

## Further Remarks

Note that our choice of the
x-y plane is a bit peculiar here: 

This has to do with the fact that 
if we index into a `List[List[str]]` the normal way, i.e. x being horizontal and
the y pointing upwards, then it would look something like

`self._maze[len(self._maze) - y][x]`

With our approach, you should be able to index normally, like `self._maze[x][y]`.
This might also be confusing, but we opted for this to make the actual 
implementation less confusing.

Further, as always, keep the attribute- and method names unchanged, as this might otherwise 
break the tests used for grading.

However, it is highly encouraged to refactor your code into understandable chunks. 
Try not to repeat yourself and assume somebody else has to understand your code.

## Submitting

The deadline is specified in moodle. Submit a zip-file of all necessary source files, 
and submit the `time.txt` in a separate file. 

The submitted file should look like this:

```
├── k1234567.zip
│   └── (all necessary code files)
└── time.txt
```
