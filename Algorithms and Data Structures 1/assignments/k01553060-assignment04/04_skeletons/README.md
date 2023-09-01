# Assignment 4 - Binary Search Trees

The goal of this assignment is to implement a fully functioning BST in Python. You will need to make sure that all methods/properties of the assignment work as the documentation describes them, while maintaining BST-structure.
All of them should be understandable by reading their documentation and method-signatures and if the theory behind BSTs is known.

In case something is unclear, first check if the question has been addressed in the forum, otherwise open your own thread there.

Some things, such as what makes a tree complete (for `tree.is_complete`), are not explained by the documentation but by the theory. Primarily, consult the slides or do your own research; if it is impossible for you to do the research yourself, ask in the forum.

## Remarks

### This might be difficult

This is a fully fletched data-structure, and maintaining the right references for all nodes at all times might turn out to be a tricky task. There are a few strategies to make it easier:

### Testing is essential

We provided a few very basic testcases for you, but it will probably be very useful to write your own. 

### Use the debugger

In [VSCode](https://code.visualstudio.com/docs/python/testing), 
[PyCharm](https://www.jetbrains.com/help/pycharm/rerunning-tests.html#debug-tests) 
and possibly other IDEs, it is possible to debug tests.

This is super handy, as it allows you to look inside the workings of your code without having to add `print`-statements everywhere. It is highly recommended to set up your debugger properly from the start and work with your tests.

### Keep it clean

This might be one of your first programming tasks that require you to manage some (code-)complexity. Refactoring, using short but informative names, documenting and commenting might help you vastly. You can both modify the node-class as well as the tree itself, hence be smart about where to add which functionality.

Sometimes, one does not need to verify all inputs of a method/function, if code within it already does that.

Be strategic about reusing code. As a very simple example, notice that `tree_node.is_external == not tree_node.is_internal`.

### Don't be scared of magic-methods

You might see peculiar methods, such as `__getitem__`, in this assignment. Those are called  magic-methods or dunder-methods and constitute to what people call the Python Data Model: https://docs.python.org/3/reference/datamodel.html

Understanding the Data Model is probably a little tricky at first, but vital to producing objects that behave as people expect it.

### "What am I allowed to use?"

This course is about understanding how to implement algorithms and data-structures; using something that quasi nullifies the challenge does not contribute to achieving that goal.

For example, just importing a BST from some downloaded module and then just writing a wrapper class around that does not really teach you anything. Hence, such uses of external code is not allowed.

However, clever uses of the standard library, such as using something from `functools` or `itertools` is not forbidden, but actually cool to see if done well.

Of course, this is a wobbly definition, but use your own judgement to decide whether using something constitutes to cheating or being clever. Also, please avoid using non-standard libraries, as tutors will have to download them.

## Submitting

The deadline is specified in moodle. Submit a zip-file of all necessary code files,  and submit the `time.txt` in a separate file. 

The submitted file should look like this:

```
├── k1234567.zip
│   └── (all necessary code files)
└── time.txt
```
