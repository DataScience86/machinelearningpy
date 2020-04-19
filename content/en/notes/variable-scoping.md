---
author: "Mohit Sharma"
title:  "Scope of Variables"
date: "2020-01-08"
description: "A quick note with examples on scope of variables in Python."
tags:
- Python
- Global variable
- Local Variable

categories:
- Programming

featured_image: "notesfeatureimg/variablescope.png"
---

**Scope** in computer programming language means the visibility of variables. It is the scope that defines which part of the program can use/see the variable. In general, a variable, once defined, can be accessed from any part of the code. However, sometimes, we would like to restrict the use of the variable to a specific section of the code. Programmers may want to do this to avoid unexpected errors.

For example, you may like to limit the scope of a variable to a specific function. This way, any changes introduced to the function will not impact the whole code in case something goes wrong.

Variables are divided into two groups considering the above restriction criteria.

1. **Global** Variables
2. **Local** Variabls

## Global Variables
In Python, a **global** variable is defined outside the function. The variable can then be accessed outside or inside the function.

``` Python
var = 20
def demo():
  print(var)

demo()
# Ouptut :- 20
```
If you try to modify a global variable inside a function, Python will throw an error stating `UnboundLocalError` which indicates that the local variable was referenced before assignment.

``` Python
var = 20
def demo():
  var *= 20
  print(var)

demo()
# Output :- UnboundLocalError: local variable 'var' referenced before assignment
```
To modify the values, we need to declare the variable as using the `global` keyword.

``` Python
var = 20
def demo():
  global var
  var *= 20
  print(var)

demo()
# Ouptut :- 400
```

## Local Variables
A variable that can only be accessed inside a function is called as a `local` variable. In other words, variables that are changed or created inside of a function are considered local variables. When you try to access local variables from outside, they return the `NameError: name 'zz' is not defined`.

``` Python
def demo():
  zz = 20
  print(zz)

print(zz)
# Ouptut :- NameError: name 'zz' is not defined
```

Please leave comments, if
1. You find anything incorrect.
2. You want to add more information to the topic.
3. You wish to add another example to the topic.
4. You need more details in regards to a specific section.
5. You are unable to execute an example code.
