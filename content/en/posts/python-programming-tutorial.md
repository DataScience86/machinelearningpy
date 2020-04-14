---
author: "Mohit Sharma"
title:  "Python Programming Tutorial"
date: "2019-12-06"
description: "A brief introduction to Python programming for beginners with examples."
tags:
- Python
- Tutorial
- Python Beginners Tutorial

categories: 
- Python
- Programming

libraries:
- mathjax

featured_image: "postfeatureimg/python.png"
---

**Python** is a powerful and versatile programming language. It is easy to understand and learn. Today, the python programming language is widely used in the industry. Some of the applications of python programming include Web Development, Robotics, 3D CAD Applications, **Data Analysis, Face Detection, Machine Learning, and Artificial Intelligence**. In this tutorial, we will cover all the fundamental building blocks of Python!
<!--more-->

## History of Python
The programming language Python was developed by [Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum) in the late 1980's. However, the implementation of the language started to take place in 1989. The **version 1** of the language was released in January 1994. The primary functional tools included were `lambda`, `map`, `reduce`, and `filter`. In October 2000, version 2.0 of the programming language was released, and a year later **Python Software Foundation(PSF)**, a nonprofit organization, was also formed. The organization is devoted to advancing open source technology related to the Python programming language.

**In December 2008, Python 3.0, also sometimes called Py3K or Python 3000, was released to rectify fundamental design flaws of the earlier versions**. As full backward compatibility was not possible, a parallel world of Python 3 and Python 2 started to co-exist. This also meant that the Python 3 code would not run Python 2 and vice-a-versa. 

According to PSF, Starting January 1, 2020, the Python 2.x versions will no longer be supported. So, at machinelearningpy.com, we only use Python version 3 or simply Python 3. 
<!--more-->

## Things to keep in mind
There are a few things we should keep in mind while learning the Python programming language.

1. Python places a particular emphasis on spacing. So, **Spacing is essential**. 
2. Python is a **case sensitive** language. That means **python** and **Python** are two different objects.

## Operators in Python
In programming languages, operators are used for performing operations on values, variables, and objects. In Python, we can group these operators into seven groups.

- **Comparison Operators**
- **Arithmetic Operators**
- **Assignment Operators** 
- **Logical Operators**
- **Membership Operators**
- **Identity Operators**
- **Bitwise Operators**

### Comparison Operators Python
As the same suggests, the operators are used for comparing two values. These can be actual values or values stored inside variables like X and Y.

| Operator&nbsp;&nbsp;&nbsp;     | Name&nbsp;&nbsp;&nbsp;  |
| ---------- | --------- |
| == | Equal  |
| != | Not equal |
| > | Greater than |
| < | Less than |
| >= | Greater than or equal to |
| <= | Less than or equal To |

#### Example #1 - Comparison Operators Python
``` python
x = 10

x == 10
# Output :- True
x != 10
# Output :- True
x > 10
# Output :- False
x < 15
# Output :- True
x >+ 10
# Output :- True
x <= 15
# Output :- True
```

### Arithmetic Operators Python
These operators are used to perform common mathematical operations. 

| Operator&nbsp;&nbsp;&nbsp;     | Name&nbsp;&nbsp;&nbsp;  |
| ---------- | --------- |
| + | Addition  |
| - | Subtraction |
| / | Division |
| * | Multiplication |
| % | Modulus |
| ** | Exponential |
| // | Floor Division |

#### Example #1 - Arithmetic Operators Python
``` python
x = 10
y = 24
print(f"Addition: {x + y}")
# Output :- Addition: 34
print(f"Subtraction: {x - y}")
# Output :- Subtraction: -14
print(f"Division: {x / y}")
# Output :- Subtraction: 0.4166666666666667
print(f"Multiplication: {x * y}")
# Output :- Multiplication: 240
print(f"Modulus: {x % y}")
# Output :- Modulus: 10
print(f"Exponential: {x ** y}")
# Output :- Exponential: Exponential: 1000000000000000000000000
print(f"Floor Division: {x // y}")
# Output :- Floor Division: 0

```
#### Python Rules of Precedence
Python follows the same rules of precedence as mathematics for its arithmetic operators.

1. The highest precedence among arithmetic operators is given to Exponential.

2. The nest precedence is given to Multiplication and Division operators, which have the same priority.

3. Subtraction and Addition also have the same precedence but are executed only after Multiplication and Division.

> Operators with the same precedences are evaluated from left to right.

To force an expression to evaluate as per your want, use parentheses. **parentheses have the highest precedence**.

#### Example #2 - Arithmetic Operators Precendece
``` Python
print(f"Left to right rule: {3 - 2 + 1}")
# Output :- As per left to right rule: 2
print(f"forced calculations: {3 - (2 + 1)}")
# Output :- forced calculations: 0
```

### Assignment Operators Python
These operators are used for assigning values to variables.

| Operator&nbsp;&nbsp;&nbsp;     | Example&nbsp;&nbsp;&nbsp;  | Alternative&nbsp;&nbsp;&nbsp;                |
| ---------- | --------- | ----------------- |
| = | x = 3  | x = 3 |
| /= | x /= 3 | x = x / 3 |
| += | x += 3 | x = x + 3 |
| -= | x -= 3 | x = x - 3 |
| %= | x %= 3 | x = x % 3 |

#### Example #1 - Assignment Operators Precendece
``` Python
x = 10
print(f"x value is: {x}")
# Output :- x value is: 10
x += 15
print(f"Adding 15, x updated value: {x}")
# Output :- Adding 15, x updated value: 25
x %= 3
print(f"Remainder: {x}")
# Output :-  Remainder: 1
```

### Logical Operators Python
These operators are used to perform everyday mathematical operations. 

| Operator&nbsp;&nbsp;&nbsp; | Details&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                | Example&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          |
| ---------- | --------- | ----------------- |
| and  | Returns True if all statements are True | A > 10 and B < 5 |
| or  | Returns True if one of statements is True| A > 10 or B < 5 |
| not  | Return True if result was False; Reverses the result | not(A > 10 and B < 5) |

#### Example #1 - Logical Operators Python

``` Python
x = 10
y = 24
print(f"AND logic result: {x > 10 and y < 30}")
# Output :- AND logic result: False
print(f"OR logic result: {x < 10 or y < 30}")
# Output :- OR logic result: True
print(f"NOT logic result: {not(x < 10 or y < 30)}")
# Output :- NOT logic result: False
```

### Membership Operators Python
Python membership operators are used to check if a value or sequence is present in another object.

| Operator&nbsp;&nbsp;&nbsp;     | Details&nbsp;&nbsp;&nbsp;  |
| ---------- | --------- |
| in | True is returned if value is present  |
| not in | True is returned if value is not present |

#### Example #1 - Membership Operators Python

``` Python
x = [10, 20, 23, 40, 76, 11]
y = 24
print(f"in membership operator result: {y in x}")
# Output :- in membership operator result: False
print(f"not in logic result: {y not in x}")
# Output :- not in membership operator result: True
```

### Identity Operators Python
Identity operators are used for comparing two objects. They check if the objects/variables are exactly the same or not. The same entities share the same memory location.

| Operator&nbsp;&nbsp;&nbsp;     | Details&nbsp;&nbsp;&nbsp;  |
| ---------- | --------- |
| is | True is returned if objects are same  |
| is not | True is returned if objects are not same |

#### Example #1 - Identity Operators Python

``` Python
x = 24
y = 24
print(f"is membership operator result: {y is x}")
# Output :- is membership operator result: True
print(f"is not membership operator result: {y is not x}")
# Output :- is not membership operator result: False
```

### Bitwise Operators Python
In Python, one can use bitwise operators to compare the binary numbers. As a data scientist, I have not used these operators. However, if you are interested - [Click Here](https://www.journaldev.com/26737/python-bitwise-operators) 

## Data Types in Python 
Below is the list of data types python supports. Examples of each type are also provided. In this section, we will look at some of the important functions and methods which will help you identify the data type of a variable in Python.

1. **integer** (`int`) - Example: 1, 2, 3
2. **float** (`float`) - Example: 1.23, 0.23
3. **boolean** (`bool`) - Example: True, False
4. **string** (`str`) - Example: "Mohit" , "Python Champions"

### `type()` Check Python Data Type
``` Python
w = 14
print(type(w))
# Output :- int
x = "Machine Learning"
print(type(x))
# Output :- str
y = 3.14
print(type(y))
# Output :- float
z = True
print(type(z))
# Output :- bool
```
### Defining Strings in Python
To define a string in one single line one can use single `'` or double `"` quotes. 

``` Python
str_ex1 = "This is an exmple string"
str_ex2 = 'This is also an exmple string'
```
If your string contains one of these symbols as part of raw text then you can include those by using `\` in your string.

``` Python
str_ex3 = 'Bob's father is a builder.'
print(str_ex3)
# Output :- File "<ipython-input-47-73cfef63c547>", line 1
#    str_ex4 = 'Bob's father is a builder.'
#                  ^
#SyntaxError: invalid syntax

str_ex4 = 'Bob\'s father is a builder.'
print(str_ex4)
# Output :- Bob's father is a builder.
```

Another way around is to use double `"` if you have single `'` quote inside your string or vice-a-versa.

``` Python
str_ex5 = "Bob's father is a builder."
print(str_ex5)
# Output :- Bob's father is a builder.
```

### Arithmetic Functions on Strings
Some arithmetic functions can be applied on strings. For example - 
- `+` operator can be used to concatinate two strings. 
``` Python
str_ex6 = "Hello"
str_ex7 = "Reader"
print(str_ex6 + str_ex7)
# Output :- HelloReader
```
Notice no spacing is provided. We shall manually provide a blank space in order to make ki more readable.
``` Python
str_ex6 = "Hello"
str_ex7 = "Reader"
print(str_ex6 + ", " + str_ex7)
# Output :- Hello, Reader
```
- We can use `*` to replicate the string multiple number of times.
``` Python
str_ex8 = "Apple"
print(str_ex8 * 2)
# Output :- AppleApple
```

### `len()` get length of an object
The `len()` function in Python returns total length of an object. We can use it to get the number of character in a string. 
``` Python
str_ex8 = "Apple"
print(f"Total number of character in string are {len(str_ex8)}")
# Output :- Total number of character in string are 5
```
## Methods in Python
Until now we have seen three functions like `print()`, `type()` and `len()`. A function in Python uses parentheses and can accept one or more arguments. In the coming lessons, we will study them in detail.  

A **method** in Python is a built-in function. They are very similar to any other Python function and are called using dot notation. For example, upper() is a string method that can be used to convert all the characters of a string to upper case: `str_ex8.upper()` to get **'APPLE'**.

### Top 10 String Methods in Python
Some of the important strings methods are listed below:

| Method Name&nbsp;&nbsp;&nbsp;     | Details&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |
| ---------- | --------- |
| lower() | converts string to lower case  |
| islower() | Returns True if all characters are in lower case  |
| uper() | converts string to upper case  |
| isupper() | Returns True if all characters are in upper case  |
| strip() | Returns trimmed strings  |
| split() | Splits string by a specified separator, and returns a list |
| format() | Formats the string  |
| count() | Counts the numer of times a specified value occurs in a string  |
| endswith() | Returns True if string ends with specified value  |
| startswith() | Returns True if string starts with specified value  |

#### Example #1 - Top 10 String Methods Python

``` Python
str_method_ex1 = "This is an ARTICLE from machine learning py blog."
print("string to lower:{} ".format(str_method_ex1.lower()))
# Output :- string to lower:  this is an article from machine learning py blog.
print("Is string in lower:{} ".format(str_method_ex1.islower()))
# Output :- Checking if string in lower:False 
print("string to upper:{} ".format(str_method_ex1.upper()))
# Output :- string to upper:  THIS IS AN ARTICLE FROM MACHINE LEARNING PY BLOG.
str_method_ex2 = "ALL TEXT IN CAPS"
print("Is string in upper:{} ".format(str_method_ex2.isupper()))
# Output :- Is string in upper:True
str_method_ex3 = "I have trailing and leadning white spaces  "
print("Striping whitespace:{} ".format(str_method_ex3.strip()))
# Output :- Striping whitespace: I have trailing and leadning white spaces
print("splitting string by space:{} ".format(str_method_ex2.split(" ")))
# Output :- splitting string by space:['ALL', 'TEXT', 'IN', 'CAPS'] 
str_method_ex4 = "There are couple of apples and I ate some of these apples"
print("Count of string apples:{} ".format(str_method_ex4.count("apples")))
# Output :- Count of string apples:2 
print("Does string ends with apples:{} ".format(str_method_ex4.endswith("apples")))
# Output :- Does string ends with apples: True
print("Does string ends with apples:{} ".format(str_method_ex4.startswith("apples")))
# Output :- Does string ends with apples: False
```

## Data Structures in Python
**Data Structures** can be thought of as a container. These containers are required to organize and store different data types. These containers or structures can be defined based upon the dimension or based upon there ability to store different types of data. In Python, we have the following types of Data Structures.

1. **Lists**
2. **Tuples**
3. **Sets**
4. **Dictionaries**
5. **Compound Structure**

While you work on these data structures, you shall also keep in mind the following:

1. Is data structure Mutable?
2. Is data structure Ordered?

<br/>

**Mutability** here refers to whether you can change the elements of a data structure. If you can modify the data structure is called **Mutable** else, it is called **immutable**.

**Ordered** refers to whether you can access the elements stored in a data structure using their position. If you can, then the data structure is called as ordered. If not, it is called as unordered. 

### Lists in Python
A `list` is a fundamental data structure in Python. It is also one of the most commonly used data structures. A list can contain a mix of different data types. You can create a list using the square `[]` brackets. 

``` Python
list_ex1 = [12, 13.13, "Python", True]
```

`list_ex1` is a list containing four elements. All the elements in the list can be accessed using an index. The indexing value starts at 0, which is assigned to the first element. Alternatively, lists are also indexed from the end using -1.  Thus, you can pull out 12 by referring to the index 0 and last element True by referring to -1.

``` Python
list_ex1[0]
# Output :- 12
list_ex1[3]
# Output :- True
list_ex1[-1]
# Output :- True
list_ex1[len(list_ex1) - 1]
# Output :- True
```

#### Slicing and Dicing List
Slicing refers to the extraction of more than one value from a data structure. The below example shows how you can **extract more than one value from the list**. 

1. **Return first three elements**
``` Python
list_ex1[0:3]
# Output :- [12, 13.13, 'Python']
```
Notice, the **lower index is inclusive** and the **upper index is exclusive**.

2. **Return all elements starting specified index**
``` Python
list_ex1[2 : ]
# Output :- ['Python', True]
```

3. **Replace value at specified index**
``` Python
list_ex1[2] = 33
list_ex1
# Output :- [12, 13.13, 33, True]
```

4. **Replace more than one value in list**
``` Python
list_ex1[0,4] = [99, 120]
list_ex1
# Output :- [99, 13.13, 33, 120]
```

> A list is both **mutable** and **ordered**.

> A `string`, on the other hand, is **immutable**. You can not change a character in a string without creating a new string. 

``` Python
string_ex1 = "Strings are immutable"
string_ex1[0] = "G"
```

#### Top 11 List Methods in Python
Some of the important strings methods are listed below:

| Method Name&nbsp;&nbsp;&nbsp;     | Details&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |
| ---------- | --------- |
| apend() | Inserts an element at the end  |
| clear() | Deletes all the elements from the list  |
| insert() | Inserts element at a specific index  |
| pop() | Remove element from the specific place  |
| remove() | removes the first occurance of specified value in a list  |
| sort() | Sorts the list in arranging and descending order |
| reverse() | order of list is reserved  |
| copy() | creates and returns the copy of list  |
| index() | For a specified value returns the index  |
| extend() | joins the elements of a list, to the end of the current list  |
| join() | returns a string consisting of the list elements joined by a separator |

#### Example #1 - Top 11 String Methods Python

``` Python
ls_method_ex1 = [12, 34, 1, 15, 66, 54]
ls_method_ex1.append(99)
ls_method_ex1
# Output :- [12, 34, 1, 15, 66, 54]
ls_method_ex1.clear()
ls_method_ex1
# Output :- []
ls_method_ex2 = [12, 34, 1, 15, 66, 54]
ls_method_ex2.insert(2, 99)
ls_method_ex2
# Output :- [12, 34, 99, 1, 15, 66, 54]
ls_method_ex2.pop(2, 99)
ls_method_ex2
# Output :- [12, 34, 1, 15, 66, 54]
ls_method_ex3 = [12, 34, 1, 15, 66, 54, 12]
ls_method_ex3.remove(12)
ls_method_ex3
# Output :- [34, 1, 15, 66, 54, 12]
ls_method_ex3.sort()
ls_method_ex3
# Output :- [1, 12, 12, 15, 34, 54, 66]
ls_method_ex3.reverse()
ls_method_ex3
# Output :- [66, 54, 34, 15, 12, 12, 1]
ls_method_ex3.index(1)
ls_method_ex3
# Output :- 6
ls_method_ex3.extend([2,3,4])
ls_method_ex3
# Output :- [66, 54, 34, 15, 12, 12, 1, 2, 3, 4]
ls_method_ex4 = ["Machine", "Learning", "Python", "Programming", "Blog"]
(" + ").join(ls_method_ex4)
ls_method_ex4
# Output :- 'Machine + Learning + Python + Programming + Blog'
```

#### Example #2 - Top 4 Handy Functions for Lists in Python

1. `len()` - Returns the count of elements in a list.
``` Python
len([1,2,3,4,5,6,7,8,9,10])
# Output :- 10
```
2. `min()` - Returns the min value inside a list.
``` Python
min([1,2,3,4,5,6,7,8,9,10])
# Output :- 1
```
3. `max()` - Returns the maximum value inside a list.
``` Python
max([1,2,3,4,5,6,7,8,9,10])
# Output :- 10
```

4. `sorted()` - Returns the maximum value inside a list.
``` Python
sorted([11,2,32,14,59,61,7,18,90,10])
# Output :- [2, 7, 10, 11, 14, 18, 32, 59, 61, 90]
```

### Tuples in Python
**Tuple** is a useful data structure in Python. Unlike `list` they are **immutable** but are **ordered** container. That means once a **tuple** is defined, you cannot change the values of this data structure. **Sorting of elements inside tuples is not possible.** Often, you will use tuples to store a related piece of information. For example, you can use them to store country names and currency names, or you can use them to store information about latitude and longitude. 

1. **How to define a tuple** - A tuple can be defined using a **parentheses**. However, it not necessary to use parentheses.

``` Python
curreny = ("India", "INR")
```
2. **Extracting values from a tuple** 
``` Python
curreny[0]
# Output :- India
```
3. **Assigning multiple variable** - Tuples are very useful for assigning multiple variable in a compact way.
``` Python
length, width = 12, 4
print(f"The dimensions are {length} x {width}")
# Output :- The dimensions are 12 x 4
```
#### Tuple Methods in Python
| Method Name&nbsp;&nbsp;&nbsp;     | Details&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |
| ---------- | --------- |
| count() | returns the count of specific value  |
| index() | returns the index of specified value  |


#### Example #1 - Tuple Methods in Python

``` Python
tuple_method_ex1 = 12, 12, 1, 66, 66, 66
tuple_method_ex1.count(66)
# Output :- 3
tuple_method_ex2 = ("cat", "dog", "mice", "rabbit")
tuple_method_ex2.index("mice")
# Output :- 2
```

### Sets in Python
**Set** is a data structure that is mutable and unordered. Mostly, Python programmers use it swiftly remove the duplicates for a list. Others use it for set operations. For example, looking for differences, intersections, and unions are mostly done using sets.

1. ** Defining a Set in Python** - To define a **set** you can use `set` function.
``` Python
set_ex1 = set([1,1,2,3,3,4,4,5,6,9,9,9])
set_ex1
# Output :- {1, 2, 3, 4, 5, 6, 9}
```
> Some of the **list methods** can be used with **sets**. For example, you can add elements to the tuple using `add` method or you can use `pop` method to remove an element. However, it is not recommended to use `pop` as a random element is removed. Remember **sets are undordered**. 



#### Top 5 Set Methods in Python

| Method Name&nbsp;&nbsp;&nbsp;     | Details&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |
| ---------- | --------- |
| add() | Inserts an element |
| difference() | Provides a set with a difference between 2 or more sets  |
| update() | Updates the set with the union of other set  |
| intersection() | Provides a set with common elements between 2 or more sets  |
| union() | Returns a set containing union of sets. |

#### Example #1 - Top 5 Set Methods in Python

``` Python
set_ex2 = set([1,2,3,4,5,6,7,8,9,10])
set_ex2.add(11)
set_ex2
# Output :- {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
set_ex3 = set([3,5,7,8,10])
set_ex2.difference(set_ex3)
# Output :- {1, 2, 4, 6, 9, 11}
set_ex2.update(set([22,33,44]))
set_ex2
# Output :-  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22, 33, 44}
set_ex2.intersection(set_ex3)
# Output :- {3, 5, 7, 8, 10}
set_ex2.union(set_ex3)
# Output :- {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22, 33, 44}
```

### Dictionaries in Python
**Dictionaries** are interesting data structures. They store values by mapping them to a unique key. A **dictionary** is a mutable data structure. However, the keys of the data structure can be of any immutable type like tuples or integers. 

> Keys in the dictionary do not necessarily require to have the same type!

1. **Defining a dictionary in Python** - There are two ways to define a dictionary. One, you can use a `dict()` function, and second, you can use curly braces `{}`. 

``` Python
dict_ex1 = {"apple": 100, "bannana": 70, "orange": 120}
```

2. **Printing the values of a specific key** - You can use square `[]` brackets enclosing the key for which you need to access the value.
``` Python
dict_ex1["apple"]
# Output :-  100 
```
3. **Re-assign the values of a specific key** 
``` Python
dict_ex1["apple"] = 150
dict_ex1["apple"]
# Output :-  150
```

4. **Check if a key exists in dictionary** - You can use `in` identity 
``` Python
print("orange" in dict_ex1)
# Output :-  True
print("mango" in dict_ex1)
# Output :-  False
```

> Values associated with keys can be a single value, a list, a tuple, or could be a dictionary in itself.

#### Top 5 Dictionary Methods in Python

| Method Name&nbsp;&nbsp;&nbsp;     | Details&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |
| ---------- | --------- |
| get() | Returns the value associated with specified key |
| items() | Returns a list containing a tuple for each key value pair  |
| keys() | Returns keys from a dictionary  |
| values() | Returns all the values as list from dictionary  |
| fromkeys() | Returns a dictionary with specific key and value |

#### Example #1 - Top 5 Dictionary Methods in Python

``` Python
dict_ex2 = {"apple": 100, "banana": 70, "orange": 120, "mango": 200}
dict_ex2.get("mango")
# Output :- 200
dict_ex2.items()
# Output :- dict_items([('apple', 100), ('banana', 70), ('orange', 120), ('mango', 200)])
dict_ex2.keys()
# Output :-  dict_keys(['apple', 'banana', 'orange', 'mango'])
dict_ex2.values()
# Output :- dict_values([100, 70, 120, 200])
```
### Compound Structures
**Compound Strutures** come into existence when we include a data structure inside another data structure. For example, nested dictionaries which contains dictionary inside a dictionary fall under **compound structures**. 

``` Python
nutrition = {"apple": {"kcal": 52,
                         "protein": 0.83,
                         "carbs": 13.8,
                         "fat": 0.2},
              "banana": {"kcal": 89,
                         "protein": 1.1,
                         "carbs": 23,
                         "fat": 0.3}}
```

Elements inside the above-nested dictionary are accessed in the same fashion as we do for regular dictionaries.

``` Python
apple = nutrition["apple"]
apple
# Output :- {'carbs': 13.8, 'fat': 0.2, 'kcal': 52, 'protein': 0.83}
```
You can see that we got the inner dictionary and now we can get values for any of the keys. For example, protein content can be extracted using the following code.
``` Python
protein = apple["protein"]
protein
# Output :- 0.83
```
To access the values directly you can also follow the below code.
``` Python
nutrition["apple"]["protein"]
# Output :- 0.83
```
## Building Logic to your Python Code
To be a programmer is nothing more than guiding the steps your program should take when encountered with different values. It is like setting up the flow of your code, and that can only be achieved through clearly defining and building logic into your code. In this section, you will learn about tools that will help you make this logic work.

The logic building tools can be further divided into the following.

1. **Conditional Statements**
2. **While and For Loops**
3. **Break and Continue**

> Comparision and Logical statements are extensively used while building logic to your code. 

### Conditional Statements Python
As part of **Conditional Statements**, Python provides the following two options.

#### IF Statement
An `if` statement is executed if the condition is met. Otherwise, no action is taken. *Please pay close attention to the indentation*.

``` Python
x = 20
if x > 10:
  print(x)
```
**One line Representation** 
``` Python
x = 20
if x > 10: print(x)
```


#### IF, Elif, ELSE 
Sometimes you may also want to take action if the condition is not met(it's a good practice use else). At other times you may be interested in checking multiple conditions, and based on each one, a different action is required.

``` Python
x = 20
y = 30
if x > y:
  print(x/y)
elif x < y:
  print((x + y)/y)
else:
  print("The two numbers are equal")
```
**If.. Else. One line Representation**
``` Python
x = 20
y = 30
print(x/y) if x > y else print("The x is less than y.")
```

### While loop Python
**while** loop is an example of **indefinite iteration**, that means the loop will repeat an unknown number of times and will only stop when it meets a certain condition. For example, we will move the last ten elements from numbers to a new list until the count reaches 10.

``` Python
numbers = list(range(1, 20))
new_list = []

while len(new_list) <= 10:
  new_list.append(numbers.pop())
```

### For loop Python
**for** loop is an example of a **definitive iteration**. It is used to do something repeatedly using an **iterable**. An iterable is an object that can return one value at a time, such as a list, tuples, dictionaries, and files.

``` Python
country_names = ["United States of America", "Canada", "Mexico", "Spain", "Chile"]

for i in country_names:
  print(i)
```
``` Markdown
# Output
United States of America
Canada
Mexico
Spain
Chile
```

The `i` variable is called an iterable variable. You can name it whatever you like. It's not necessary to call it `i` always. A good practice is to give the same names to both the iterable and iteration variable.

#### Example #1 - For loop with tuples

``` Python
country_names = ("United States of America", "Canada", "Mexico", "Spain", "Chile")

for country in country_names:
  print(country)
```

#### Example #2 - For loops for iterating through dictionaries
for loops when used with dictionaries only return the keys.
``` Python
nutrition = {"apple": 52,
             "lemon": 0.83,
             "mango": 13.8,
             "egg": 0.2,
             "banana": 89,
             "milk": 1.1,
             "beacon": 23,
             "meat": 0.3}
                         
for food in nutrition:
  print(food)
```
``` Markdown
# Output
apple
lemon
mango
egg
banana
milk
beacon
meat
```

To iterate over both keys and values, you can use items() method as given below.

``` Python
nutrition = {"apple": 52,
             "lemon": 0.83,
             "mango": 13.8,
             "egg": 0.2,
             "banana": 89,
             "milk": 1.1,
             "beacon": 23,
             "meat": 0.3}
                         
for key, value in nutrition.items():
  print(f"Food Name: {key}, Nutrition Value: {value} ")
```
``` Markdown
# Output
Food Name: apple, Nutrition Value: 52 
Food Name: lemon, Nutrition Value: 0.83 
Food Name: mango, Nutrition Value: 13.8 
Food Name: egg, Nutrition Value: 0.2 
Food Name: banana, Nutrition Value: 89 
Food Name: milk, Nutrition Value: 1.1 
Food Name: beacon, Nutrition Value: 23 
Food Name: meat, Nutrition Value: 0.3 
```

### Break and Continue in Python
Many times your logic will require the iteration to end or skip a value when a specific condition is met. In such cases, keywords like `break` and `continue` will help you out. 
- `break` can terminate a loop as soon as a specific condition is met.

``` Python
for i in list(range(0, 20)):
  if i == 7:
    break
  print(i)
```
``` Markdown
# Output
0
1
2
3
4
5
6
```

- `continue` is used to skip an iterable value when a specific condition is met.

``` Python
for i in list(range(0, 10)):
  if i == 7:
    continue
  print(i)
```
``` Markdown
# Output
0
1
2
3
4
5
6
8
9
10
```
### Bonus - Zip and Enumerate Python
While working with loops you will find `zip` and `enumerate` very useful.

#### zip function in python
`zip` is used to combine multiple iterables into tuples. Each tuple contains the combination of elements by position. For example, We have two lists - 1. `names` as ["Bob", "Roxana", "Charlie", "Tango"]. 2. `weights` as [120, 90, 80, 100].

we can use `zip` function to return an iterator combining name and weights values like - [('Bob', 120), ('Roxana', 90), ('Charlie', 80), ('Tango', 100)].

``` Python
names = ["Bob", "Roxana", "Charlie", "Tango"]
weights = [120, 90, 80, 100]

for name, weight in zip(names, weights):
  print(f"Name is {name}, and Weight is {weight}")
```
``` Markdown
# Output
Name is Bob, and Weight is 120
Name is Roxana, and Weight is 90
Name is Charlie, and Weight is 80
Name is Tango, and Weight is 100
```

#### enumerate function in python
`enumerate` function is somewhat similar to `zip` function. However, instead of combining two different iterables, the function combines an iterable with the indices. 
``` Python
names = ["Bob", "Roxana", "Charlie", "Tango"]

for name in enumerate(names):
  print(name)
```
``` Markdown
# Output
(0, 'Bob')
(1, 'Roxana')
(2, 'Charlie')
(3, 'Tango')
```
## Functions in Python
A **function** is a way of wrapping a specific task related code into some container so that one can use it repeatedly without writing those multiple lines of code again and again.

For example - the below function returns the whole square of A + B using the following formula.

\(a + b\)^2 = a^2 + b^2 + 2ab

### Example #1 Defining Functions
``` Python
def whole_square(a = 1, b = 1):
  return((a*a) + (b*b) + 2*a*b)
  
whole_square(4, 6)
# Output :- 100
````
A function definition includes the following important parts. 

1. The **def** keyword - It indicates it is a function.
2. **Function name** - This is the name of the function. 
3. **Arguments** - A function can have as many arguments as possible. All arguments are passed as inputs to the function and are used with the function call. You need to mention them inside the *parentheses*.

Note- it is not necessary to pass arguments one can write a function without inputs.  

4. **Body** - The block of code after the `:` is referred to as a body. You can refer to the argument variables or can define new variables inside the body. 

5. **return** - The body often ends with the `return` statement. The keyword is followed by an expression that is evaluated to get the output. The function with no `return` statement returns `None`. 

### Example #2 Anonymous Functions Python
A **lambda expression** can be used to define an **anonymous function** in Python. This function is called anonymous because there is no name. You can use **lambda function** if you don't need them again and again. 

The above whole_square function can be reduced to lambda function as given below:

``` Python
whole_square = lambda x, y: x * x + y * y + 2*x*y
whole_square(4,6)
```
### Example #3 Documentation Strings in Python
To ensure that your function is readable, it is important that you provide documentation of your code. Documented code is easier to use and understand. Python provides docstrings specifically for documentation purposes. 

A docstring or document string is nothing more than a comment which is mentioned using Tripple `"""`.

``` Python
def whole_square(a = 1, b = 1):
  """Calculate the whole square of A + B.
  INPUT:
  a: int value
  b: int value
  OUTPUT:
  a^2 + b^2 + 2*a*b
  """
  return((a*a) + (b*b) + 2*a*b)
```

> You can also include examples into the document string.

## Errors and Exceptions in Python
As a programmer, one is likely to make mistakes. These mistakes in Python can be categorized as either **Errors** or **Exception**.

1. **Errors** - These occur when Python is not able to interpret your code due to **syntax** related issues. Mostly, typos are the reasons for such errors. For this reason, they are also referred to as **Syntax Errors**.

2. **Exceptions** - Even if your code is syntactically correct, some unexpected things can prevent your code from executing. All such cases fall under exceptions. Python provides a collection of different in-built exceptions. They help in understanding the underlying problem, which is preventing code from executing.

As Errors and Exceptions are not good, you need to understand how to handle them. 

### Handling Errors and Exceptions Python
To handle exceptions in Python, you can make use of the `try` statement. The `try` statement evaluates a block of code, and if it encounters an exception, it jumps to the `except` block. We can also mention what should happen if no exception is found in the `else` block. Finally, after executing the `try`, `else` or ` except` blocks, the code runs the final block called `finally`.

``` Python
x = int(12)
while True:
  try:
    if isinstance(x, int):
        print(x)
  except ValueError:
      print("x is not an integer")
  else:
      x += 1
      print(f"printing updated {x}") 
  finally:
      print("This is an example covering all the block")
```
``` Markdown
#Output
12
13
This is an example covering all the block
```
In the above example, the `try` statement is only going to look for ValueError and will ignore others. If you wish to catch multiple exceptions, then you can pick any of the below mentioned coding styles.

``` Python
x = int(12)
while True:
  try:
    if isinstance(x, int):
        print(x)
  except (ValueError, ZeroDivisionError):
      print("We found errors")
```

``` Python
x = int(12)
while True:
  try:
    if isinstance(x, int):
        print(x)
  except ValueError:
      print("We found Value Errors")
  except ZeroDivisionError:
      print("We found ZeroDivisionError")
```

### Capturing and Accessing Error Messages
Most exceptions come with the error messages. That means one can access these error messages and print them on the console for straightforward interpretation by users.

``` Python
x = 12
y = 0

try:
  z = x/y
except ZeroDivisionError as e:
  print("Error Occurred: {}".format(e))
# Output :- Error Occurred: division by zero
```

> Even if you don't have an idea about the kind of error, you can still capture the error message.

``` Python
x = 12
y = 0

try:
  z = x/y
except Exception as e:
  print("Error Occurred: {}".format(e))
# Output :- Error Occurred: division by zero
```

## Reading and Writing Files in Python
To read a file in Python, you need to open it by using the `open` built-in function. You can provide some arguments like read-only, write-only, or read and write. Once that is done, you can use a `read()` method to read the contents of the file.

Once finished with the file reading, you should close it using `close()` built-in function.

### Example #1 Reading Text File Python
``` Python
file = open("/data/MartinLutherKing.txt", 'r')
dream_speech = file.read()
file.close()
```

### Example #2 Writing Text File Python
The steps involved in writing the text file are also very similar.

1. You need to open the file in writing mode. If the file does not exist, don't worry, Python will create one for you. 
2. Use a writing mode to write the content to the file.
3. Close the connection.
``` Python
file = open("/data/speech_copy.txt", 'w')
file.write()
file.close()
```
### Example #3 Auto closing the opened files
You can use `with` function to auto-close the opened files after the reading or writing step is completed.

``` Python
with open('/data/MartinLutherKing.txt', 'r') as file:
    dream_speech = file.read()
```
## Loading Local and Third-Party Libraries
Python comes with many useful built-in functions and methods.  There are also a huge number of third-party libraries. These libraries need to be installed and loaded into your session to access the specific functions. 

You can also load local scripts into your current session. This is mostly required while working on larger projects. For larger projects, it is advised to split your code into multiple scripts to organize them better. For example, one can list all user-defined functions in one script. 

### Example #1 Importing Local Scripts Python
If the current script and the script you wish to load are in the same folder, then you can use `import` followed by the script name to load the contents of the script.

``` Python
import project_functions_list
```
If the script name is too big, as in the above case, we can also provide an alias to it by use `as`.

``` Python
import project_functions_list as pft
```

### Example #2 Importing Third Party libraries
There are thousands of third-party libraries written by individuals. To install these libraries, you can `pip` a package manager for Python 3. `pip` is the standard package manager for the Python, but it is not the only one. For example, Anaconda, which is specifically designed for`data science`, has its own package manager named `conda`. 

1. To install packages - `pip install package_name`
2. To install packages from Jupiter notebook - `! pip install package_name`.

### Example #3 Function Calling Using dot notation
To call a function from a package/module, you can use dot notation. For example, below, we call `sum()` function from `numpy` package in Python to get the total sum of elements in a list.

``` Python
import numpy as np
np.sum([22,123,65, 9, 12])
# Output : - 231
```
## Useful Online Resources

1. [Python Tutorial By Google](https://developers.google.com/edu/python): it is a free class offered by google for people with a little bit of programming experience.

2. [Official Python Tutorial](https://docs.python.org/3/tutorial/): The site provides examples, and is written using less technical language compared to main [Python](https://www.python.org/doc/) site.

3. [Third-Party Library Documentation](https://readthedocs.org/): Read the Docs is a huge resource that millions of developers rely on for software documentation of thrid parties. Easy to read and consume, I love this site.

### Best Practice:
It is best to define variables in the limited scope based upon the usage. It is rarely a good idea to define a function that can refer to a variable with a broader scope. As your code gets complicated, you may not remember or know what all variables you have defined.

Please leave comments, if 
1. You find anything incorrect.
2. You want to add more information to the topic.
3. You wish to add another example to the topic.
4. You need more details in regards to a specific section. 
5. You are unable to execute an example code.


