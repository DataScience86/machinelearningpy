---
author: "Mohit Sharma"
title:  "Python - deep copy vs shallow copy"
date: "2020-04-19"
description: "Learn the difference between shallow copy, deep copy and normal assignment operator in python."
tags:
- Python
- deep copy
- copy
- shallow copy

categories:
- Python
- Programming

featured_image: "notesfeatureimg/copy.png"
---
## How to copy an object in Python
If you are new to programming, then there is a fair chance that you may not have come across a logical scenario that requires one to use a deep copy of an object instead of a shallow copy. Before we dive into when to use what and why let's look at an example on assignment operator. You need to understand what exactly happens when we use an assignment `=` operator in Python.

```
# Creating a list object
ls = [[9, 18, 27], [2, 4, 6], [3, 6, 9]]

# Print list - ls
print(f"Original list: {ls}")
print(f"Original object id: "id(ls)" )

# Duplicate copy using assignment `=` operator
dup_copy = ls

# Print list - dup_copy
print(f"Duplicate copy of list: {dup_copy}")
print(f"Duplicate object id: "id(dup_copy)" )
```

{{< boxmd >}}
**Output**
Original list: [[9, 18, 27], [2, 4, 6], [3, 6, 9]]
Original object id: 1897609773640
Duplicate copy of list: [[9, 18, 27], [2, 4, 6], [3, 6, 9]]
Duplicate object id: 1897609773640
{{< /boxmd >}}

As you can see from the output that both objects have the same object id as `1897609773640`. In Python, if you use `=` operator it creates a new variable that shares the same reference `id` as the original object. This essentially means that modifications done to one will be reflected in the other.
**Let's look at an example:**

```
# Creating a list object
ls = [[9, 18, 27], [2, 4, 6], [3, 6, 9]]

# Create a duplicate copy using `=`
dup_copy = ls

# Modify original list - ls by appending new list
ls.append([4, 8, 12])

print(f"Original copy of list: {ls}")
print(f"Duplicate copy of list: {dup_copy}")
```
{{< boxmd >}}
**Output**
Original copy of list: [[9, 18, 27], [2, 4, 6], [3, 6, 9], [4, 8, 12]]
Duplicate copy of list: [[9, 18, 27], [2, 4, 6], [3, 6, 9], [4, 8, 12]]
{{< /boxmd >}}

{{< alert theme="warning" >}} So, what should you do? If you wish to create a copy of an object to preserve the original values and only want to modify the new values. {{< /alert >}}  

In Python this can be achieved in two different ways:
1. **Shallow Copy**
2. **Deep Copy**

To get the shallow or deep copy, we will be using functions from the `copy` module in Python. You can import that library using the `import copy` statement in Python. In the below example, we will create both shallow and deep copies of the list object and check their object id.

```
import copy
ls = [[9, 18, 27], [2, 4, 6], [3, 6, 9]]
soft_copy = copy.copy(ls)
deep_copy = copy.deepcopy(ls)

print(f"Original list id: {id(ls)}")
print(f"soft_copy id: {id(soft_copy)}")
print(f"deep_copy id: {id(deep_copy)}")
```
{{< boxmd >}}
**Output**
Original list id: 1897609958600
soft_copy id: 1897609959624
deep_copy id: 1897609958984
{{< /boxmd >}}

As you can see, that object id's of all the three objects are different. That means they are not just merely references of each other. They are three different objects. Let us learn and understand how they are created from the original object.

## What is a deep copy in python
A **deep copy** of an object is **created recursively**. It means, first, the new collection object is created and then **each element of the original object is recursively populated** in the new object. In practice, it means that any **changes made to the original or to the copy do not impact each other**.

### Deep Copy Example 1
```
# Modify the deep copy
deep_copy[2][2] = 999

print(f"Original : {ls}")
print(f"deep_copy : {deep_copy}")
```
{{< boxmd >}}
**Output**
Original : [[9, 18, 27], [2, 4, 6], [3, 6, 9]]
deep_copy : [[9, 18, 27], [2, 4, 6], [3, 6, 999]]
{{< /boxmd >}}

## What is a shallow copy in python
A **shallow copy** of an object is created by constructing a new object, just like a deep copy. However, the **elements of the object are not copied recursively**. Instead, they are **populated through references to the items** found in the original object. It means any **changes made to the original elements will be reflected in both**. However, **addition of new elements** will only be **reflected in the shallow copy and not the original** object.

### Shallow Copy Example 1
In this example, you will see how changing original elements affects both the objects.

```
# Modify the soft_copy
soft_copy[2][2] = 99

print(f"Original : {ls}")
print(f"soft_copy : {soft_copy}")
```
{{< boxmd >}}
**Output**
Original : [[9, 18, 27], [2, 4, 6], [3, 6, 99]]
soft_copy : [[9, 18, 27], [2, 4, 6], [3, 6, 99]]
{{< /boxmd >}}

### Shallow Copy Example 2
In this example, you will see that appending elements does not affect the other object.

```
# Appending new list to soft copy
soft_copy.append([14,28,42])

print(f"Original : {ls}")
print(f"soft_copy : {soft_copy}")
```
{{< boxmd >}}
**Output**
Original : [[9, 18, 27], [2, 4, 6], [3, 6, 9]]
soft_copy : [[9, 18, 27], [2, 4, 6], [3, 6, 9], [14, 28, 42]]
{{< /boxmd >}}


Please leave comments, if
1. You find anything incorrect.
2. You want to add more information to the topic.
3. You wish to add another example to the topic.
4. You need more details in regards to a specific section.
5. You are unable to execute an example code.
