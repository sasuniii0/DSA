# Complete Data Structures & Algorithms Encyclopedia
### The Ultimate Guide for Computer Science Students

---

## 📖 Table of Contents

### [PART I: LINEAR DATA STRUCTURES](#part-i-linear-data-structures-1)
1. [Arrays](#1-arrays)
2. [Linked Lists](#2-linked-lists)
3. [Stacks](#3-stacks)
4. [Queues](#4-queues)
5. [Priority Queues](#5-priority-queues)
6. [Hash Tables](#6-hash-tables)

### [PART II: NON-LINEAR DATA STRUCTURES](#part-ii-non-linear-data-structures-1)
7. [Trees](#7-trees)
8. [Binary Trees](#8-binary-trees)
9. [Binary Search Trees (BST)](#9-binary-search-trees-bst)
10. [AVL Trees](#10-avl-trees)
11. [Heaps](#11-heaps)
12. [Tries](#12-tries)
13. [Graphs](#13-graphs)

### [PART III: SORTING ALGORITHMS](#part-iii-sorting-algorithms-1)
14. [Introduction to Sorting](#14-introduction-to-sorting)
15. [Bubble Sort](#15-bubble-sort)
16. [Selection Sort](#16-selection-sort)
17. [Insertion Sort](#17-insertion-sort)
18. [Merge Sort](#18-merge-sort)
19. [Quick Sort](#19-quick-sort)
20. [Heap Sort](#20-heap-sort)
21. [Counting Sort](#21-counting-sort)

### [PART IV: SEARCHING ALGORITHMS](#part-iv-searching-algorithms-1)
22. [Introduction to Searching](#22-introduction-to-searching)
23. [Linear Search](#23-linear-search)
24. [Binary Search](#24-binary-search)
25. [Depth-First Search (DFS)](#25-depth-first-search-dfs)
26. [Breadth-First Search (BFS)](#26-breadth-first-search-bfs)

---

# PART I: LINEAR DATA STRUCTURES

Linear data structures organize data elements sequentially, where each element has exactly one predecessor and one successor (except for the first and last elements). Elements are arranged in a linear sequence, making traversal straightforward from beginning to end.

---

## 1. Arrays

### What is an Array?

An **array** is a collection of elements stored at contiguous memory locations. Each element can be accessed directly using an index, making arrays one of the most fundamental data structures in computer science.

### Key Characteristics

- **Contiguous Memory**: Elements stored in adjacent memory locations
- **Fixed Size**: Size typically determined at creation (in static arrays)
- **Random Access**: Direct access to any element via index in O(1) time
- **Homogeneous**: All elements of the same data type
- **Cache-Friendly**: Sequential storage improves cache performance

### Memory Representation

```
Array: [10, 20, 30, 40, 50]
Index:   0   1   2   3   4

Memory Layout:
Address: 1000  1004  1008  1012  1016
Value:    10    20    30    40    50
```

### Types of Arrays

#### 1. **Static Array**
- Fixed size at compile time
- Cannot grow or shrink

#### 2. **Dynamic Array**
- Can grow or shrink at runtime
- Implemented as Python lists, Java ArrayList, C++ vector

### Time Complexity Analysis

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| **Access** | O(1) | Direct index calculation |
| **Search** | O(n) | Must check each element |
| **Insert at end** | O(1) amortized | May require resizing |
| **Insert at position** | O(n) | Must shift elements |
| **Delete at end** | O(1) | Simple removal |
| **Delete at position** | O(n) | Must shift elements |

### Complete Implementation

```python
class DynamicArray:
    """
    A dynamic array implementation similar to Python's list
    Automatically resizes when capacity is reached
    """
    
    def __init__(self, capacity=1):
        """Initialize with given capacity"""
        self.capacity = capacity
        self.size = 0
        self.array = [None] * capacity
    
    def __len__(self):
        """Return the number of elements"""
        return self.size
    
    def __getitem__(self, index):
        """Get element at index - O(1)"""
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds")
        return self.array[index]
    
    def __setitem__(self, index, value):
        """Set element at index - O(1)"""
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds")
        self.array[index] = value
    
    def _resize(self, new_capacity):
        """
        Resize internal array - O(n)
        Creates new array and copies all elements
        """
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity
    
    def append(self, element):
        """
        Add element to end - O(1) amortized
        Doubles capacity when full
        """
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        
        self.array[self.size] = element
        self.size += 1
    
    def insert(self, index, element):
        """
        Insert element at specific index - O(n)
        All elements from index onwards shift right
        """
        if not 0 <= index <= self.size:
            raise IndexError("Index out of bounds")
        
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        
        # Shift elements to the right
        for i in range(self.size, index, -1):
            self.array[i] = self.array[i - 1]
        
        self.array[index] = element
        self.size += 1
    
    def delete(self, index):
        """
        Delete element at index - O(n)
        All elements after index shift left
        """
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds")
        
        # Shift elements to the left
        for i in range(index, self.size - 1):
            self.array[i] = self.array[i + 1]
        
        self.array[self.size - 1] = None
        self.size -= 1
        
        # Shrink if using less than 1/4 of capacity
        if self.size > 0 and self.size == self.capacity // 4:
            self._resize(self.capacity // 2)
    
    def pop(self):
        """Remove and return last element - O(1)"""
        if self.size == 0:
            raise IndexError("Pop from empty array")
        
        element = self.array[self.size - 1]
        self.delete(self.size - 1)
        return element
    
    def search(self, element):
        """
        Linear search for element - O(n)
        Returns index if found, -1 otherwise
        """
        for i in range(self.size):
            if self.array[i] == element:
                return i
        return -1
    
    def reverse(self):
        """Reverse array in-place - O(n)"""
        left, right = 0, self.size - 1
        while left < right:
            self.array[left], self.array[right] = \
                self.array[right], self.array[left]
            left += 1
            right -= 1
    
    def __str__(self):
        """String representation"""
        return '[' + ', '.join(str(self.array[i]) 
                               for i in range(self.size)) + ']'


# Example Usage and Testing
if __name__ == "__main__":
    # Create dynamic array
    arr = DynamicArray()
    
    # Append elements
    print("Adding elements...")
    for i in range(5):
        arr.append(i * 10)
    print(f"Array: {arr}")  # [0, 10, 20, 30, 40]
    print(f"Size: {len(arr)}, Capacity: {arr.capacity}")
    
    # Insert element
    arr.insert(2, 15)
    print(f"\nAfter inserting 15 at index 2: {arr}")
    
    # Access elements
    print(f"\nElement at index 3: {arr[3]}")
    
    # Search
    index = arr.search(30)
    print(f"Index of 30: {index}")
    
    # Delete
    arr.delete(1)
    print(f"\nAfter deleting index 1: {arr}")
    
    # Reverse
    arr.reverse()
    print(f"After reversing: {arr}")
    
    # Pop
    popped = arr.pop()
    print(f"\nPopped element: {popped}")
    print(f"Final array: {arr}")
```

### Common Array Operations

#### 1. **Rotation**

```python
def rotate_left(arr, d):
    """Rotate array left by d positions - O(n)"""
    n = len(arr)
    d = d % n  # Handle d > n
    return arr[d:] + arr[:d]

def rotate_right(arr, d):
    """Rotate array right by d positions - O(n)"""
    n = len(arr)
    d = d % n
    return arr[n-d:] + arr[:n-d]

# Example
arr = [1, 2, 3, 4, 5]
print(rotate_left(arr, 2))   # [3, 4, 5, 1, 2]
print(rotate_right(arr, 2))  # [4, 5, 1, 2, 3]
```

#### 2. **Finding Maximum and Minimum**

```python
def find_max_min(arr):
    """Find max and min in single pass - O(n)"""
    if not arr:
        return None, None
    
    max_val = min_val = arr[0]
    
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
        if arr[i] < min_val:
            min_val = arr[i]
    
    return max_val, min_val

# Example
arr = [3, 7, 1, 9, 2, 5]
max_val, min_val = find_max_min(arr)
print(f"Max: {max_val}, Min: {min_val}")  # Max: 9, Min: 1
```

#### 3. **Two Pointer Technique**

```python
def two_sum(arr, target):
    """
    Find two numbers that sum to target - O(n)
    Array must be sorted
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return None

# Example
arr = [1, 2, 3, 4, 5, 6]
result = two_sum(arr, 9)
print(f"Indices: {result}")  # [2, 5] (3 + 6 = 9)
```

#### 4. **Sliding Window**

```python
def max_sum_subarray(arr, k):
    """
    Find maximum sum of k consecutive elements - O(n)
    Uses sliding window technique
    """
    if len(arr) < k:
        return None
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Example
arr = [1, 4, 2, 10, 23, 3, 1, 0, 20]
k = 4
print(f"Max sum of {k} elements: {max_sum_subarray(arr, k)}")  # 39
```

### Multi-Dimensional Arrays

```python
class Matrix:
    """2D Array (Matrix) implementation"""
    
    def __init__(self, rows, cols, default=0):
        """Initialize matrix with given dimensions"""
        self.rows = rows
        self.cols = cols
        self.matrix = [[default] * cols for _ in range(rows)]
    
    def get(self, i, j):
        """Get element at (i, j) - O(1)"""
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise IndexError("Index out of bounds")
        return self.matrix[i][j]
    
    def set(self, i, j, value):
        """Set element at (i, j) - O(1)"""
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise IndexError("Index out of bounds")
        self.matrix[i][j] = value
    
    def transpose(self):
        """Return transposed matrix - O(rows * cols)"""
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.set(j, i, self.get(i, j))
        return result
    
    def __str__(self):
        """String representation"""
        return '\n'.join([' '.join(map(str, row)) 
                         for row in self.matrix])

# Example
m = Matrix(3, 3)
m.set(0, 0, 1)
m.set(1, 1, 5)
m.set(2, 2, 9)
print("Matrix:")
print(m)
print("\nTransposed:")
print(m.transpose())
```

### Real-World Applications

1. **Image Processing**: Images stored as 2D/3D arrays of pixels
2. **Database Tables**: Rows stored as arrays of records
3. **Buffers**: Circular buffers using arrays
4. **Lookup Tables**: Hash tables, routing tables
5. **Matrix Operations**: Scientific computing, graphics
6. **Dynamic Programming**: Memoization tables

### Advantages ✓

- **Fast Access**: O(1) random access via index
- **Memory Efficiency**: No extra memory for pointers
- **Cache Locality**: Better cache performance due to contiguous storage
- **Simple**: Easy to understand and implement
- **Predictable**: Fixed memory layout

### Disadvantages ✗

- **Fixed Size**: Static arrays can't grow (dynamic arrays can but expensive)
- **Expensive Insertions/Deletions**: O(n) time due to shifting
- **Memory Waste**: May allocate more than needed
- **No Built-in Methods**: Basic arrays lack high-level operations

### When to Use Arrays

✅ **Use arrays when:**
- You need fast random access to elements
- Size is known in advance or changes infrequently
- Memory locality is important for performance
- Implementing other data structures (stacks, queues, heaps)

❌ **Avoid arrays when:**
- Frequent insertions/deletions in the middle
- Size changes dramatically and unpredictably
- Memory is very limited and size unknown

---

## 2. Linked Lists

### What is a Linked List?

A **linked list** is a linear data structure where elements (nodes) are not stored contiguously in memory. Instead, each node contains data and a reference (pointer) to the next node in the sequence.

### Node Structure

```
┌─────────┬──────┐
│  Data   │ Next │
└─────────┴──────┘
```

### Memory Representation

```
Array:      [10][20][30][40]  (Contiguous)
            1000 1004 1008 1012

Linked List:
Head → [10|•]→[20|•]→[30|•]→[40|NULL]
       1000   2004   3008   4012
       (Scattered in memory)
```

### Types of Linked Lists

#### 1. Singly Linked List
Each node points to the next node only.

```
NULL ← HEAD → [A]→[B]→[C]→[D]→NULL
```

#### 2. Doubly Linked List
Each node has pointers to both next and previous nodes.

```
NULL←[A]⇄[B]⇄[C]⇄[D]→NULL
     ↑
    HEAD
```

#### 3. Circular Linked List
Last node points back to the first node.

```
    ┌──────────────┐
    ↓              │
HEAD→[A]→[B]→[C]→[D]┘
```

### Time Complexity Analysis

| Operation | Singly LL | Doubly LL | Array |
|-----------|-----------|-----------|-------|
| **Access** | O(n) | O(n) | O(1) |
| **Search** | O(n) | O(n) | O(n) |
| **Insert at head** | O(1) | O(1) | O(n) |
| **Insert at tail** | O(n)* | O(1)** | O(1) |
| **Insert at middle** | O(n) | O(n) | O(n) |
| **Delete at head** | O(1) | O(1) | O(n) |
| **Delete at tail** | O(n) | O(1)** | O(1) |
| **Delete given node** | O(n) | O(1)*** | O(n) |

*O(1) if tail pointer maintained  
**With tail pointer  
***If node reference is known

### Singly Linked List Implementation

```python
class Node:
    """Node class for singly linked list"""
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
    """Complete implementation of singly linked list"""
    
    def __init__(self):
        """Initialize empty list"""
        self.head = None
        self.size = 0
    
    def __len__(self):
        """Return size of list"""
        return self.size
    
    def is_empty(self):
        """Check if list is empty - O(1)"""
        return self.head is None
    
    def insert_at_beginning(self, data):
        """
        Insert at head - O(1)
        Most efficient insertion in singly linked list
        """
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def insert_at_end(self, data):
        """
        Insert at tail - O(n)
        Must traverse entire list
        """
        new_node = Node(data)
        
        if self.is_empty():
            self.head = new_node
            self.size += 1
            return
        
        current = self.head
        while current.next:
            current = current.next
        
        current.next = new_node
        self.size += 1
    
    def insert_at_position(self, data, position):
        """
        Insert at specific position - O(n)
        Position 0 is the head
        """
        if position < 0 or position > self.size:
            raise IndexError("Position out of bounds")
        
        if position == 0:
            self.insert_at_beginning(data)
            return
        
        new_node = Node(data)
        current = self.head
        
        # Navigate to node before insertion point
        for _ in range(position - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    def delete_at_beginning(self):
        """Delete first node - O(1)"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        data = self.head.data
        self.head = self.head.next
        self.size -= 1
        return data
    
    def delete_at_end(self):
        """Delete last node - O(n)"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        if self.head.next is None:
            data = self.head.data
            self.head = None
            self.size -= 1
            return data
        
        current = self.head
        while current.next.next:
            current = current.next
        
        data = current.next.data
        current.next = None
        self.size -= 1
        return data
    
    def delete_by_value(self, value):
        """
        Delete first occurrence of value - O(n)
        Returns True if deleted, False if not found
        """
        if self.is_empty():
            return False
        
        # Delete head
        if self.head.data == value:
            self.head = self.head.next
            self.size -= 1
            return True
        
        current = self.head
        while current.next:
            if current.next.data == value:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        
        return False
    
    def search(self, value):
        """
        Search for value - O(n)
        Returns position if found, -1 otherwise
        """
        current = self.head
        position = 0
        
        while current:
            if current.data == value:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def get(self, position):
        """Get value at position - O(n)"""
        if position < 0 or position >= self.size:
            raise IndexError("Position out of bounds")
        
        current = self.head
        for _ in range(position):
            current = current.next
        
        return current.data
    
    def reverse(self):
        """
        Reverse the linked list in-place - O(n)
        Changes the direction of all pointers
        """
        prev = None
        current = self.head
        
        while current:
            next_node = current.next  # Save next
            current.next = prev       # Reverse pointer
            prev = current            # Move prev forward
            current = next_node       # Move current forward
        
        self.head = prev
    
    def detect_cycle(self):
        """
        Detect if list has a cycle - O(n)
        Uses Floyd's Cycle Detection (Tortoise and Hare)
        """
        if self.is_empty():
            return False
        
        slow = fast = self.head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    
    def find_middle(self):
        """
        Find middle element - O(n)
        Uses two pointers: slow and fast
        """
        if self.is_empty():
            return None
        
        slow = fast = self.head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow.data
    
    def remove_duplicates(self):
        """
        Remove duplicates from sorted list - O(n)
        """
        if self.is_empty():
            return
        
        current = self.head
        
        while current and current.next:
            if current.data == current.next.data:
                current.next = current.next.next
                self.size -= 1
            else:
                current = current.next
    
    def display(self):
        """Display list as string - O(n)"""
        if self.is_empty():
            return "Empty List"
        
        elements = []
        current = self.head
        
        while current:
            elements.append(str(current.data))
            current = current.next
        
        return " → ".join(elements) + " → NULL"
    
    def to_array(self):
        """Convert to array - O(n)"""
        arr = []
        current = self.head
        
        while current:
            arr.append(current.data)
            current = current.next
        
        return arr
    
    def __str__(self):
        """String representation"""
        return self.display()


# Example Usage
if __name__ == "__main__":
    # Create linked list
    ll = SinglyLinkedList()
    
    # Insert elements
    print("Inserting elements...")
    ll.insert_at_end(10)
    ll.insert_at_end(20)
    ll.insert_at_end(30)
    ll.insert_at_beginning(5)
    print(f"List: {ll}")  # 5 → 10 → 20 → 30 → NULL
    print(f"Size: {len(ll)}")
    
    # Insert at position
    ll.insert_at_position(15, 2)
    print(f"\nAfter inserting 15 at position 2: {ll}")
    
    # Search
    pos = ll.search(20)
    print(f"\nPosition of 20: {pos}")
    
    # Get element
    element = ll.get(2)
    print(f"Element at position 2: {element}")
    
    # Delete operations
    ll.delete_by_value(15)
    print(f"\nAfter deleting 15: {ll}")
    
    # Find middle
    middle = ll.find_middle()
    print(f"\nMiddle element: {middle}")
    
    # Reverse
    ll.reverse()
    print(f"After reversing: {ll}")
    
    # Convert to array
    arr = ll.to_array()
    print(f"\nAs array: {arr}")
```

### Doubly Linked List Implementation

```python
class DoublyNode:
    """Node class for doubly linked list"""
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    """Complete implementation of doubly linked list"""
    
    def __init__(self):
        """Initialize empty list"""
        self.head = None
        self.tail = None
        self.size = 0
    
    def is_empty(self):
        """Check if list is empty - O(1)"""
        return self.head is None
    
    def insert_at_beginning(self, data):
        """Insert at head - O(1)"""
        new_node = DoublyNode(data)
        
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, data):
        """Insert at tail - O(1) with tail pointer"""
        new_node = DoublyNode(data)
        
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def insert_after_node(self, node, data):
        """
        Insert after given node - O(1)
        Advantage of doubly linked list
        """
        if node is None:
            raise ValueError("Node cannot be None")
        
        new_node = DoublyNode(data)
        new_node.next = node.next
        new_node.prev = node
        
        if node.next:
            node.next.prev = new_node
        else:
            self.tail = new_node
        
        node.next = new_node
        self.size += 1
    
    def delete_node(self, node):
        """
        Delete given node - O(1) if node reference known
        Major advantage over singly linked list
        """
        if node is None:
            raise ValueError("Node cannot be None")
        
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        
        self.size -= 1
    
    def delete_at_beginning(self):
        """Delete first node - O(1)"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        data = self.head.data
        self.head = self.head.next
        
        if self.head:
            self.head.prev = None
        else:
            self.tail = None
        
        self.size -= 1
        return data
    
    def delete_at_end(self):
        """Delete last node - O(1) with tail pointer"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        data = self.tail.data
        self.tail = self.tail.prev
        
        if self.tail:
            self.tail.next = None
        else:
            self.head = None
        
        self.size -= 1
        return data
    
    def reverse(self):
        """Reverse the list - O(n)"""
        current = self.head
        self.head, self.tail = self.tail, self.head
        
        while current:
            current.prev, current.next = current.next, current.prev
            current = current.prev
    
    def display_forward(self):
        """Display list forward - O(n)"""
        if self.is_empty():
            return "Empty List"
        
        elements = []
        current = self.head
        
        while current:
            elements.append(str(current.data))
            current = current.next
        
        return " ⇄ ".join(elements) + " ⇄ NULL"
    
    def display_backward(self):
        """Display list backward - O(n)"""
        if self.is_empty():
            return "Empty List"
        
        elements = []
        current = self.tail
        
        while current:
            elements.append(str(current.data))
            current = current.prev
        
        return "NULL ⇄ " + " ⇄ ".join(elements)
    
    def __str__(self):
        return self.display_forward()


# Example Usage
dll = DoublyLinkedList()
dll.insert_at_end(10)
dll.insert_at_end(20)
dll.insert_at_end(30)
dll.insert_at_beginning(5)

print("Forward:", dll.display_forward())
print("Backward:", dll.display_backward())

dll.reverse()
print("After reverse:", dll)
```

### Circular Linked List Implementation

```python
class CircularLinkedList:
    """Circular Singly Linked List"""
    
    def __init__(self):
        self.head = None
        self.size = 0
    
    def insert_at_beginning(self, data):
        """Insert at beginning - O(n) due to finding last node"""
        new_node = Node(data)
        
        if not self.head:
            new_node.next = new_node
            self.head = new_node
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            
            new_node.next = self.head
            current.next = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, data):
        """Insert at end - O(n)"""
        new_node = Node(data)
        
        if not self.head:
            new_node.next = new_node
            self.head = new_node
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            
            current.next = new_node
            new_node.next = self.head
        
        self.size += 1
    
    def display(self):
        """Display circular list - O(n)"""
        if not self.head:
            return "Empty List"
        
        elements = []
        current = self.head
        
        while True:
            elements.append(str(current.data))
            current = current.next
            if current == self.head:
                break
        
        return " → ".join(elements) + " → (back to head)"

# Example
cll = CircularLinkedList()
cll.insert_at_end(1)
cll.insert_at_end(2)
cll.insert_at_end(3)
print(cll.display())  # 1 → 2 → 3 → (back to head)
```

### Advanced Linked List Problems

#### 1. Merge Two Sorted Lists

```python
def merge_sorted_lists(l1, l2):
    """
    Merge two sorted linked lists - O(m + n)
    Returns head of merged list
    """
    dummy = Node(0)
    current = dummy
    
    while l1 and l2:
        if l1.data <= l2.data:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    # Attach remaining nodes
    current.next = l1 if l1 else l2
    
    return dummy.next
```

#### 2. Detect and Remove Cycle

```python
def detect_and_remove_cycle(head):
    """Detect cycle and remove it - O(n)"""
    if not head or not head.next:
        return
    
    slow = fast = head
    
    # Detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            break
    else:
        return  # No cycle
    
    # Find start of cycle
    slow = head
    while slow.next != fast.next:
        slow = slow.next
        fast = fast.next
    
    # Remove cycle
    fast.next = None
```

#### 3. Clone List with Random Pointer

```python
class RandomNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.random = None

def clone_random_list(head):
    """
    Clone linked list with random pointers - O(n)
    Each node has next and random pointer
    """
    if not head:
        return None
    
    # Step 1: Create copy nodes
    current = head
    while current:
        copy = RandomNode(current.data)
        copy.next = current.next
        current.next = copy
        current = copy.next
    
    # Step 2: Assign random pointers
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next
    
    # Step 3: Separate lists
    current = head
    copy_head = head.next
    
    while current:
        copy = current.next
        current.next = copy.next
        copy.next = copy.next.next if copy.next else None
        current = current.next
    
    return copy_head
```

#### 4. Add Two Numbers

```python
def add_two_numbers(l1, l2):
    """
    Add two numbers represented as linked lists - O(max(m,n))
    Lists store digits in reverse order
    Example: 342 + 465 = 807 → [2,4,3] + [5,6,4] = [7,0,8]
    """
    dummy = Node(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.data if l1 else 0
        val2 = l2.data if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        
        current.next = Node(total % 10)
        current = current.next
        
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next
```

### Real-World Applications

1. **Music Playlists**: Next/Previous song navigation
2. **Browser History**: Back/Forward buttons
3. **Undo/Redo Functionality**: Doubly linked list for bidirectional traversal
4. **Memory Management**: Free memory blocks in OS
5. **Image Viewer**: Next/Previous image
6. **LRU Cache**: Doubly linked list + hash map
7. **Polynomial Arithmetic**: Each term as a node

### Advantages ✓

- **Dynamic Size**: Grows and shrinks easily
- **Efficient Insertions/Deletions**: O(1) at known positions
- **No Wasted Memory**: Uses exactly what's needed
- **Easy Implementation**: Of other structures (stacks, queues)

### Disadvantages ✗

- **No Random Access**: Must traverse from head
- **Extra Memory**: Pointers require additional space
- **Cache Unfriendly**: Nodes scattered in memory
- **Traversal**: Only forward (singly), slower access

### When to Use Linked Lists

✅ **Use linked lists when:**
- Frequent insertions/deletions at beginning
- Unknown or dynamically changing size
- Memory scattered is acceptable
- Implementing stacks, queues, graphs

❌ **Avoid linked lists when:**
- Random access is required frequently
- Memory locality is critical
- Extra pointer memory is a concern
- Searching is the primary operation

---

## 3. Stacks

### What is a Stack?

A **stack** is a linear data structure that follows the **LIFO** (Last In, First Out) principle. The last element added is the first one to be removed, like a stack of plates.

### Key Operations

```
┌─────────┐
│    3    │ ← Top (Last In, First Out)
├─────────┤
│    2    │
├─────────┤
│    1    │
└─────────┘

Push(4):        Pop():
┌─────────┐     ┌─────────┐
│    4    │     │    2    │ ← New Top
├─────────┤     ├─────────┤
│    3    │     │    1    │
├─────────┤     └─────────┘
│    2    │
├─────────┤
│    1    │
└─────────┘
```

### Core Operations

| Operation | Description | Time |
|-----------|-------------|------|
| **push(item)** | Add item to top | O(1) |
| **pop()** | Remove and return top item | O(1) |
| **peek()** | Return top item without removing | O(1) |
| **isEmpty()** | Check if stack is empty | O(1) |
| **size()** | Get number of elements | O(1) |

### Implementation Methods

#### 1. Array-Based Stack

```python
class ArrayStack:
    """Stack implementation using dynamic array"""
    
    def __init__(self, capacity=10):
        """Initialize stack with optional capacity"""
        self.capacity = capacity
        self.stack = []
        self.top = -1
    
    def is_empty(self):
        """Check if stack is empty - O(1)"""
        return self.top == -1
    
    def is_full(self):
        """Check if stack is full - O(1)"""
        return self.top == self.capacity - 1
    
    def push(self, item):
        """
        Push item onto stack - O(1)
        Raises OverflowError if stack is full
        """
        if self.is_full():
            raise OverflowError("Stack overflow")
        
        self.stack.append(item)
        self.top += 1
    
    def pop(self):
        """
        Pop item from stack - O(1)
        Raises IndexError if stack is empty
        """
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        
        self.top -= 1
        return self.stack.pop()
    
    def peek(self):
        """
        Return top item without removing - O(1)
        Raises IndexError if stack is empty
        """
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        
        return self.stack[self.top]
    
    def size(self):
        """Return number of elements - O(1)"""
        return self.top + 1
    
    def display(self):
        """Display stack contents"""
        if self.is_empty():
            return "Empty Stack"
        
        return " | ".join(str(self.stack[i]) 
                         for i in range(self.top, -1, -1))
    
    def __str__(self):
        return f"Stack (top→bottom): {self.display()}"


# Example Usage
stack = ArrayStack()
stack.push(10)
stack.push(20)
stack.push(30)
print(stack)  # Stack (top→bottom): 30 | 20 | 10

print(f"Peek: {stack.peek()}")  # 30
print(f"Pop: {stack.pop()}")    # 30
print(f"Size: {stack.size()}")  # 2
print(stack)  # Stack (top→bottom): 20 | 10
```

#### 2. Linked List-Based Stack

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedStack:
    """Stack implementation using linked list"""
    
    def __init__(self):
        """Initialize empty stack"""
        self.top = None
        self.length = 0
    
    def is_empty(self):
        """Check if stack is empty - O(1)"""
        return self.top is None
    
    def push(self, item):
        """
        Push item onto stack - O(1)
        No overflow as linked list is dynamic
        """
        new_node = Node(item)
        new_node.next = self.top
        self.top = new_node
        self.length += 1
    
    def pop(self):
        """Pop item from stack - O(1)"""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        
        data = self.top.data
        self.top = self.top.next
        self.length -= 1
        return data
    
    def peek(self):
        """Return top item - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        
        return self.top.data
    
    def size(self):
        """Return number of elements - O(1)"""
        return self.length
    
    def display(self):
        """Display stack contents - O(n)"""
        if self.is_empty():
            return "Empty Stack"
        
        elements = []
        current = self.top
        
        while current:
            elements.append(str(current.data))
            current = current.next
        
        return " → ".join(elements)
    
    def __str__(self):
        return f"Stack (top→bottom): {self.display()}"


# Example Usage
stack = LinkedStack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack)  # Stack (top→bottom): 3 → 2 → 1
```

### Stack Applications and Examples

#### 1. Balanced Parentheses Checker

```python
def is_balanced(expression):
    """
    Check if parentheses are balanced - O(n)
    
    Examples:
        "((()))" → True
        "({[]})" → True
        "((]" → False
        "({)" → False
    """
    stack = []
    matching = {')': '(', '}': '{', ']': '['}
    
    for char in expression:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack.pop() != matching[char]:
                return False
    
    return not stack


# Test cases
test_cases = [
    "(())",      # True
    "({[]})",    # True
    "((]",       # False
    "({)}",      # False
    "()[]{}"     # True
]

for expr in test_cases:
    print(f"{expr}: {is_balanced(expr)}")
```

#### 2. Infix to Postfix Conversion

```python
def infix_to_postfix(expression):
    """
    Convert infix to postfix notation - O(n)
    
    Infix: A + B * C
    Postfix: A B C * +
    
    Operator precedence: ^ > *, / > +, -
    """
    stack = []
    output = []
    
    # Operator precedence
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    
    # Right associative operators
    right_assoc = {'^'}
    
    for char in expression:
        if char.isalnum():
            # Operand: add to output
            output.append(char)
        
        elif char == '(':
            stack.append(char)
        
        elif char == ')':
            # Pop until matching '('
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove '('
        
        elif char in precedence:
            # Operator
            while (stack and 
                   stack[-1] != '(' and
                   stack[-1] in precedence and
                   (precedence[stack[-1]] > precedence[char] or
                    (precedence[stack[-1]] == precedence[char] and
                     char not in right_assoc))):
                output.append(stack.pop())
            
            stack.append(char)
    
    # Pop remaining operators
    while stack:
        output.append(stack.pop())
    
    return ''.join(output)


# Test cases
expressions = [
    "A+B*C",           # ABC*+
    "(A+B)*C",         # AB+C*
    "A+B*C-D",         # ABC*+D-
    "A*(B+C)/D"        # ABC+*D/
]

for expr in expressions:
    print(f"{expr} → {infix_to_postfix(expr)}")
```

#### 3. Postfix Evaluation

```python
def evaluate_postfix(expression):
    """
    Evaluate postfix expression - O(n)
    
    Example:
        "5 6 2 + * 12 4 / -" = 5 * (6+2) - 12/4 = 37
    """
    stack = []
    
    for token in expression.split():
        if token.lstrip('-').isdigit():
            # Operand: push to stack
            stack.append(int(token))
        else:
            # Operator: pop two operands
            if len(stack) < 2:
                raise ValueError("Invalid expression")
            
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a / b)
            elif token == '^':
                stack.append(a ** b)
            else:
                raise ValueError(f"Unknown operator: {token}")
    
    if len(stack) != 1:
        raise ValueError("Invalid expression")
    
    return stack[0]


# Test cases
expressions = [
    "5 6 +",              # 11
    "5 6 2 + *",          # 40
    "5 6 2 + * 12 4 / -", # 37
    "15 7 1 1 + - / 3 * 2 1 1 + + -"  # 5
]

for expr in expressions:
    print(f"{expr} = {evaluate_postfix(expr)}")
```

#### 4. Next Greater Element

```python
def next_greater_element(arr):
    """
    Find next greater element for each element - O(n)
    
    Example:
        [4, 5, 2, 25] → [5, 25, 25, -1]
    """
    n = len(arr)
    result = [-1] * n
    stack = []
    
    # Traverse from right to left
    for i in range(n - 1, -1, -1):
        # Pop elements smaller than current
        while stack and stack[-1] <= arr[i]:
            stack.pop()
        
        # Next greater element is top of stack
        if stack:
            result[i] = stack[-1]
        
        # Push current element
        stack.append(arr[i])
    
    return result


# Test
arr = [4, 5, 2, 25, 10]
result = next_greater_element(arr)
print(f"Array: {arr}")
print(f"Next Greater: {result}")
# [5, 25, 25, -1, -1]
```

#### 5. Stock Span Problem

```python
def calculate_span(prices):
    """
    Calculate stock span for each day - O(n)
    Span = number of consecutive days before current day
           where price <= current price
    
    Example:
        [100, 80, 60, 70, 60, 75, 85]
        →[1, 1, 1, 2, 1, 4, 6]
    """
    n = len(prices)
    span = [0] * n
    stack = []  # Stack of indices
    
    for i in range(n):
        # Pop elements with price <= current
        while stack and prices[stack[-1]] <= prices[i]:
            stack.pop()
        
        # Span is distance to previous greater element
        span[i] = i + 1 if not stack else i - stack[-1]
        
        stack.append(i)
    
    return span


# Test
prices = [100, 80, 60, 70, 60, 75, 85]
spans = calculate_span(prices)
print(f"Prices: {prices}")
print(f"Spans:  {spans}")
# [1, 1, 1, 2, 1, 4, 6]
```

#### 6. Min Stack (Get Minimum in O(1))

```python
class MinStack:
    """
    Stack that supports getMin() in O(1)
    Uses two stacks: main and min
    """
    
    def __init__(self):
        self.main_stack = []
        self.min_stack = []
    
    def push(self, val):
        """Push value - O(1)"""
        self.main_stack.append(val)
        
        # Update min stack
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        """Pop value - O(1)"""
        if not self.main_stack:
            raise IndexError("Pop from empty stack")
        
        val = self.main_stack.pop()
        
        # Update min stack
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        
        return val
    
    def top(self):
        """Get top value - O(1)"""
        if not self.main_stack:
            raise IndexError("Stack is empty")
        return self.main_stack[-1]
    
    def getMin(self):
        """Get minimum value - O(1)"""
        if not self.min_stack:
            raise IndexError("Stack is empty")
        return self.min_stack[-1]


# Example Usage
min_stack = MinStack()
min_stack.push(3)
min_stack.push(5)
print(f"Min: {min_stack.getMin()}")  # 3

min_stack.push(2)
min_stack.push(1)
print(f"Min: {min_stack.getMin()}")  # 1

min_stack.pop()
print(f"Min: {min_stack.getMin()}")  # 2

min_stack.pop()
print(f"Min: {min_stack.getMin()}")  # 3
```

#### 7. Largest Rectangle in Histogram

```python
def largest_rectangle_area(heights):
    """
    Find largest rectangle in histogram - O(n)
    Uses stack to find previous and next smaller elements
    
    Example:
        [2, 1, 5, 6, 2, 3] → 10
    """
    stack = []
    max_area = 0
    index = 0
    
    while index < len(heights):
        # Push if increasing
        if not stack or heights[index] >= heights[stack[-1]]:
            stack.append(index)
            index += 1
        else:
            # Calculate area with popped bar as smallest
            top = stack.pop()
            width = index if not stack else index - stack[-1] - 1
            area = heights[top] * width
            max_area = max(max_area, area)
    
    # Process remaining bars
    while stack:
        top = stack.pop()
        width = index if not stack else index - stack[-1] - 1
        area = heights[top] * width
        max_area = max(max_area, area)
    
    return max_area


# Test
heights = [2, 1, 5, 6, 2, 3]
print(f"Heights: {heights}")
print(f"Max area: {largest_rectangle_area(heights)}")  # 10
```

#### 8. Valid Substring Removal

```python
def min_remove_to_make_valid(s):
    """
    Remove minimum parentheses to make string valid - O(n)
    
    Example:
        "lee(t(c)o)de)" → "lee(t(c)o)de"
        "a)b(c)d" → "ab(c)d"
    """
    stack = []
    to_remove = set()
    
    # Find invalid closing parentheses
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                to_remove.add(i)
    
    # Remaining '(' are invalid
    to_remove.update(stack)
    
    # Build result
    return ''.join(char for i, char in enumerate(s) 
                   if i not in to_remove)


# Test
test_cases = [
    "lee(t(c)o)de)",
    "a)b(c)d",
    "))(("
]

for s in test_cases:
    print(f"{s} → {min_remove_to_make_valid(s)}")
```

### Real-World Applications

1. **Function Call Stack**: Program execution, recursion
2. **Undo/Redo**: Text editors, graphics applications
3. **Browser History**: Back button functionality
4. **Expression Evaluation**: Calculators, compilers
5. **Backtracking Algorithms**: Maze solving, puzzle solving
6. **Memory Management**: Stack memory allocation
7. **String Reversal**: Text processing
8. **Depth-First Search**: Graph/tree traversal

### Advantages ✓

- **Simple**: Easy to implement and understand
- **Fast Operations**: All operations are O(1)
- **Memory Efficient**: No wasted space (array implementation)
- **Natural Recursion**: Mimics recursive call stack
- **LIFO Access**: Perfect for undo/backtracking

### Disadvantages ✗

- **No Random Access**: Can only access top element
- **Limited Size**: Array implementation has capacity limit
- **No Search**: Must pop elements to search

### When to Use Stacks

✅ **Use stacks when:**
- Need LIFO behavior
- Implementing undo/redo
- Expression parsing/evaluation
- Backtracking algorithms
- Recursive algorithms can be converted to iterative
- Function call simulation

❌ **Avoid stacks when:**
- Need random access to elements
- FIFO behavior required
- Need to search frequently
- Elements need to be accessed from both ends

---

*[Continuing in next part due to length...]*

Would you like me to continue with the remaining sections (Queues through Searching Algorithms)?

## 4. Queues

### What is a Queue?

A **queue** is a linear data structure that follows the **FIFO** (First In, First Out) principle. The first element added is the first one to be removed, like a line of people waiting.

### Key Operations

```
Front                    Rear
  ↓                        ↓
[1] ← [2] ← [3] ← [4] ← [5]
  ↓                        ↑
Dequeue                 Enqueue

Enqueue(6):
[1] ← [2] ← [3] ← [4] ← [5] ← [6]

Dequeue():
      [2] ← [3] ← [4] ← [5] ← [6]
Returns 1
```

### Core Operations

| Operation | Description | Time |
|-----------|-------------|------|
| **enqueue(item)** | Add item to rear | O(1) |
| **dequeue()** | Remove and return front item | O(1) |
| **front()** | Return front item | O(1) |
| **isEmpty()** | Check if queue is empty | O(1) |
| **size()** | Get number of elements | O(1) |

### Types of Queues

1. **Simple Queue**: Basic FIFO queue
2. **Circular Queue**: Last position connected to first
3. **Priority Queue**: Elements have priorities
4. **Deque (Double-Ended Queue)**: Insert/delete from both ends

### Simple Queue Implementation

```python
from collections import deque

class Queue:
    """Queue implementation using Python's deque"""
    
    def __init__(self):
        """Initialize empty queue"""
        self.items = deque()
    
    def is_empty(self):
        """Check if queue is empty - O(1)"""
        return len(self.items) == 0
    
    def enqueue(self, item):
        """Add item to rear - O(1)"""
        self.items.append(item)
    
    def dequeue(self):
        """
        Remove and return front item - O(1)
        Raises IndexError if queue is empty
        """
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.popleft()
    
    def front(self):
        """
        Return front item without removing - O(1)
        Raises IndexError if queue is empty
        """
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.items[0]
    
    def rear(self):
        """Return rear item - O(1)"""
        if self.is_empty():
            raise IndexError("Rear from empty queue")
        return self.items[-1]
    
    def size(self):
        """Return number of elements - O(1)"""
        return len(self.items)
    
    def display(self):
        """Display queue contents"""
        if self.is_empty():
            return "Empty Queue"
        return " ← ".join(str(item) for item in self.items)
    
    def __str__(self):
        return f"Front [{self.display()}] Rear"


# Example Usage
q = Queue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
print(q)  # Front [1 ← 2 ← 3] Rear

print(f"Dequeue: {q.dequeue()}")  # 1
print(f"Front: {q.front()}")       # 2
print(q)  # Front [2 ← 3] Rear
```

### Circular Queue Implementation

```python
class CircularQueue:
    """
    Circular Queue using array
    Efficient use of space - no wasted slots
    """
    
    def __init__(self, capacity):
        """Initialize with fixed capacity"""
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = -1
        self.rear = -1
        self.count = 0
    
    def is_empty(self):
        """Check if queue is empty - O(1)"""
        return self.count == 0
    
    def is_full(self):
        """Check if queue is full - O(1)"""
        return self.count == self.capacity
    
    def enqueue(self, item):
        """
        Add item to queue - O(1)
        Raises OverflowError if queue is full
        """
        if self.is_full():
            raise OverflowError("Queue is full")
        
        if self.is_empty():
            self.front = 0
        
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.count += 1
    
    def dequeue(self):
        """
        Remove and return front item - O(1)
        Raises IndexError if queue is empty
        """
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        item = self.queue[self.front]
        
        if self.front == self.rear:
            # Only one element
            self.front = self.rear = -1
        else:
            self.front = (self.front + 1) % self.capacity
        
        self.count -= 1
        return item
    
    def get_front(self):
        """Get front element - O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.queue[self.front]
    
    def get_rear(self):
        """Get rear element - O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.queue[self.rear]
    
    def size(self):
        """Get queue size - O(1)"""
        return self.count
    
    def display(self):
        """Display queue contents - O(n)"""
        if self.is_empty():
            return "Empty Queue"
        
        elements = []
        i = self.front
        for _ in range(self.count):
            elements.append(str(self.queue[i]))
            i = (i + 1) % self.capacity
        
        return " ← ".join(elements)
    
    def __str__(self):
        return f"Circular Queue [{self.display()}]"


# Example Usage
cq = CircularQueue(5)
for i in range(1, 6):
    cq.enqueue(i)

print(cq)  # Circular Queue [1 ← 2 ← 3 ← 4 ← 5]

cq.dequeue()
cq.dequeue()
print(f"After 2 dequeues: {cq}")

cq.enqueue(6)
cq.enqueue(7)
print(f"After 2 enqueues: {cq}")  # [3 ← 4 ← 5 ← 6 ← 7]
```

### Deque (Double-Ended Queue) Implementation

```python
class Deque:
    """
    Double-Ended Queue
    Allows insertion/deletion from both ends
    """
    
    def __init__(self):
        """Initialize empty deque"""
        self.items = []
    
    def is_empty(self):
        """Check if deque is empty - O(1)"""
        return len(self.items) == 0
    
    def add_front(self, item):
        """Add item to front - O(n) for list"""
        self.items.insert(0, item)
    
    def add_rear(self, item):
        """Add item to rear - O(1)"""
        self.items.append(item)
    
    def remove_front(self):
        """Remove item from front - O(n) for list"""
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        return self.items.pop(0)
    
    def remove_rear(self):
        """Remove item from rear - O(1)"""
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        return self.items.pop()
    
    def peek_front(self):
        """View front item - O(1)"""
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items[0]
    
    def peek_rear(self):
        """View rear item - O(1)"""
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items[-1]
    
    def size(self):
        """Get size - O(1)"""
        return len(self.items)
    
    def __str__(self):
        if self.is_empty():
            return "Empty Deque"
        return " ⇄ ".join(str(item) for item in self.items)


# Using Python's built-in deque (more efficient)
from collections import deque

class EfficientDeque:
    """Deque using collections.deque - O(1) for all operations"""
    
    def __init__(self):
        self.items = deque()
    
    def add_front(self, item):
        """Add to front - O(1)"""
        self.items.appendleft(item)
    
    def add_rear(self, item):
        """Add to rear - O(1)"""
        self.items.append(item)
    
    def remove_front(self):
        """Remove from front - O(1)"""
        if not self.items:
            raise IndexError("Remove from empty deque")
        return self.items.popleft()
    
    def remove_rear(self):
        """Remove from rear - O(1)"""
        if not self.items:
            raise IndexError("Remove from empty deque")
        return self.items.pop()


# Example
dq = Deque()
dq.add_rear(1)
dq.add_rear(2)
dq.add_front(0)
dq.add_front(-1)
print(dq)  # -1 ⇄ 0 ⇄ 1 ⇄ 2
```

### Queue Applications and Examples

#### 1. Level Order Tree Traversal (BFS)

```python
def level_order_traversal(root):
    """
    Level order traversal of binary tree - O(n)
    Uses queue for BFS
    """
    if not root:
        return []
    
    result = []
    queue = Queue()
    queue.enqueue(root)
    
    while not queue.is_empty():
        level_size = queue.size()
        level = []
        
        for _ in range(level_size):
            node = queue.dequeue()
            level.append(node.val)
            
            if node.left:
                queue.enqueue(node.left)
            if node.right:
                queue.enqueue(node.right)
        
        result.append(level)
    
    return result
```

#### 2. Generate Binary Numbers

```python
def generate_binary_numbers(n):
    """
    Generate binary numbers from 1 to n - O(n)
    Uses queue for efficient generation
    
    Example: n=5 → ["1", "10", "11", "100", "101"]
    """
    result = []
    queue = Queue()
    queue.enqueue("1")
    
    for _ in range(n):
        binary = queue.dequeue()
        result.append(binary)
        
        queue.enqueue(binary + "0")
        queue.enqueue(binary + "1")
    
    return result


# Test
print(generate_binary_numbers(10))
# ["1", "10", "11", "100", "101", "110", "111", "1000", "1001", "1010"]
```

#### 3. First Non-Repeating Character in Stream

```python
def first_non_repeating(stream):
    """
    Find first non-repeating character in stream - O(n)
    Uses queue and frequency map
    
    Example:
        "aabcddbc" → ['a', -1, 'b', 'b', 'b', 'b', 'c', -1]
    """
    from collections import deque, Counter
    
    queue = deque()
    freq = Counter()
    result = []
    
    for char in stream:
        freq[char] += 1
        queue.append(char)
        
        # Remove repeated characters from front
        while queue and freq[queue[0]] > 1:
            queue.popleft()
        
        if queue:
            result.append(queue[0])
        else:
            result.append(-1)
    
    return result


# Test
stream = "aabcddbc"
print(first_non_repeating(stream))
# ['a', -1, 'b', 'b', 'b', 'b', 'c', -1]
```

#### 4. Implement Stack Using Queues

```python
class StackUsingQueues:
    """
    Implement stack using two queues
    Push: O(n), Pop: O(1)
    """
    
    def __init__(self):
        self.q1 = Queue()
        self.q2 = Queue()
    
    def push(self, item):
        """Push item - O(n)"""
        # Add to q2
        self.q2.enqueue(item)
        
        # Move all from q1 to q2
        while not self.q1.is_empty():
            self.q2.enqueue(self.q1.dequeue())
        
        # Swap q1 and q2
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self):
        """Pop item - O(1)"""
        if self.q1.is_empty():
            raise IndexError("Pop from empty stack")
        return self.q1.dequeue()
    
    def top(self):
        """Get top item - O(1)"""
        if self.q1.is_empty():
            raise IndexError("Stack is empty")
        return self.q1.front()
    
    def is_empty(self):
        """Check if empty - O(1)"""
        return self.q1.is_empty()


# Example
stack = StackUsingQueues()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # 3
print(stack.top())  # 2
```

#### 5. Sliding Window Maximum

```python
def max_sliding_window(nums, k):
    """
    Find max in each sliding window - O(n)
    Uses deque to store indices
    
    Example:
        nums = [1,3,-1,-3,5,3,6,7], k = 3
        → [3, 3, 5, 5, 6, 7]
    """
    from collections import deque
    
    if not nums or k == 0:
        return []
    
    result = []
    dq = deque()  # Stores indices
    
    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (not useful)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result (from k-1 onwards)
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


# Test
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(max_sliding_window(nums, k))
# [3, 3, 5, 5, 6, 7]
```

#### 6. Rotting Oranges (Multi-Source BFS)

```python
def oranges_rotting(grid):
    """
    Find time for all oranges to rot - O(m*n)
    Uses BFS with queue
    
    0 = empty, 1 = fresh, 2 = rotten
    Rotten orange rots adjacent fresh oranges each minute
    """
    from collections import deque
    
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    
    # Find all rotten oranges and count fresh
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1
    
    # BFS
    minutes = 0
    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    
    while queue:
        r, c, mins = queue.popleft()
        minutes = max(minutes, mins)
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] == 1):
                grid[nr][nc] = 2
                fresh -= 1
                queue.append((nr, nc, mins + 1))
    
    return minutes if fresh == 0 else -1


# Test
grid = [
    [2,1,1],
    [1,1,0],
    [0,1,1]
]
print(oranges_rotting(grid))  # 4
```

### Real-World Applications

1. **CPU Scheduling**: Process queues in operating systems
2. **Print Queue**: Managing print jobs
3. **BFS Traversal**: Graph/tree level-order traversal
4. **Call Center Systems**: Customer service queues
5. **Request Handling**: Web servers, databases
6. **Keyboard Buffer**: Storing keystrokes
7. **Breadth-First Search**: Shortest path algorithms
8. **Cache Implementation**: FIFO page replacement

### Advantages ✓

- **Fair Access**: FIFO ensures fairness
- **Fast Operations**: Enqueue/dequeue in O(1)
- **Natural Order**: Maintains insertion order
- **Simple**: Easy to implement and understand
- **Efficient**: Good for sequential processing

### Disadvantages ✗

- **No Random Access**: Can only access front/rear
- **Limited Operations**: Can't access middle elements
- **Fixed Size**: Array implementation has capacity limit
- **Memory Overhead**: Linked list implementation uses extra space

### When to Use Queues

✅ **Use queues when:**
- FIFO behavior needed
- Level-order traversal (BFS)
- Task scheduling
- Request handling
- Resource sharing
- Buffering

❌ **Avoid queues when:**
- LIFO behavior needed (use stack)
- Random access required
- Priority-based processing (use priority queue)
- Need to access middle elements frequently

---

## 5. Priority Queues

### What is a Priority Queue?

A **priority queue** is an abstract data type where each element has a priority. Elements with higher priority are served before elements with lower priority, regardless of insertion order.

### Key Characteristics

- Elements have associated priorities
- Higher priority elements dequeued first
- Same priority → FIFO order
- Typically implemented using heaps

```
Regular Queue (FIFO):
Enqueue: [1] [2] [3] [4] [5]
Dequeue: 1 (first in)

Priority Queue:
Elements: [(3, priority=2), (1, priority=5), (4, priority=1)]
Dequeue: 1 (highest priority=5)
```

### Core Operations

| Operation | Heap Implementation | Time |
|-----------|-------------------|------|
| **insert(item, priority)** | Add element | O(log n) |
| **extract_max/min()** | Remove highest priority | O(log n) |
| **peek()** | View highest priority | O(1) |
| **increase_priority()** | Update priority | O(log n) |
| **is_empty()** | Check if empty | O(1) |

### Implementation Using Heap

```python
import heapq

class PriorityQueue:
    """
    Priority Queue using min-heap
    Higher priority value = higher priority
    """
    
    def __init__(self):
        """Initialize empty priority queue"""
        self.heap = []
        self.entry_count = 0  # For tie-breaking
    
    def is_empty(self):
        """Check if queue is empty - O(1)"""
        return len(self.heap) == 0
    
    def push(self, item, priority):
        """
        Add item with priority - O(log n)
        Negates priority for max-heap behavior
        """
        # Python's heapq is min-heap, negate for max-heap
        entry = (-priority, self.entry_count, item)
        heapq.heappush(self.heap, entry)
        self.entry_count += 1
    
    def pop(self):
        """
        Remove and return highest priority item - O(log n)
        Raises IndexError if queue is empty
        """
        if self.is_empty():
            raise IndexError("Pop from empty priority queue")
        
        priority, _, item = heapq.heappop(self.heap)
        return item, -priority
    
    def peek(self):
        """
        View highest priority item - O(1)
        Raises IndexError if queue is empty
        """
        if self.is_empty():
            raise IndexError("Peek from empty priority queue")
        
        priority, _, item = self.heap[0]
        return item, -priority
    
    def size(self):
        """Get number of elements - O(1)"""
        return len(self.heap)
    
    def __str__(self):
        """String representation"""
        if self.is_empty():
            return "Empty Priority Queue"
        
        # Sort by priority for display
        items = [(-p, item) for p, _, item in self.heap]
        items.sort(reverse=True)
        return " < ".join(f"{item}(p={p})" for p, item in items)


# Example Usage
pq = PriorityQueue()

# Add tasks with priorities
pq.push("Low priority task", 1)
pq.push("High priority task", 5)
pq.push("Medium priority task", 3)
pq.push("Critical task", 10)

print(f"Size: {pq.size()}")  # 4

# Process tasks by priority
while not pq.is_empty():
    task, priority = pq.pop()
    print(f"Processing: {task} (priority {priority})")

# Output:
# Critical task (priority 10)
# High priority task (priority 5)
# Medium priority task (priority 3)
# Low priority task (priority 1)
```

### Min Priority Queue

```python
class MinPriorityQueue:
    """
    Min Priority Queue - smallest value has highest priority
    Used for: Dijkstra's, Prim's, Huffman coding
    """
    
    def __init__(self):
        self.heap = []
    
    def push(self, item, priority):
        """Add item with priority - O(log n)"""
        heapq.heappush(self.heap, (priority, item))
    
    def pop(self):
        """Remove minimum priority item - O(log n)"""
        if not self.heap:
            raise IndexError("Pop from empty queue")
        priority, item = heapq.heappop(self.heap)
        return item, priority
    
    def peek(self):
        """View minimum priority item - O(1)"""
        if not self.heap:
            raise IndexError("Peek from empty queue")
        priority, item = self.heap[0]
        return item, priority
    
    def is_empty(self):
        return len(self.heap) == 0


# Example: Find k smallest elements
def k_smallest_elements(arr, k):
    """Find k smallest elements using min heap - O(n log k)"""
    pq = MinPriorityQueue()
    
    for num in arr:
        pq.push(num, num)
    
    result = []
    for _ in range(k):
        if not pq.is_empty():
            item, _ = pq.pop()
            result.append(item)
    
    return result


arr = [7, 10, 4, 3, 20, 15]
print(f"3 smallest: {k_smallest_elements(arr, 3)}")  # [3, 4, 7]
```

### Custom Priority Queue Implementation

```python
class CustomHeap:
    """
    Custom Min-Heap implementation for priority queue
    Understanding heap operations
    """
    
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, key):
        """
        Insert element - O(log n)
        Add at end and bubble up
        """
        self.heap.append(key)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        """Maintain heap property upward"""
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            self.swap(i, self.parent(i))
            i = self.parent(i)
    
    def extract_min(self):
        """
        Remove minimum element - O(log n)
        Replace root with last, then bubble down
        """
        if not self.heap:
            raise IndexError("Extract from empty heap")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return min_val
    
    def _heapify_down(self, i):
        """Maintain heap property downward"""
        min_idx = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left] < self.heap[min_idx]:
            min_idx = left
        
        if right < len(self.heap) and self.heap[right] < self.heap[min_idx]:
            min_idx = right
        
        if min_idx != i:
            self.swap(i, min_idx)
            self._heapify_down(min_idx)
    
    def get_min(self):
        """Get minimum without removing - O(1)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def decrease_key(self, i, new_val):
        """
        Decrease value at index - O(log n)
        Used in Dijkstra's algorithm
        """
        if new_val > self.heap[i]:
            raise ValueError("New value is greater")
        
        self.heap[i] = new_val
        self._heapify_up(i)
    
    def delete(self, i):
        """Delete element at index - O(log n)"""
        self.decrease_key(i, float('-inf'))
        self.extract_min()
    
    def build_heap(self, arr):
        """
        Build heap from array - O(n)
        More efficient than inserting one by one
        """
        self.heap = arr.copy()
        
        # Start from last non-leaf node
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def size(self):
        return len(self.heap)


# Example
heap = CustomHeap()
elements = [5, 3, 8, 1, 9, 2]

for elem in elements:
    heap.insert(elem)

print("Extracting in sorted order:")
while not heap.is_empty():
    print(heap.extract_min(), end=" ")
# Output: 1 2 3 5 8 9
```

### Priority Queue Applications

#### 1. Task Scheduler

```python
class Task:
    def __init__(self, name, priority, duration):
        self.name = name
        self.priority = priority
        self.duration = duration
    
    def __repr__(self):
        return f"{self.name}(p={self.priority})"

def task_scheduler(tasks):
    """
    Schedule tasks by priority - O(n log n)
    Higher priority tasks execute first
    """
    pq = PriorityQueue()
    
    # Add all tasks
    for task in tasks:
        pq.push(task, task.priority)
    
    schedule = []
    total_time = 0
    
    while not pq.is_empty():
        task, _ = pq.pop()
        schedule.append((total_time, task))
        total_time += task.duration
    
    return schedule


# Example
tasks = [
    Task("Email", 2, 5),
    Task("Bug Fix", 5, 30),
    Task("Meeting", 4, 15),
    Task("Documentation", 1, 10)
]

schedule = task_scheduler(tasks)
print("Task Schedule:")
for time, task in schedule:
    print(f"Time {time}: {task.name} (priority {task.priority})")
```

#### 2. Merge K Sorted Arrays

```python
def merge_k_sorted_arrays(arrays):
    """
    Merge k sorted arrays - O(n log k)
    Uses min heap to track smallest elements
    """
    import heapq
    
    # Min heap: (value, array_index, element_index)
    heap = []
    
    # Add first element from each array
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))
    
    result = []
    
    while heap:
        val, arr_i, elem_i = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same array
        if elem_i + 1 < len(arrays[arr_i]):
            next_val = arrays[arr_i][elem_i + 1]
            heapq.heappush(heap, (next_val, arr_i, elem_i + 1))
    
    return result


# Test
arrays = [
    [1, 4, 7],
    [2, 5, 8],
    [3, 6, 9]
]
print(merge_k_sorted_arrays(arrays))
# [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

#### 3. Find Median from Data Stream

```python
class MedianFinder:
    """
    Find median from data stream - O(log n) insert, O(1) median
    Uses two heaps: max heap (lower half) and min heap (upper half)
    """
    
    def __init__(self):
        self.max_heap = []  # Lower half (negated for max)
        self.min_heap = []  # Upper half
    
    def add_num(self, num):
        """Add number to stream - O(log n)"""
        # Add to max heap (lower half)
        heapq.heappush(self.max_heap, -num)
        
        # Balance: move largest from max to min
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        
        # Ensure max heap has ≥ elements
        if len(self.max_heap) < len(self.min_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def find_median(self):
        """Find median - O(1)"""
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2


# Example
mf = MedianFinder()
for num in [1, 2, 3, 4, 5]:
    mf.add_num(num)
    print(f"After adding {num}: median = {mf.find_median()}")

# Output:
# After adding 1: median = 1
# After adding 2: median = 1.5
# After adding 3: median = 2
# After adding 4: median = 2.5
# After adding 5: median = 3
```

#### 4. K Closest Points to Origin

```python
def k_closest_points(points, k):
    """
    Find k closest points to origin - O(n log k)
    Uses max heap of size k
    """
    import heapq
    
    # Max heap of (-distance, point)
    heap = []
    
    for x, y in points:
        dist = -(x*x + y*y)  # Negative for max heap
        
        if len(heap) < k:
            heapq.heappush(heap, (dist, (x, y)))
        elif dist > heap[0][0]:
            heapq.heapreplace(heap, (dist, (x, y)))
    
    return [point for _, point in heap]


# Test
points = [[1,3], [-2,2], [5,8], [0,1]]
k = 2
print(f"{k} closest points: {k_closest_points(points, k)}")
# [[0, 1], [-2, 2]]
```

### Real-World Applications

1. **Dijkstra's Algorithm**: Shortest path finding
2. **Huffman Coding**: Data compression
3. **Prim's Algorithm**: Minimum spanning tree
4. **A* Search**: Pathfinding in games/maps
5. **Operating Systems**: Process scheduling (priority-based)
6. **Event-Driven Simulation**: Processing events by timestamp
7. **Bandwidth Management**: Quality of Service (QoS)
8. **Load Balancing**: Distributing tasks by priority

### Advantages ✓

- **Efficient Priority Access**: O(log n) operations
- **Dynamic**: Can handle changing priorities
- **Flexible**: Works with any comparable data type
- **Optimal**: Best-known complexity for priority operations
- **Versatile**: Many algorithmic applications

### Disadvantages ✗

- **No Random Access**: Can't access middle elements
- **Complex**: More complex than simple queue
- **Memory**: Heap structure requires contiguous array
- **Search**: O(n) to search for specific element

### When to Use Priority Queues

✅ **Use priority queues when:**
- Elements have priorities
- Need efficient access to highest/lowest priority
- Implementing graph algorithms (Dijkstra's, Prim's)
- Task scheduling with priorities
- Event-driven systems
- K-largest/smallest problems

❌ **Avoid priority queues when:**
- FIFO order sufficient (use simple queue)
- Need random access
- All elements have equal priority
- Frequent priority updates (consider other structures)

---

*[The encyclopedia continues with Hash Tables, then Non-Linear Data Structures, Sorting Algorithms, and Searching Algorithms...]*
