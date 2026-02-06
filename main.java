import java.util.Arrays;
import java.util.*;

//Dynamic Array (ArrayList logic)

class DynamicArray {
    private int[] arr;
    private int size;
    private int capacity;

    public DynamicArray() {
        capacity = 1;
        size = 0;
        arr = new int[capacity];
    }

    private void resize(int newCapacity) {
        arr = Arrays.copyOf(arr, newCapacity);
        capacity = newCapacity;
    }

    public void append(int item) {
        if (size == capacity) resize(2 * capacity);
        arr[size++] = item;
    }

    public void insert(int index, int item) {
        if (size == capacity) resize(2 * capacity);
        for (int i = size; i > index; i--) arr[i] = arr[i - 1];
        arr[index] = item;
        size++;
    }

    public void delete(int index) {
        for (int i = index; i < size - 1; i++) arr[i] = arr[i + 1];
        size--;
    }
}

class Node {
    int data;
    Node next;
    Node(int data) { this.data = data; this.next = null; }
}

// Singly Linked List

class SinglyLinkedList {
    Node head;

    public void insertAtBeginning(int data) {
        Node newNode = new Node(data);
        newNode.next = head;
        head = newNode;
    }

    public void reverse() {
        Node prev = null, current = head, next;
        while (current != null) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        head = prev;
    }
}

//Stacks and Queues

// Stack Example
Deque<Integer> stack = new ArrayDeque<>();
stack.push(10); // Push
int top = stack.pop(); // Pop

// Queue Example
Queue<Integer> queue = new LinkedList<>();
queue.add(10); // Enqueue
int front = queue.poll(); // Dequeue


//Binary Search Tree (BST)
class BinaryTreeNode {
    int data;
    BinaryTreeNode left, right;
    BinaryTreeNode(int data) { this.data = data; }
}

class BST {
    BinaryTreeNode root;

    public void insert(int data) {
        root = insertRecursive(root, data);
    }

    private BinaryTreeNode insertRecursive(BinaryTreeNode root, int data) {
        if (root == null) return new BinaryTreeNode(data);
        if (data < root.data) root.left = insertRecursive(root.left, data);
        else root.right = insertRecursive(root.right, data);
        return root;
    }
}

//Min-Heap (Priority Queue)
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
minHeap.add(10);
minHeap.add(5);
int smallest = minHeap.poll(); // Returns 5

//Merge Sort (Divide & Conquer)
class Sorting {
    void mergeSort(int[] arr, int l, int r) {
        if (l < r) {
            int m = l + (r - l) / 2;
            mergeSort(arr, l, m);
            mergeSort(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }

    void merge(int[] arr, int l, int m, int r) {
        int n1 = m - l + 1;
        int n2 = r - m;
        int[] L = new int[n1];
        int[] R = new int[n2];
        System.arraycopy(arr, l, L, 0, n1);
        System.arraycopy(arr, m + 1, R, 0, n2);

        int i = 0, j = 0, k = l;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) arr[k++] = L[i++];
            else arr[k++] = R[j++];
        }
        while (i < n1) arr[k++] = L[i++];
        while (j < n2) arr[k++] = R[j++];
    }
}

//Binary Search (Iterative)
public int binarySearch(int[] arr, int target) {
    int left = 0, right = arr.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

//Graph BFS
class Graph {
    private Map<Integer, List<Integer>> adj = new HashMap<>();

    public void bfs(int start) {
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();

        visited.add(start);
        queue.add(start);

        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            System.out.print(vertex + " ");
            for (int neighbor : adj.getOrDefault(vertex, new ArrayList<>())) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.add(neighbor);
                }
            }
        }
    }
}