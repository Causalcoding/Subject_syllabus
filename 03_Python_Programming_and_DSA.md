# Python Programming and Data Structures & Algorithms — Interview Prep Syllabus

Every technical interview loop for **Data Scientist**, **Machine Learning Engineer**, and **AI Engineer** roles rests on this subject even when it isn't the headline topic. Data Scientists are usually judged on whether they can turn a notebook prototype into correct, efficient, production-safe Python (vectorized pandas/NumPy instead of nested loops, proper memory handling on large datasets, clean OOP for reusable pipelines). Machine Learning Engineers and AI Engineers, however, are typically **tested hardest on raw Data Structures & Algorithms** — companies expect them to whiteboard-code trees, graphs, DP, and pattern-based problems (two pointers, sliding window, heaps) under time pressure, because these skills map directly to building efficient training loops, low-latency inference services, feature pipelines, and custom data structures inside ML frameworks. In practice, an ML/AI Engineer candidate who cannot pass a LeetCode-medium-style DSA round will be rejected regardless of modeling knowledge, while a Data Scientist candidate is more likely to be probed on Python/pandas efficiency and OOP design than on graph algorithms. This syllabus covers both ends of that spectrum in full depth so you are ready for either kind of round.

## Table of Contents

1. [Python Language Fundamentals](#1-python-language-fundamentals)
2. [Python for Data / ML](#2-python-for-data--ml)
3. [Complexity Analysis](#3-complexity-analysis)
4. [Core Data Structures](#4-core-data-structures)
5. [Core Algorithms](#5-core-algorithms)
6. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## 1. Python Language Fundamentals

### 1.1 Data Types, Mutability vs Immutability, Dynamic Typing

Python is **dynamically typed** (variable types are checked at runtime, not compile time) and **strongly typed** (no implicit unsafe coercions like `"2" + 2`). A name is just a label bound to an object; the object — not the name — carries the type.

**Built-in types by mutability:**

| Immutable | Mutable |
|---|---|
| `int`, `float`, `complex`, `bool` | `list` |
| `str` | `dict` |
| `tuple` (but can hold mutable elements) | `set` |
| `frozenset` | `bytearray` |
| `bytes` | custom classes (by default) |

```python
x = 10          # x is bound to an int object
x = "hello"     # now bound to a str object — no type declaration needed, checked at runtime

a = (1, [2, 3])  # tuple is immutable, but the list inside it is mutable
a[1].append(4)   # legal! a is now (1, [2, 3, 4])
# a[0] = 99      # TypeError: 'tuple' object does not support item assignment
```

**Why it matters:** immutable objects are hashable (usable as dict keys / set members) and safe to share across threads/functions without defensive copies. Mutable objects enable in-place efficiency but create aliasing bugs.

```python
def append_item(lst, item):
    lst.append(item)   # mutates caller's list — no need to return

nums = [1, 2, 3]
append_item(nums, 4)
print(nums)   # [1, 2, 3, 4] — mutation visible to caller
```

**Common pitfall:** believing strings are "passed by reference and can be mutated." Strings are immutable — any `+=` on a string creates a *new* string object.

```python
s = "abc"
t = s
s += "d"
print(t)   # "abc" — t still points to the original object
```

**Interning & small-int caching:** CPython caches small integers (-5 to 256) and some string literals, which explains surprising `is` behavior (see §1.7).

### 1.2 Comprehensions, Generators, Iterators/Iterables Protocol

**Comprehensions** build a new collection in one expression — faster and more idiomatic than manual loops because the loop runs in C internally.

```python
squares = [x*x for x in range(10)]                     # list comprehension
evens_set = {x for x in range(20) if x % 2 == 0}        # set comprehension
sq_map = {x: x*x for x in range(5)}                     # dict comprehension
gen_expr = (x*x for x in range(10**8))                  # generator expression — lazy!
nested = [x*y for x in range(3) for y in range(3)]       # nested comprehension
matrix_t = [[row[i] for row in matrix] for i in range(len(matrix[0]))]  # transpose
```

Time complexity of a comprehension over `n` items is O(n) just like the equivalent loop — the speedup is a constant-factor win from avoiding repeated attribute lookups (`list.append`) in bytecode, not an asymptotic one.

**Iterators vs Iterables:**
- An **iterable** implements `__iter__()`, which returns an **iterator**.
- An **iterator** implements `__iter__()` (returning itself) and `__next__()`, raising `StopIteration` when exhausted.

```python
class CountUp:
    """Custom iterable + iterator."""
    def __init__(self, limit):
        self.limit = limit
    def __iter__(self):
        self.n = 0
        return self
    def __next__(self):
        if self.n >= self.limit:
            raise StopIteration
        self.n += 1
        return self.n

for i in CountUp(3):
    print(i)   # 1 2 3
```

**Generators** are a shorthand for building iterators using `yield`. Execution pauses at `yield` and resumes on the next `next()` call, preserving local state — this gives **O(1) memory** per step instead of materializing the whole sequence.

```python
def fibonacci(limit):
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b

for f in fibonacci(50):
    print(f)

# Generators support .send(), .close(), and delegation via `yield from`
def chain(*iterables):
    for it in iterables:
        yield from it

print(list(chain([1, 2], (3, 4), "ab")))  # [1, 2, 3, 4, 'a', 'b']
```

**When to use generators:** streaming large files/datasets, infinite sequences, pipelines where you don't need random access, memory-constrained ETL. **Pitfall:** a generator can only be iterated once; you cannot `len()` it or re-iterate without recreating it.

```python
def read_large_file(path):
    with open(path) as f:
        for line in f:          # file objects are themselves iterators — O(1) memory per line
            yield line.strip()
```

### 1.3 Functions: `*args`/`**kwargs`, Default Argument Gotcha, Closures, Decorators, Lambda

**`*args` / `**kwargs`** let a function accept a variable number of positional/keyword arguments.

```python
def summarize(*args, **kwargs):
    print("positional:", args)   # tuple
    print("keyword:", kwargs)    # dict

summarize(1, 2, 3, name="Ann", age=30)
# positional: (1, 2, 3)
# keyword: {'name': 'Ann', 'age': 30}

def combine(a, b, *args, sep="-", **kwargs):
    return f"{a}{sep}{b}" + "".join(str(x) for x in args)
```

Order matters: `def f(pos, *args, kw_only, **kwargs)` — positional-or-keyword, then varargs, then keyword-only, then var-keyword.

**Mutable default argument trap** — Python evaluates default argument values **once**, at function-definition time, not on every call.

```python
def add_item(item, bucket=[]):   # BUG: bucket is created ONCE, shared across all calls
    bucket.append(item)
    return bucket

print(add_item(1))   # [1]
print(add_item(2))   # [1, 2]  <-- surprise! Not [2]

# Correct pattern:
def add_item(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
```

**Closures** — an inner function that captures variables from an enclosing (non-global) scope, keeping them alive after the outer function returns.

```python
def make_multiplier(factor):
    def multiplier(x):
        return x * factor      # `factor` is captured in the closure
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(5), triple(5))    # 10 15
print(double.__closure__[0].cell_contents)  # 2
```

**Late-binding closure pitfall** in loops:

```python
funcs = [lambda: i for i in range(3)]
print([f() for f in funcs])   # [2, 2, 2] — all closures share the same `i` cell

# Fix: bind by default argument (evaluated at definition time)
funcs = [lambda i=i: i for i in range(3)]
print([f() for f in funcs])   # [0, 1, 2]
```

**Decorators** — higher-order functions that wrap another function/class to add behavior without modifying its source (open/closed principle). Use `functools.wraps` to preserve `__name__`/`__doc__`.

```python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.4f}s")
        return result
    return wrapper

@timer
def slow_add(a, b):
    time.sleep(0.1)
    return a + b

slow_add(2, 3)   # prints timing, returns 5
```

Decorators with arguments require an extra layer of nesting:

```python
def retry(times=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == times - 1:
                        raise
                    print(f"retry {attempt+1}: {e}")
        return wrapper
    return decorator

@retry(times=5)
def flaky_call():
    ...
```

Useful built-in decorators: `@staticmethod`, `@classmethod`, `@property`, `functools.lru_cache`, `functools.cached_property`.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def area(self):
        return 3.14159 * self._radius ** 2

    @area.setter
    def area(self, value):
        raise AttributeError("area is read-only")
```

**Lambda** — an anonymous, single-expression function. Cannot contain statements (no `if/else` blocks, only conditional expressions; no assignment via `=`, though `:=` walrus works in 3.8+).

```python
square = lambda x: x * x
sorted_by_age = sorted(people, key=lambda p: p["age"])
```

**Pitfall:** overusing lambdas hurts readability and they cannot be pickled — problematic for `multiprocessing`.

### 1.4 OOP in Python: Classes, Inheritance, MRO, Dunder Methods, ABCs, Composition vs Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow"
```

**Method Resolution Order (MRO)** — the order Python searches base classes for attributes/methods, computed via the **C3 linearization** algorithm to support cooperative multiple inheritance.

```python
class A:
    def greet(self): return "A"
class B(A):
    def greet(self): return "B"
class C(A):
    def greet(self): return "C"
class D(B, C):
    pass

print(D.__mro__)
# (<class D>, <class B>, <class C>, <class A>, <class object>)
print(D().greet())   # "B" — leftmost parent wins per C3 order
```

`super()` follows the MRO chain (not just "my direct parent"), which is what makes cooperative multiple inheritance work:

```python
class Base:
    def __init__(self):
        print("Base init")

class Mixin:
    def __init__(self):
        print("Mixin init")
        super().__init__()

class Combined(Mixin, Base):
    def __init__(self):
        print("Combined init")
        super().__init__()

Combined()
# Combined init
# Mixin init
# Base init
```

**Dunder (magic) methods** let user classes integrate with Python's built-in operators/protocols.

| Method | Purpose |
|---|---|
| `__init__`, `__new__` | construction |
| `__repr__`, `__str__` | debug repr / user-facing string |
| `__eq__`, `__lt__`, `__hash__` | comparisons, hashability |
| `__len__`, `__getitem__`, `__setitem__`, `__iter__` | container protocol |
| `__enter__`, `__exit__` | context manager protocol |
| `__add__`, `__radd__`, `__iadd__` | operator overloading |
| `__call__` | make instances callable |

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)
    def __hash__(self):
        return hash((self.x, self.y))

v = Vector(1, 2) + Vector(3, 4)
print(v)   # Vector(4, 6)
```

**Pitfall:** if you override `__eq__` you must also define `__hash__` (or explicitly set `__hash__ = None`), otherwise the class becomes unhashable in Python 3 — instances default to identity-based equality when neither is set, but defining only `__eq__` disables the default hash.

**Abstract Base Classes (ABC)** enforce an interface contract at instantiation time.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...
    @abstractmethod
    def perimeter(self) -> float: ...

class Rectangle(Shape):
    def __init__(self, w, h):
        self.w, self.h = w, h
    def area(self): return self.w * self.h
    def perimeter(self): return 2 * (self.w + self.h)

# Shape()  # TypeError: Can't instantiate abstract class Shape
```

**Composition vs Inheritance** — "favor composition over inheritance." Inheritance models *is-a* and tightly couples subclass to superclass implementation (fragile base class problem); composition models *has-a* and is more flexible/testable.

```python
# Inheritance (is-a) — tight coupling
class Car(Engine):
    ...

# Composition (has-a) — preferred for flexibility
class Car:
    def __init__(self, engine: Engine):
        self.engine = engine   # can swap engine implementations, mock in tests
    def start(self):
        return self.engine.ignite()
```

Rule of thumb: use inheritance for genuine polymorphic substitutability (Liskov substitution principle holds); use composition/mixins/protocols when you just want to reuse behavior.

### 1.5 Context Managers and Exception Handling

A **context manager** guarantees setup/teardown code runs (even on exceptions) via `__enter__`/`__exit__`.

```python
class ManagedFile:
    def __init__(self, path, mode):
        self.path, self.mode = path, mode
    def __enter__(self):
        self.file = open(self.path, self.mode)
        return self.file
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False   # False/None propagates exceptions; True swallows them

with ManagedFile("out.txt", "w") as f:
    f.write("hello")
# file.close() runs even if an exception is raised inside the block
```

Shorthand with `contextlib.contextmanager`:

```python
from contextlib import contextmanager

@contextmanager
def timer_ctx():
    import time
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"elapsed: {time.perf_counter() - start:.4f}s")

with timer_ctx():
    sum(range(10**6))
```

Other handy tools: `contextlib.suppress(Exception)`, `contextlib.ExitStack` for a dynamic number of context managers, `with a, b:` for multiple managers in one statement.

**Exception handling best practices:**

```python
try:
    value = risky_operation()
except (KeyError, IndexError) as e:      # catch specific exceptions, never bare `except:`
    logger.warning("lookup failed: %s", e)
    value = default
except Exception:
    logger.exception("unexpected error")  # logs full traceback
    raise                                  # re-raise unless you can truly handle it
else:
    logger.info("succeeded")               # runs only if no exception
finally:
    cleanup()                              # always runs
```

- Never use bare `except:` — it also catches `KeyboardInterrupt`/`SystemExit`.
- Prefer specific exception types over broad `Exception`.
- Use custom exceptions for domain errors: `class InsufficientFundsError(Exception): ...`
- Use `raise NewError("...") from original_error` to preserve the causal chain.
- EAFP ("easier to ask forgiveness than permission") is the Pythonic idiom over LBYL ("look before you leap"): prefer `try/except KeyError` over `if key in dict` when checking then acting is redundant work, because it avoids race conditions in concurrent code and is often faster in the common case.

### 1.6 Memory Management, Reference Counting, Garbage Collection, the GIL

CPython manages memory via:
1. **Reference counting** — every object has a count of references pointing to it (`sys.getrefcount`); when it drops to 0, memory is freed immediately.
2. **Generational garbage collector** (`gc` module) — reference counting alone cannot free **reference cycles** (e.g., two objects referencing each other). The cyclic GC periodically scans three "generations" of objects to detect and collect unreachable cycles.

```python
import gc, sys

a = []
b = [a]
a.append(b)          # a -> b -> a : a reference cycle
del a, b              # refcounts don't reach 0 because of the cycle
gc.collect()          # cyclic GC reclaims them
```

**The GIL (Global Interpreter Lock)** — CPython allows only **one thread to execute Python bytecode at a time**, even on multi-core machines. It exists to make CPython's memory management (refcounting) simple and thread-safe without fine-grained locks on every object.

**Implications:**

| Workload | Best tool | Why |
|---|---|---|
| CPU-bound (heavy computation) | `multiprocessing` | Each process has its own GIL/interpreter; true parallelism across cores |
| I/O-bound (network, disk, DB) | `threading` or `asyncio` | GIL is released during blocking I/O; threads/coroutines overlap wait time |
| Massive I/O concurrency (10k+ connections) | `asyncio` | Single-threaded event loop avoids per-thread OS overhead |

```python
# CPU-bound: threading does NOT help (GIL serializes bytecode execution)
import threading, time

def cpu_task(_=None):
    total = 0
    for i in range(10**7):
        total += i

start = time.time()
threads = [threading.Thread(target=cpu_task) for _ in range(4)]
[t.start() for t in threads]; [t.join() for t in threads]
print("threading:", time.time() - start)   # roughly same as sequential, or worse

# multiprocessing achieves real speedup on CPU-bound work
from multiprocessing import Pool
with Pool(4) as p:
    p.map(cpu_task, range(4))   # must be a top-level, picklable callable — NOT a lambda
```

Note: Python 3.13 introduced an experimental **free-threaded (no-GIL) build**; know this exists but treat classic CPython (with GIL) as the default assumption unless asked.

### 1.7 `is` vs `==`, Shallow vs Deep Copy, Mutable Default Argument Trap

`==` calls `__eq__` and checks **value equality**. `is` checks **identity** (same object in memory, i.e., same `id()`).

```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True  (same values)
print(a is b)   # False (different objects)

c = a
print(a is c)   # True (same object)

x = 256; y = 256
print(x is y)   # True — CPython caches small ints [-5, 256]
x = 257; y = 257
print(x is y)   # implementation-defined; often False for separately created literals
```

**Rule:** always use `is` for `None`/`True`/`False` comparisons (`if x is None`), and `==` for value comparisons. Never rely on integer/string caching behavior — it's a CPython implementation detail, not a language guarantee.

**Shallow vs deep copy:**

```python
import copy

original = [[1, 2], [3, 4]]

shallow = copy.copy(original)          # or original[:], list(original)
shallow[0].append(99)
print(original)   # [[1, 2, 99], [3, 4]] — inner lists are SHARED

deep = copy.deepcopy(original)
deep[0].append(100)
print(original)   # unaffected — deepcopy recursively copies nested objects
```

Shallow copy duplicates only the outer container; nested mutable objects are still shared references. Deep copy recursively duplicates everything, handling cycles safely, but is O(total nodes) in time/space vs O(n) for shallow copy.

**Mutable default argument trap** — already covered in §1.3; restated here as it is one of the most frequently asked "gotcha" questions.

### 1.8 Concurrency: Threading, Multiprocessing, Asyncio

```python
# threading — shared memory, GIL-limited for CPU work, good for blocking I/O
import threading
lock = threading.Lock()
counter = 0
def increment():
    global counter
    with lock:          # without the lock, this is a race condition
        counter += 1

# multiprocessing — separate processes, separate memory, true parallel CPU work
from multiprocessing import Process, Queue
def worker(q, n):
    q.put(n * n)
q = Queue()
procs = [Process(target=worker, args=(q, i)) for i in range(4)]
[p.start() for p in procs]; [p.join() for p in procs]

# asyncio — single-threaded cooperative concurrency for I/O-bound work
import asyncio
async def fetch(url):
    await asyncio.sleep(1)   # simulate non-blocking I/O
    return f"data from {url}"

async def main():
    results = await asyncio.gather(*(fetch(u) for u in ["a", "b", "c"]))
    print(results)

asyncio.run(main())
```

| Model | Memory | Parallelism | Overhead | Best for |
|---|---|---|---|---|
| `threading` | shared | concurrent, not parallel (GIL) | low | blocking I/O, simple concurrency |
| `multiprocessing` | isolated (IPC needed) | true parallel | high (process spawn, serialization) | CPU-bound number crunching |
| `asyncio` | shared, single thread | concurrent, cooperative | very low | high-fan-out I/O (web scraping, API calls, chat/LLM streaming) |

**Pitfalls:** forgetting locks around shared mutable state in threads (race conditions); forgetting that objects passed to `multiprocessing.Process`/`Pool` must be picklable; blocking calls inside `async def` functions (e.g., `time.sleep` instead of `await asyncio.sleep`) freeze the entire event loop.

### 1.9 Type Hints and Static Typing

Python's type hints (PEP 484 onward) are **optional, gradual, and not enforced at runtime** by the interpreter — they exist purely to let humans and static tools (`mypy`, `pyright`, IDEs) reason about and verify code correctness before it ships. This distinguishes Python's typing from statically-typed languages: `def f(x: int) -> int: return x` will happily accept and return a string at runtime with zero error.

```python
from typing import Optional, Union, List, Dict, Tuple

def greet(name: str, times: int = 1) -> str:
    return (name + " ") * times

def find_user(user_id: int) -> Optional[dict]:   # Optional[X] is shorthand for Union[X, None]
    return db.get(user_id)                        # may return None

def parse(value: Union[int, str]) -> int:          # Python 3.10+: `int | str` is equivalent
    return int(value)

def bucket_counts(items: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts
```

**Generics** — write container/utility classes and functions that are type-safe for *any* type `T`, using `TypeVar` and `Generic`.

```python
from typing import TypeVar, Generic, List

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

int_stack: Stack[int] = Stack()
int_stack.push(1)
# int_stack.push("a")   # mypy would flag this as an error; runtime allows it silently
```

**`Protocol`** (PEP 544) enables **structural typing** ("static duck typing") — a class satisfies a Protocol simply by implementing the right methods/attributes, with no explicit inheritance required. This models Python's actual duck-typing idiom in a way static tools can check.

```python
from typing import Protocol

class SupportsArea(Protocol):
    def area(self) -> float: ...

def total_area(shapes: list) -> float:
    return sum(s.area() for s in shapes)   # any object with .area() -> float works

class Circle:
    def __init__(self, r: float): self.r = r
    def area(self) -> float: return 3.14159 * self.r ** 2

total_area([Circle(2), Circle(3)])   # valid to mypy even though Circle never mentions SupportsArea
```

**mypy's role in production code:** type hints alone do nothing at runtime; `mypy` (or `pyright`) statically analyzes the codebase and flags type mismatches *before* deployment, functioning like a fast, cheap unit test that catches an entire class of bugs (wrong argument types, `None`-handling mistakes, mismatched return types) without executing any code. Many ML teams run `mypy --strict` as a CI gate on library/pipeline code, though it's often relaxed for exploratory notebook-adjacent scripts. Runtime enforcement (raising `TypeError` for bad input) requires a separate library like `pydantic` or `typeguard` — mypy checks are purely static.

**Pitfalls:**
- Type hints are hints, not contracts — nothing stops a caller from ignoring them and passing the wrong type at runtime; they cannot replace input validation at true trust boundaries (e.g., parsing request bodies).
- Overusing `Any` defeats the purpose — it silently disables checking for that value's entire usage chain.
- Self-referencing or forward-referenced classes/circular-import types need either string literal annotations (`def add(self, other: "Node") -> "Node":`) or `from __future__ import annotations` (defers all annotation evaluation, avoiding the issue entirely).
- `Optional[X]` describes the *type*, not a default — `def f(x: Optional[int] = None)` still requires you to handle the `None` case explicitly in the body; mypy will flag missing `None` checks if you dereference `x` without narrowing it first.

### Interview Questions

1. **Q: What's the difference between a list and a tuple, and when would you choose one over the other?**
   A: Lists are mutable, dynamically resizable, slightly larger in memory overhead, and not hashable. Tuples are immutable, fixed-size, hashable (if their contents are), and slightly faster to create/iterate. Use tuples for fixed records (e.g., coordinate pairs, function returns with multiple values, dict keys), lists for collections that grow/shrink or need in-place mutation.

2. **Q: Explain the mutable default argument bug and how to avoid it.**
   A: Default argument values are evaluated once at function-definition time and reused across every call. If the default is mutable (list/dict/set), all calls that don't pass that argument share the *same* object, causing accumulating state across calls. Fix: use `None` as sentinel default and create a fresh object inside the function body. (Full example in §1.3.)

3. **Q: What is a closure? Write a function `make_counter()` that returns a function which increments and returns a counter each time it's called.**
   ```python
   def make_counter():
       count = 0
       def counter():
           nonlocal count       # needed to rebind the outer variable
           count += 1
           return count
       return counter

   c = make_counter()
   print(c(), c(), c())   # 1 2 3
   ```
   Complexity: O(1) per call; O(1) space for the captured cell.

4. **Q: What's the difference between `@staticmethod`, `@classmethod`, and a regular instance method?**
   A: An instance method takes `self` and operates on an instance. A `@classmethod` takes `cls` instead and operates on the class itself (commonly used for alternative constructors, e.g. `from_json`). A `@staticmethod` takes neither — it's just a regular function namespaced inside the class for organizational purposes.

5. **Q: Explain Python's MRO and why `super()` isn't simply "call my parent."**
   A: MRO (Method Resolution Order) is the linear ordering of a class's ancestors computed by C3 linearization, ensuring a consistent, monotonic order even under diamond-shaped multiple inheritance. `super()` returns a proxy that follows this MRO chain from the *current* class forward — not literally "my direct base class" — which is what allows cooperative multiple inheritance (each class's `__init__` calling `super().__init__()` in a chain) to work correctly. See §1.4 example.

6. **Q: Write a decorator `@memoize` that caches results of a function by its arguments, then explain its time/space tradeoff.**
   ```python
   import functools

   def memoize(func):
       cache = {}
       @functools.wraps(func)
       def wrapper(*args):
           if args not in cache:
               cache[args] = func(*args)
           return cache[args]
       return wrapper

   @memoize
   def fib(n):
       return n if n < 2 else fib(n-1) + fib(n-2)

   print(fib(30))
   ```
   Complexity: reduces naive recursive Fibonacci from O(2^n) time to O(n) time, at the cost of O(n) extra space for the cache. (In practice, use `functools.lru_cache(maxsize=None)` instead of hand-rolling.)

7. **Q: What is the GIL, and why doesn't multithreading speed up CPU-bound Python code?**
   A: The GIL is a mutex that allows only one thread to execute Python bytecode at a time in CPython, so multiple threads doing pure computation still run essentially serially (with added context-switch overhead), yielding no parallel speedup — sometimes actually slower than single-threaded. Threads *do* help I/O-bound work because the GIL is released during blocking I/O calls. For true CPU parallelism, use `multiprocessing` (separate interpreters/processes) or push work into C-extension code (NumPy, etc.) which releases the GIL internally.

8. **Q: `is` vs `==` — what's the difference, and what's the correct idiom for `None` checks?**
   A: `==` invokes `__eq__` for value equality; `is` compares object identity (`id()`). Always use `is None` / `is not None` rather than `== None`, because `is` is unambiguous, faster (no method dispatch), and immune to a class that overrides `__eq__` in a way that breaks comparisons with `None`.

9. **Q: You have a class with instances stored in a `set`. What must you implement, and what's the pitfall if you only override `__eq__`?**
   A: You must implement both `__eq__` and `__hash__` consistently (equal objects must have equal hashes). If you override only `__eq__`, Python 3 automatically sets `__hash__` to `None`, making instances unhashable — `TypeError: unhashable type`. If you never override `__eq__`, default identity-based hashing/equality is used, which is always internally consistent but may not match your notion of "equal."

10. **Q: Given a JSON-like nested structure of lists/dicts, deep copy it manually without using `copy.deepcopy`. What's the complexity?**
    ```python
    def deep_copy(obj):
        if isinstance(obj, dict):
            return {deep_copy(k): deep_copy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_copy(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(deep_copy(item) for item in obj)
        return obj   # immutable / atomic values returned as-is
    ```
    Complexity: O(n) time and O(n) extra space where n is total number of nested elements (this naive version doesn't handle cycles — `copy.deepcopy` uses a memo dict keyed by `id()` to handle cyclic references safely, which is the key reason to prefer the standard library version in production).

11. **Q: Explain the difference between `threading`, `multiprocessing`, and `asyncio`, and pick the right one for: (a) scraping 10,000 URLs, (b) training-time matrix multiplication on CPU, (c) a Flask endpoint calling a slow external API.**
    A: (a) `asyncio` with `aiohttp` — thousands of concurrent I/O-bound requests, minimal per-connection overhead; threading would also work but with more OS thread overhead. (b) `multiprocessing` (or better, delegate to NumPy/BLAS which uses native threads outside the GIL) — CPU-bound, needs real parallel cores. (c) either `threading` (simple) or `asyncio` if the framework is async-native (e.g., FastAPI) — I/O-bound, GIL released during the network wait.

12. **Q: What happens when an exception is raised inside a `with` block? Walk through the `__exit__` contract.**
    A: Python calls `__exit__(exc_type, exc_val, exc_tb)` with the exception info before propagating it. If `__exit__` returns a falsy value (`None`/`False`), the exception propagates normally after cleanup runs. If it returns `True`, the exception is suppressed. This guarantees resources (files, locks, connections) are released regardless of whether the block succeeded or raised.

13. **Q: Why is `list.append` in a loop generally preferred over repeatedly doing `list = list + [item]`?**
    A: `list.append(item)` is amortized O(1) because Python over-allocates capacity on the underlying dynamic array. `list = list + [item]` creates a brand-new list and copies all existing elements every iteration, making the whole loop O(n²) instead of O(n).

14. **Q: Trace the output:**
    ```python
    def f(a, b=[]):
        b.append(a)
        return b
    print(f(1))
    print(f(2))
    print(f(3, []))
    print(f(4))
    ```
    A: `[1]`, `[1, 2]`, `[3]`, `[1, 2, 4]` — the default list persists across calls that don't supply `b` explicitly; passing an explicit empty list in the third call bypasses the shared default entirely.

15. **Q: Design (conceptually) a thread-safe counter class, and explain why a naive `count += 1` across threads is unsafe even though it looks atomic.**
    A: `count += 1` is actually three bytecode-level steps (load, increment, store); the GIL can switch threads between any of them, so two threads can both read the same old value and overwrite each other's increment (a race condition / lost update). Fix with `threading.Lock`:
    ```python
    import threading
    class SafeCounter:
        def __init__(self):
            self._value = 0
            self._lock = threading.Lock()
        def increment(self):
            with self._lock:
                self._value += 1
        @property
        def value(self):
            return self._value
    ```

16. **Q: What's the difference between `Optional[int]` and just `int` with a default value of `None`, from mypy's perspective?**
    A: They're related but distinct: `Optional[int]` (equivalently `int | None`) declares that the *value itself* may legitimately be `None` anywhere it's used, and mypy will require you to narrow/check for `None` before treating it as an `int` (e.g., calling `.bit_length()` on it). Simply giving a parameter a default of `None` without annotating `Optional` used to be silently inferred as optional by older mypy defaults, but under `--strict`/modern mypy this must be explicit — the annotation communicates intent to both mypy and human readers, the default value alone does not.

17. **Q: Explain structural typing via `Protocol` vs classic ABC-based nominal typing. When would you prefer one over the other?**
    A: `Protocol` checks conformance by *shape* (does the object have the right methods/attributes?) with zero coupling — any class satisfies it implicitly, including ones from third-party libraries you can't modify to inherit from your ABC. Classic ABCs (`abc.ABC` + `@abstractmethod`) enforce conformance by *explicit inheritance*, checked at instantiation time, and are better when you want a real, enforced contract within your own codebase (fails loudly if a concrete subclass forgets a method). Prefer `Protocol` for "duck typing with static safety" across independent codebases/libraries; prefer ABCs for a tightly-controlled internal class hierarchy.

18. **Q: Does adding type hints to a Python function make it run faster?**
    A: No — CPython's interpreter ignores type annotations entirely at runtime (they're stored in `__annotations__` and otherwise inert); there is no JIT-style specialization from hints alone in standard CPython. Any performance benefit from typing is indirect — catching bugs earlier via mypy, or enabling tools like Cython/mypyc to compile type-annotated code to faster C extensions.

19. **Q: Write a generic `first_or_default` function with type hints that returns the first element of a list or a provided default, and explain why `TypeVar` is needed instead of just typing both parameters as `Any`.**
    ```python
    from typing import TypeVar, List

    T = TypeVar("T")

    def first_or_default(items: List[T], default: T) -> T:
        return items[0] if items else default
    ```
    A: `TypeVar` links the types of `items`' elements, `default`, and the return value together — mypy can verify that `first_or_default([1,2,3], 0)` returns an `int` and would flag `first_or_default([1,2,3], "x")` as a type mismatch between the list's element type and the default. Typing everything as `Any` would silence all such checks, defeating the purpose of adding hints at all.

---

## 2. Python for Data / ML

### 2.1 NumPy: `ndarray`, Broadcasting, Vectorization, Views vs Copies, Memory Layout

NumPy's core object is the **`ndarray`** — a fixed-type, contiguous (or strided) block of memory described by `shape`, `dtype`, and `strides`. Because elements are homogeneous and stored contiguously, NumPy operations run in optimized C loops (or SIMD/BLAS) instead of Python's per-object bytecode loop.

```python
import numpy as np

a = np.array([1, 2, 3], dtype=np.int64)
print(a.shape, a.dtype, a.strides)   # (3,) int64 (8,)

# Vectorization: avoid explicit Python loops
n = 10**6
x = np.arange(n)

# Slow: pure Python loop — O(n) but with huge per-element interpreter overhead
result = [v * 2 for v in x]

# Fast: vectorized — same O(n) but executed in a tight C loop
result = x * 2
```

**Broadcasting** — NumPy's rule set for applying element-wise operations to arrays of different (but compatible) shapes without explicitly copying data.

Rules: compare shapes from the trailing dimension; two dimensions are compatible if they're equal, or one of them is 1 (it gets "stretched" virtually, no memory copy).

```python
a = np.ones((3, 4))
b = np.array([1, 2, 3, 4])       # shape (4,)
print((a + b).shape)             # (3, 4) — b is broadcast across each row

c = np.ones((3, 1))
print((a * c).shape)             # (3, 4) — c broadcast across columns

# Incompatible shapes raise:
# np.ones((3,4)) + np.ones((3,5))   -> ValueError: operands could not be broadcast together
```

**Views vs copies** — slicing an ndarray returns a **view** (shares the same underlying buffer); fancy indexing (boolean masks, integer arrays) returns a **copy**.

```python
a = np.arange(10)
b = a[2:5]          # view — shares memory
b[0] = 99
print(a)             # [ 0  1 99  3  4  5  6  7  8  9] — original mutated!

c = a[[1, 2, 3]]     # fancy indexing — copy
c[0] = -1
print(a)             # unaffected

print(a[2:5].base is a)          # True -> confirms it's a view
print(np.shares_memory(a, b))    # True
```

**Memory layout: C-order vs Fortran-order.** Row-major (`C`) stores rows contiguously; column-major (`F`) stores columns contiguously. Layout affects cache locality — iterating along the contiguous axis is much faster.

```python
a = np.zeros((1000, 1000), order='C')
# a.sum(axis=1) (row-wise) is faster in C-order because each row is contiguous in memory
a.flags['C_CONTIGUOUS']   # True
a.T.flags['C_CONTIGUOUS'] # False — transpose is a view with swapped strides, not a copy
```

**Common pitfalls:**
- Chained fancy indexing assignment silently creates a copy and doesn't mutate the original (`a[mask][0] = 5` does nothing to `a`; use `a[mask][0]` -> use `a[np.where(mask)[0][0]] = 5` or boolean assignment `a[mask] = 5` directly).
- Integer overflow with fixed-width dtypes (`np.int8` wraps silently rather than raising).
- Implicit upcasting to `float64` can blow up memory on large arrays — set `dtype` explicitly.

| Operation | Time Complexity | Notes |
|---|---|---|
| Element-wise op (`a + b`) | O(n) | vectorized, C-speed constant factor |
| Matrix multiply (`a @ b`, n×n) | O(n³) | BLAS/LAPACK still do O(n³) work — the speedup vs. a naive Python triple loop is a huge *constant-factor* win (cache blocking, SIMD, multithreading), not a better asymptotic algorithm. Galactic algorithms like Coppersmith–Winograd (~O(n^2.37)) exist in theory but aren't used in production BLAS due to enormous constants and poor cache behavior. |
| `np.sort` | O(n log n) | quicksort by default, `kind='stable'` for mergesort |
| Broadcasting | O(output size) | no extra copy of the smaller array |

### 2.2 Pandas: Series/DataFrame Internals, Indexing, GroupBy, Merge/Join, Performance

A **Series** is a 1-D labeled array (values + an `Index`). A **DataFrame** is a dict-like collection of Series sharing a common index, internally often backed by column-oriented NumPy blocks (`BlockManager`) grouped by dtype for efficiency.

```python
import pandas as pd

df = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["a", "b", "c"],
    "score": [90.5, 85.0, 77.2],
})
```

**`loc` vs `iloc`:**

| | `loc` | `iloc` |
|---|---|---|
| Selects by | label | integer position |
| Slicing end | inclusive | exclusive (like Python slicing) |
| Example | `df.loc[2, "score"]` | `df.iloc[2, 2]` |

```python
df = df.set_index("id")
df.loc[2, "name"]     # label-based -> "b"
df.iloc[1, 0]          # position-based -> "b"
df.loc[1:2]            # rows with labels 1 through 2, INCLUSIVE
df.iloc[0:2]           # rows at positions 0,1 — EXCLUSIVE of 2
```

**`groupby`-`apply`/`agg`** implements the split-apply-combine pattern.

```python
sales = pd.DataFrame({"region": ["E","E","W","W"], "amount": [100, 150, 200, 50]})

sales.groupby("region")["amount"].sum()
# region
# E    250
# W    250

sales.groupby("region").agg(total=("amount", "sum"), avg=("amount", "mean"))

# .apply lets you run an arbitrary function per group but is much slower than .agg/.transform
sales.groupby("region")["amount"].apply(lambda s: s.max() - s.min())
```

`.transform` returns a result broadcast back to the original shape (useful for group-wise normalization):

```python
sales["pct_of_region"] = sales["amount"] / sales.groupby("region")["amount"].transform("sum")
```

**Merge/join types:**

```python
left = pd.DataFrame({"key": [1, 2, 3], "val_l": ["a", "b", "c"]})
right = pd.DataFrame({"key": [2, 3, 4], "val_r": ["x", "y", "z"]})

pd.merge(left, right, on="key", how="inner")   # only keys 2,3
pd.merge(left, right, on="key", how="left")    # all of left, NaN for unmatched right
pd.merge(left, right, on="key", how="right")   # all of right
pd.merge(left, right, on="key", how="outer")   # union of both, NaN where missing
pd.merge(left, right, on="key", how="cross")   # cartesian product
left.join(right.set_index("key"), on="key")    # join is index-based merge
```

| Join type | Rows kept |
|---|---|
| inner | intersection of keys |
| left | all left rows |
| right | all right rows |
| outer | union of keys |
| cross | every combination |

**Performance pitfalls — `apply` vs vectorized ops:**

```python
# SLOW: row-wise Python-level apply, O(n) Python function calls
df["bonus"] = df["score"].apply(lambda s: s * 1.1 if s > 80 else s)

# FAST: vectorized with np.where, same O(n) but executed in C
import numpy as np
df["bonus"] = np.where(df["score"] > 80, df["score"] * 1.1, df["score"])
```

`.apply()` on a DataFrame/Series falls back to a Python-level loop under the hood (even though it looks vectorized), which is typically **10-100x slower** than native vectorized pandas/NumPy operations for simple arithmetic/comparisons. Reserve `.apply` for logic that genuinely can't be vectorized (complex conditional branching, calling external APIs per row).

**Handling large data — chunking, dtypes, categoricals:**

```python
# Chunked reading avoids loading a huge CSV entirely into memory
total = 0
for chunk in pd.read_csv("huge.csv", chunksize=100_000):
    total += chunk["amount"].sum()

# Downcasting dtypes shrinks memory footprint significantly
df["small_int_col"] = pd.to_numeric(df["small_int_col"], downcast="integer")
df["flag"] = df["flag"].astype("bool")

# Categorical dtype for low-cardinality string columns: big memory + speed win
df["region"] = df["region"].astype("category")   # stores as integer codes + a lookup table
print(df.memory_usage(deep=True))
```

Other large-data tools: `dtype` specified up front in `read_csv` (avoids two-pass type inference), `usecols` to skip unneeded columns, `pyarrow` backend (`pd.read_parquet`, `dtype_backend="pyarrow"`) for better memory efficiency and native support for nulls, or moving to `Dask`/`polars`/chunked processing when data exceeds RAM.

### 2.3 Writing Efficient, Production-Quality Python for Data Pipelines

Principles that interviewers probe for in senior DS/MLE/AI Engineer rounds:

- **Vectorize first, loop last** — reach for pandas/NumPy vector ops before writing `for` loops over rows.
- **Fail fast and explicitly** — validate schema/dtypes at pipeline boundaries (`pydantic`, `pandera`, assertions) instead of letting bad data silently propagate as `NaN`.
- **Idempotency** — a pipeline stage run twice on the same input should produce the same output (critical for retries in orchestrators like Airflow).
- **Streaming/chunked processing** for datasets larger than memory; avoid `df.iterrows()` (notoriously slow, boxes every row into a Series) — prefer `itertuples()`, vectorized ops, or `.to_numpy()` for tight inner loops.
- **Logging, not printing** — use the `logging` module with levels, not `print`, so production log volume is controllable.
- **Type hints** for public functions (`def clean(df: pd.DataFrame) -> pd.DataFrame:`) — aids static analysis (`mypy`) and documents intent.
- **Separate I/O from logic** — pure transformation functions are unit-testable without touching disk/network; wrap I/O at the edges.
- **Config over hardcoding** — paths, thresholds, and credentials belong in config/env vars, not literals in code.

```python
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def clean_scores(df: pd.DataFrame, min_score: float = 0.0) -> pd.DataFrame:
    """Pure function: no I/O, easily unit-tested."""
    before = len(df)
    df = df.loc[df["score"] >= min_score].copy()
    logger.info("dropped %d invalid rows", before - len(df))
    df["score"] = df["score"].astype("float32")   # explicit dtype, memory-conscious
    return df
```

### 2.4 Unit Testing in Python: pytest, Fixtures, Parametrize, Mocking

Data Scientists and ML Engineers are increasingly tested on testing discipline because ML pipelines fail silently far more often than they crash loudly: a schema change upstream, a subtly wrong groupby, or a feature-engineering off-by-one can quietly corrupt a model's training data for weeks before anyone notices in a metric. Interviewers probe whether you write **regression tests for data transformations** (not "does the model get 95% accuracy," which is not a unit-testable assertion) and whether you can isolate code from slow/nondeterministic dependencies (databases, APIs, the filesystem) using mocks.

**pytest basics** — test files/functions discovered by the `test_*.py` / `*_test.py` and `test_*`/`Test*` naming convention; plain `assert` statements (no `self.assertEqual` boilerplate needed, unlike `unittest`).

```python
# test_clean_scores.py
import pandas as pd
from mymodule import clean_scores

def test_clean_scores_drops_rows_below_threshold():
    df = pd.DataFrame({"score": [10, 60, 90]})
    result = clean_scores(df, min_score=50)
    assert len(result) == 2
    assert (result["score"] >= 50).all()

def test_clean_scores_empty_input():
    df = pd.DataFrame({"score": []})
    result = clean_scores(df, min_score=50)
    assert result.empty
```

Run with `pytest -v test_clean_scores.py`; `pytest` auto-discovers all matching files under the current directory tree.

**Fixtures** (`@pytest.fixture`) provide reusable setup (and teardown, via `yield`), injected into test functions by argument name — pytest resolves dependencies automatically.

```python
import pytest

@pytest.fixture
def sample_df():
    return pd.DataFrame({"score": [10, 60, 90], "region": ["E", "W", "E"]})

@pytest.fixture
def db_connection():
    conn = create_connection()   # setup
    yield conn
    conn.close()                  # teardown, runs even if the test fails

def test_with_fixture(sample_df):
    assert len(sample_df) == 3
```

Fixture `scope` (`function` [default], `class`, `module`, `session`) controls how often expensive setup reruns — e.g. `scope="session"` for a Spark session or test database created once for the whole test run.

**`@pytest.mark.parametrize`** runs the same test body against multiple input/expected-output pairs, avoiding copy-pasted near-duplicate tests.

```python
@pytest.mark.parametrize("min_score,expected_len", [
    (0, 3),
    (50, 2),
    (100, 0),
])
def test_clean_scores_parametrized(sample_df, min_score, expected_len):
    assert len(clean_scores(sample_df, min_score)) == expected_len
```

**Mocking with `unittest.mock`** — replace slow/external/nondeterministic dependencies (network calls, DB queries, `datetime.now()`, random sampling) with controllable stand-ins so tests are fast, deterministic, and isolated.

```python
from unittest.mock import Mock, patch

def fetch_price(api_client, symbol: str) -> float:
    return api_client.get_price(symbol)

def test_fetch_price_uses_client():
    mock_client = Mock()
    mock_client.get_price.return_value = 42.0
    assert fetch_price(mock_client, "AAPL") == 42.0
    mock_client.get_price.assert_called_once_with("AAPL")   # verify the call happened

# patch() replaces a name in the target module's namespace for the duration of the test
@patch("mymodule.requests.get")
def test_external_api_call(mock_get):
    mock_get.return_value.json.return_value = {"price": 100}
    result = fetch_remote_price("AAPL")
    assert result == 100
```

**Complexity/cost framing an interviewer expects:** tests should run in milliseconds-to-low-seconds total, not minutes — anything hitting a real network/DB belongs in a separate, smaller "integration test" suite, not the unit-test suite that runs on every commit.

**Pitfalls:** over-mocking (mocking so much that the test just re-asserts the mock's own configured return value, testing nothing real); testing implementation details instead of behavior (breaks on harmless refactors); using production data in tests (non-deterministic, slow, potentially sensitive) instead of small synthetic fixtures; forgetting `assert_called_with`-style verification, so a mock silently absorbs a call that should have failed.

### 2.5 Profiling and Performance Debugging

Before optimizing anything, **profile first** — intuition about where Python code spends time is frequently wrong, and optimizing the wrong function wastes effort while leaving the real bottleneck untouched.

**`cProfile`** — the standard library's deterministic profiler; instruments every function call and reports call counts, cumulative time, and per-call time.

```python
import cProfile
import pstats

def slow_function():
    total = 0
    for i in range(10**6):
        total += i ** 2
    return total

cProfile.run("slow_function()", "profile_output")
stats = pstats.Stats("profile_output")
stats.sort_stats("cumulative").print_stats(10)   # top 10 functions by cumulative time
```

Or from the command line without touching the source: `python -m cProfile -s cumulative myscript.py`.

**`line_profiler`** (third-party) drills down from function-level to **line-level** timing — essential when `cProfile` tells you *which function* is slow but not *which line inside it*. Decorate the target function with `@profile` and run `kernprof -l -v script.py`.

**`memory_profiler`** (third-party) tracks memory consumption line-by-line similarly (`@profile` decorator + `python -m memory_profiler script.py`), or programmatically via `memory_usage()` — critical for diagnosing why a pandas pipeline's memory footprint balloons well past the raw data size.

**Diagnosing bottleneck type** (the actual interview-relevant skill — knowing *which tool fixes which problem*):

| Symptom | Likely cause | Confirm with | Fix |
|---|---|---|---|
| High CPU (~100% on one core), profiler time concentrated in tight computation loops | CPU-bound | `cProfile`, `top`/Task Manager showing one core pegged | vectorize (NumPy), `multiprocessing`, native extensions (Cython/Numba) |
| Low CPU utilization, high wall-clock time, profiler time sits in `read`/`recv`/`socket`/`requests` calls | I/O-bound | `cProfile` cumulative time in I/O calls; CPU usage graph mostly idle | `asyncio`, `threading`, batching requests |
| Multi-threaded CPU-bound code shows no speedup (or a slowdown) as threads increase | GIL-bound | Compare `threading` vs `multiprocessing` wall time for the same CPU-bound task; near-identical or worse threading time signals the GIL | switch to `multiprocessing`, or push work into a GIL-releasing C extension (NumPy/BLAS ops release the GIL internally) |

```python
# Quick microbenchmark comparison without a full profiler
import timeit
print(timeit.timeit("sum(range(1000))", number=10_000))
print(timeit.timeit("[x for x in range(1000)]", number=10_000))
```

**Pitfalls:** profiling overhead itself distorts absolute numbers (`cProfile` adds real per-call overhead, more pronounced with many small/cheap function calls — trust relative rankings, not absolute times, or switch to a sampling profiler like `py-spy` for lower overhead on production processes); profiling toy/unrepresentative inputs instead of realistic data sizes/distributions; optimizing before profiling ("premature optimization is the root of all evil" — Knuth); forgetting that vectorized NumPy/pandas calls show up as one profiler entry even though they do O(n) work internally, so a profiler alone won't tell you if *that* line is itself inefficient relative to a better algorithm — pair profiling with complexity reasoning.

### 2.6 Virtual Environments and Dependency Management

Every ML/AI Engineer interview loop eventually asks a practical setup question ("how do you manage Python dependencies for a project," "what happens if two projects need different pandas versions") — this is about **isolation and reproducibility**, not a deep CS topic, but candidates who've never set one up consistently stumble on it.

**`venv`** (standard library) creates an isolated Python environment with its own `site-packages`, so installing packages for one project doesn't affect the system Python or other projects.

```bash
python -m venv .venv                 # create an isolated environment in ./.venv
source .venv/bin/activate            # macOS/Linux
.venv\Scripts\activate                # Windows

pip install pandas numpy scikit-learn
pip freeze > requirements.txt         # snapshot exact installed versions
pip install -r requirements.txt       # reproduce the environment elsewhere
deactivate
```

**`requirements.txt` vs lockfiles:** `pip freeze` output pins exact versions of everything currently installed, but it's a flat snapshot — it doesn't distinguish direct dependencies (what your project actually imports) from transitive ones (dependencies of dependencies), doesn't record package hashes for integrity verification, and can drift silently if regenerated on a different platform/Python version. Modern tools generate true **lockfiles** (`poetry.lock`, `uv.lock`, or `pip-tools`' compiled `requirements.txt`) that pin exact versions *and* cryptographic hashes for the full transitive dependency graph, computed by a real dependency resolver — guaranteeing byte-for-byte reproducible installs across machines and time, which `pip freeze` alone cannot.

```bash
# Poetry: declares dependencies in pyproject.toml, resolves + locks in poetry.lock
poetry add pandas
poetry install                        # installs exactly what's in poetry.lock

# uv: a newer, Rust-based package/environment manager, drop-in-compatible with pip,
# dramatically faster dependency resolution and installs; rapidly adopted in ML tooling
uv venv
uv pip install -r requirements.txt
uv add pandas                         # manages pyproject.toml + uv.lock, like poetry
```

**Conda** is worth naming distinctly: unlike `venv`/`pip` (Python-package-only), conda manages non-Python binary dependencies too (CUDA toolkits, MKL, compiled C/Fortran libraries) — historically why it dominates ML/scientific-Python setups needing GPU libraries, though `uv`/`pip` plus prebuilt wheels have closed much of that gap for many packages.

**Pitfalls:** forgetting to activate the virtual environment and unintentionally installing into the global/system Python; not pinning versions at all ("works on my machine" bugs from an unpinned transitive dependency upgrading); committing the `.venv`/`venv` directory to version control (should always be `.gitignore`d — commit only the manifest/lockfile, not the environment itself); mixing conda and pip installs in the same environment without care, which can produce inconsistent dependency resolution.

### Interview Questions

1. **Q: What makes NumPy operations fast compared to native Python loops?**
   A: Contiguous, homogeneously-typed memory layout lets NumPy dispatch to compiled C/Fortran loops (and SIMD/BLAS) that avoid the per-element overhead of Python's dynamic dispatch, object boxing, and reference counting. A Python loop pays interpreter overhead per iteration; a vectorized NumPy op pays it once for the whole array call.

2. **Q: Explain broadcasting with an example, and state the compatibility rule.**
   A: Broadcasting lets arrays of different shapes be combined element-wise without explicit copying, by virtually stretching size-1 dimensions. Two dimensions (compared from the trailing edge) are compatible if equal or one is 1. Example: `(3,4) + (4,)` broadcasts the `(4,)` array across all 3 rows. `(3,4) + (3,1)` broadcasts the column across all 4 columns.

3. **Q: When does slicing a NumPy array return a view vs a copy? Why does this matter?**
   A: Basic slicing (`a[1:3]`) returns a view sharing memory with the original — mutating it mutates the source. Fancy indexing (boolean masks or integer-array indices, e.g. `a[[0,2]]` or `a[a>5]`) always returns a copy. This matters because relying on view semantics when you actually got a copy (or vice versa) is a common silent-bug source; check with `np.shares_memory(a, b)` or `b.base is a`.

4. **Q: Write a function to compute row-wise z-score normalization of a 2D NumPy array without using explicit loops.**
   ```python
   import numpy as np
   def row_zscore(a: np.ndarray) -> np.ndarray:
       mean = a.mean(axis=1, keepdims=True)
       std = a.std(axis=1, keepdims=True)
       return (a - mean) / std
   ```
   Complexity: O(n·m) time for an n×m matrix, single pass plus broadcasting, O(n·m) space for the output (O(n) extra for the mean/std vectors).

5. **Q: What's the difference between `df.loc` and `df.iloc`? Give an example where confusing them causes a bug.**
   A: `loc` indexes by label (inclusive slicing), `iloc` by integer position (exclusive slicing, like Python lists). Bug example: after filtering a DataFrame (`df = df[df.score > 50]`), the index is no longer 0..n-1 contiguous; `df.iloc[0]` still gives the first remaining row by position, but `df.loc[0]` may raise `KeyError` or return an unrelated row if label `0` still exists elsewhere. Always `reset_index(drop=True)` after filtering if you need positional access afterward.

6. **Q: Why is `df.apply(lambda row: ..., axis=1)` often a performance red flag, and what should you do instead?**
   A: `axis=1` apply iterates row-by-row in Python, reconstructing a Series object per row — this is effectively a disguised Python loop and loses pandas' vectorization advantage, often 10-100x slower than an equivalent vectorized expression using `np.where`, boolean masks, or built-in vectorized string/datetime methods. Reserve `.apply` for logic that truly cannot be vectorized.

7. **Q: Explain the four main `pd.merge` join types and what happens to unmatched rows in each.**
   A: `inner` keeps only keys present in both frames (unmatched rows dropped from both sides); `left` keeps all left rows, filling unmatched right columns with `NaN`; `right` is the mirror of left; `outer` keeps the union of keys, filling `NaN` on whichever side lacks a match. (Full table in §2.2.)

8. **Q: How would you process a 50GB CSV file that doesn't fit in memory using pandas?**
   A: Use `pd.read_csv(path, chunksize=N)` to stream fixed-size chunks, aggregate incrementally (e.g., running sums/counts) rather than concatenating chunks into one giant frame; specify `dtype` up front to avoid pandas' expensive type-inference pass and to reduce memory per chunk; convert low-cardinality string columns to `category`; consider `usecols` to drop unneeded columns before loading; for genuinely bigger-than-RAM workloads, switch to `Dask`, `polars`, or `pyarrow`-backed / out-of-core processing, or push aggregation into a database/Spark job.

9. **Q: What is the memory benefit of pandas' `category` dtype, and when should you avoid it?**
   A: Categorical dtype stores each unique value once in a lookup table and stores integer codes in the column, drastically cutting memory for low-cardinality string columns (e.g., "region" with 5 unique values across 10M rows) and speeding up groupby/comparison operations. Avoid it for high-cardinality columns (e.g., unique IDs) where the codes-plus-lookup overhead approaches or exceeds the cost of plain strings, and be cautious with categories in arithmetic/serialization contexts that expect plain strings.

10. **Q: Given a DataFrame of transactions with columns `user_id`, `amount`, write vectorized pandas code to compute each user's amount as a percentage of their own total spend.**
    ```python
    df["pct_of_user_total"] = df["amount"] / df.groupby("user_id")["amount"].transform("sum")
    ```
    Complexity: O(n) — a single groupby pass to compute sums, and `transform` broadcasts the result back to every row without a Python-level loop, unlike `.apply`.

11. **Q: What's wrong with this code, and how would you fix it for large DataFrames?**
    ```python
    total = 0
    for i, row in df.iterrows():
        total += row["amount"] * row["quantity"]
    ```
    A: `iterrows()` reconstructs a full `Series` (with type coercion across the row's mixed dtypes) for every row — extremely slow on large data. Fix: `total = (df["amount"] * df["quantity"]).sum()` — fully vectorized, O(n) with a tiny constant factor versus `iterrows`' large per-row overhead.

12. **Q: Explain C-contiguous vs Fortran-contiguous memory layout and why it can affect performance.**
    A: C-order stores rows contiguously in memory (row-major); Fortran-order stores columns contiguously (column-major). Operations that iterate along the contiguous axis exploit CPU cache locality and are faster; iterating along the non-contiguous axis causes cache misses/strided access. `.T` (transpose) doesn't copy data — it returns a view with swapped strides, which is why transposing then summing rows can be slower than summing columns directly, depending on the underlying layout.

13. **Q: How would you validate that an incoming DataFrame matches an expected schema before running a pipeline on it, in a production-quality way?**
    A: Use an explicit schema-validation layer (e.g., `pandera` schemas or `pydantic` models for row-level records) that checks column presence, dtypes, null constraints, and value ranges up front, raising a clear, typed error immediately rather than letting malformed data silently produce `NaN`/wrong results downstream. This "fail fast" approach is critical in production pipelines where silent data corruption is far more costly than an early, loud failure.

14. **Q: Why do interviewers ask Data Scientists/MLEs about unit testing, given that "the model's accuracy is the real test"?**
    A: Model accuracy tells you nothing about whether a specific data-cleaning function, feature-engineering transform, or pipeline stage is correct in isolation — a model can score well on a benchmark while a silent bug in feature computation degrades production performance in ways that never surface in a notebook. Unit tests target the deterministic, testable pieces (parsing, joins, feature transforms, business-rule filters) so regressions are caught by CI within seconds, long before a full retrain/evaluation cycle would ever reveal the problem.

15. **Q: Write a pytest test (with a fixture and `parametrize`) for a function `is_valid_email(s: str) -> bool`.**
    ```python
    import pytest
    from mymodule import is_valid_email

    @pytest.fixture
    def valid_emails():
        return ["a@b.com", "user.name@sub.domain.org"]

    def test_valid_emails_pass(valid_emails):
        assert all(is_valid_email(e) for e in valid_emails)

    @pytest.mark.parametrize("bad_email", ["", "no-at-sign", "a@", "@b.com"])
    def test_invalid_emails_fail(bad_email):
        assert is_valid_email(bad_email) is False
    ```
    A: The fixture supplies reusable known-good input; `parametrize` runs the same assertion across several known-bad inputs without duplicating test bodies — a single failing case shows up as its own labeled failure rather than aborting a combined test early.

16. **Q: You're testing a function that calls an external pricing API. Why shouldn't the unit test make a real network call, and how do you avoid it?**
    A: Real network calls make tests slow, flaky (subject to outages/rate limits/latency), non-deterministic (prices change), and dependent on network access in CI — none of which the test is actually trying to verify (the test should verify *your code's logic*, not the API's availability). Fix with `unittest.mock.patch` to replace the HTTP call with a `Mock` returning a fixed, known response, so the test exercises your function's handling logic deterministically in microseconds.

17. **Q: You suspect a data pipeline is slow because of a particular pandas transformation, but you're not sure which line. How do you find out precisely, and what's the risk of just guessing?**
    A: Use `line_profiler` (`@profile` + `kernprof -l -v script.py`) to get per-line timing within the suspect function, rather than guessing — profiler-informed optimization targets the actual bottleneck, while guessing risks spending effort rewriting an already-fast line while the true hot line (often a hidden `.apply` or an accidental copy) goes untouched.

18. **Q: A colleague says "threading didn't speed up our CPU-bound feature-extraction job at all, so I switched to multiprocessing and got a 4x speedup on our 4-core machine." What does this tell you was the bottleneck, and why does the fix work?**
    A: This is a classic GIL-bound symptom: threading added concurrency but not parallelism because the GIL serializes bytecode execution for pure-Python CPU work, so wall-clock time stayed flat (or worsened slightly from context-switch overhead). `multiprocessing` sidesteps this by running each worker in its own OS process with its own interpreter and GIL, achieving genuine parallel CPU execution across the 4 cores — hence the near-linear 4x speedup.

19. **Q: What's the practical difference between `pip freeze > requirements.txt` and a lockfile produced by `poetry`/`uv`/`pip-tools`?**
    A: `pip freeze` dumps whatever happens to be installed right now — exact versions, but no hashes, no direct-vs-transitive distinction, and no guarantee it was resolved the same way on a different platform/Python version. A real lockfile is the output of a deterministic dependency resolver that records exact versions **and** cryptographic hashes for the entire transitive dependency graph, so installs are verifiably reproducible byte-for-byte across machines and time — meaningfully stronger than a `pip freeze` snapshot.

---

## 3. Complexity Analysis

### 3.1 Big-O, Big-Theta, Big-Omega Notation

Asymptotic notation describes how an algorithm's resource usage (time or space) grows as input size `n → ∞`, ignoring constant factors and lower-order terms.

| Notation | Meaning | Bound type |
|---|---|---|
| **O(f(n))** | Big-O | **Upper bound** — algorithm grows *no faster than* f(n) (worst case, most commonly cited) |
| **Ω(f(n))** | Big-Omega | **Lower bound** — algorithm grows *at least as fast as* f(n) |
| **Θ(f(n))** | Big-Theta | **Tight bound** — grows *exactly* at rate f(n) (both O and Ω hold) |

```python
def linear_search(arr, target):
    for i, val in enumerate(arr):     # O(n) worst case (target at end/absent)
        if val == target:
            return i                  # Ω(1) best case (target at index 0)
    return -1
# Overall: O(n) worst case, Ω(1) best case — NOT Θ(n) because best and worst differ.
```

Common growth-rate hierarchy (slowest to fastest growth):

```
O(1) < O(log n) < O(√n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2ⁿ) < O(n!)
```

### 3.2 Best/Average/Worst Case, Amortized Analysis

- **Best case** — the most favorable input (e.g., already-sorted array for insertion sort → O(n)).
- **Average case** — expected performance over a distribution of inputs (e.g., quicksort's O(n log n) average with random pivots).
- **Worst case** — the most adversarial input (e.g., quicksort's O(n²) worst case with poor pivot choices on already-sorted data using naive pivot selection). Interviews almost always want **worst case** unless stated otherwise.

**Amortized analysis** — bounds the *average* cost per operation over a sequence of operations, even if individual operations occasionally cost more, using techniques like the **aggregate method** or **accounting method**.

Classic example: **dynamic array (Python `list`) `append`**. Occasionally, appending triggers a resize (allocate new buffer ~1.125-2x larger, copy all elements — O(n)), but resizes happen exponentially less often as the array grows, so the *amortized* cost per `append` is **O(1)**.

```python
# Aggregate method: sum of costs for n appends with doubling resizes:
# 1 + 2 + 4 + ... + n = O(2n) total for n appends => O(1) amortized per append
lst = []
for i in range(n):
    lst.append(i)   # O(1) amortized, though a few individual calls are O(n)
```

Another classic amortized example: incrementing a binary counter — most flips are O(1), carries are rare, giving O(1) amortized per increment.

### 3.3 Space Complexity

Space complexity measures **additional memory** used relative to input size, usually excluding the input itself unless it's being copied/modified.

```python
def sum_array(arr):        # O(1) extra space — just a running total
    total = 0
    for x in arr:
        total += x
    return total

def reverse_copy(arr):     # O(n) extra space — new list of same size
    return arr[::-1]

def fib_recursive(n):      # O(n) space due to call stack depth, despite O(1) "extra variables"
    if n < 2:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)
```

**Pitfall:** forgetting that recursion consumes **call-stack space** proportional to recursion depth — a recursive solution that looks O(1) "auxiliary space" in variables can still be O(n) or O(log n) space once the stack is counted. Always state both time and space complexity, and mention whether space includes the recursion stack.

### Interview Questions

1. **Q: What's the difference between O, Ω, and Θ?**
   A: O is an upper bound (worst case, "grows no faster than"), Ω is a lower bound ("grows at least as fast as"), Θ is a tight bound where both hold simultaneously (the algorithm's growth rate is *exactly* that order). Most interview answers default to Big-O because it captures worst-case guarantees, which matter most for reliability.

2. **Q: Is quicksort O(n log n)?**
   A: Its *average* case is Θ(n log n), but its *worst* case is O(n²) (e.g., already-sorted input with a naive "always pick first/last element" pivot strategy). Randomized pivot selection or median-of-three mitigates but doesn't eliminate the theoretical worst case. In interviews, always state which case you're describing.

3. **Q: Explain amortized O(1) for Python list `append` using the aggregate method.**
   A: See §3.2 — total cost of n appends with geometric (doubling) resizing sums to O(n) overall (a convergent geometric series of resize costs), so dividing by n operations gives O(1) amortized cost per append, even though individual resize operations cost O(n).

4. **Q: What is the time and space complexity of this function?**
   ```python
   def contains_duplicate(nums):
       seen = set()
       for n in nums:
           if n in seen:
               return True
           seen.add(n)
       return False
   ```
   A: Time: O(n) — one pass, O(1) average-case hash set lookup/insert. Space: O(n) worst case (all elements unique, stored in `seen`).

5. **Q: Compare the time complexity of checking membership in a Python `list` vs a `set`.**
   A: `list`: O(n) — linear scan. `set`: O(1) average case (hash table lookup), O(n) worst case under pathological hash collisions (rare in practice, and Python's set uses open addressing with good hash distribution for common types).

6. **Q: A recursive function computes Fibonacci without memoization: `fib(n) = fib(n-1) + fib(n-2)`. Derive its time complexity.**
   A: Each call branches into two calls, forming a binary recursion tree of depth n, giving roughly O(2ⁿ) time (more precisely Θ(φⁿ) where φ is the golden ratio, but O(2ⁿ) is the standard interview-accepted bound). Space is O(n) due to the maximum recursion depth (the deepest single path down the tree), not O(2ⁿ), since only one path's stack frames exist at a time.

7. **Q: Give an example where an algorithm has O(1) space complexity in "extra variables" but is not truly O(1) once you account for recursion.**
   A: A recursive binary search: at each call only a few variables (`lo`, `hi`, `mid`) are used, seemingly O(1), but the recursion stack itself grows to depth O(log n), so true space complexity is O(log n), not O(1). An iterative binary search achieves genuine O(1) space.

8. **Q: What is the worst-case time complexity of inserting into a Python dict, and why can it theoretically degrade to O(n)?**
   A: Average case O(1) due to hashing; worst case O(n) if many keys hash to the same bucket (hash collisions), forcing a linear probe/chain traversal. Python mitigates this with a well-designed hash function and (for adversarial input on custom objects) hash randomization for strings (`PYTHONHASHSEED`), but the theoretical worst case remains O(n) per operation.

9. **Q: Two algorithms: Algorithm A is O(n log n) always; Algorithm B is O(n) average but O(n²) worst case. Which would you pick for a production system, and why?**
   A: Generally prefer Algorithm A for production systems where predictable, guaranteed performance matters (e.g., merge sort over quicksort for a real-time system with adversarial or attacker-controlled input, since O(n²) worst case could be a denial-of-service vector). Algorithm B may be preferred when average-case throughput dominates and worst-case inputs are provably rare/impossible (e.g., internal batch jobs on random data), and its lower constant factor matters more.

---

## 4. Core Data Structures

### 4.1 Arrays vs Linked Lists (Singly/Doubly)

**Arrays** store elements in contiguous memory, enabling O(1) random access via index arithmetic (`address = base + index * element_size`) but O(n) insertion/deletion in the middle (must shift elements).

**Linked lists** store elements as nodes with pointers to the next (and previous, for doubly-linked) node. O(1) insertion/deletion at a known position (just pointer rewiring), but O(n) access by index (must traverse from the head).

```python
class Node:
    def __init__(self, val, nxt=None, prev=None):
        self.val, self.next, self.prev = val, nxt, prev

class DoublyLinkedList:
    def __init__(self):
        self.head = self.tail = None
        self.size = 0

    def push_front(self, val):                  # O(1)
        node = Node(val, nxt=self.head)
        if self.head:
            self.head.prev = node
        self.head = node
        if self.tail is None:
            self.tail = node
        self.size += 1

    def pop_front(self):                         # O(1)
        if not self.head:
            raise IndexError("pop from empty list")
        val = self.head.val
        self.head = self.head.next
        if self.head:
            self.head.prev = None
        else:
            self.tail = None
        self.size -= 1
        return val
```

| Operation | Array | Singly Linked List | Doubly Linked List |
|---|---|---|---|
| Access by index | O(1) | O(n) | O(n) |
| Insert/delete at head | O(n) | O(1) | O(1) |
| Insert/delete at tail | O(1) amortized | O(n) (unless tail pointer kept) | O(1) |
| Insert/delete in middle (position known via pointer) | O(n) (shift) | O(1) | O(1) |
| Search | O(n) | O(n) | O(n) |
| Extra memory per element | none | 1 pointer | 2 pointers |
| Cache locality | excellent (contiguous) | poor (scattered) | poor (scattered) |

**When to use:** arrays/`list` for random access and cache-friendly iteration (the default choice in Python); linked lists for frequent insert/delete at arbitrary positions when you already hold a node reference (e.g., LRU cache internals, implementing deques/queues from scratch).

### 4.2 Stacks, Queues, Deques

**Stack** — LIFO (last-in-first-out). Operations: `push`, `pop`, `peek`, all O(1).

```python
stack = []
stack.append(1); stack.append(2); stack.append(3)
print(stack.pop())   # 3 — LIFO
```

Classic use: function call stacks, undo/redo, balanced-parentheses validation, DFS.

**Queue** — FIFO (first-in-first-out). Use `collections.deque`, **not** `list`, for O(1) `popleft` (a plain list's `pop(0)` is O(n) because it shifts every remaining element).

```python
from collections import deque
q = deque()
q.append(1); q.append(2); q.append(3)
print(q.popleft())   # 1 — FIFO, O(1)
```

**Deque (double-ended queue)** — O(1) push/pop from *both* ends; implemented as a doubly-linked list of fixed-size blocks in CPython.

```python
from collections import deque
d = deque([1, 2, 3])
d.appendleft(0)   # O(1)
d.append(4)       # O(1)
print(d)          # deque([0, 1, 2, 3, 4])
d = deque(maxlen=3)   # bounded deque — great for sliding-window problems
```

| Structure | Push | Pop | Access middle | Typical use |
|---|---|---|---|---|
| Stack | O(1) | O(1) | O(n) | DFS, parsing, backtracking |
| Queue | O(1) | O(1) | O(n) | BFS, task scheduling |
| Deque | O(1) both ends | O(1) both ends | O(n) | sliding window, palindrome checks |

**Pitfall:** using `list.pop(0)` or `list.insert(0, x)` for queue-like behavior — both are O(n), silently turning an intended O(n) algorithm into O(n²).

### 4.3 Hash Tables / Hash Maps

A hash table maps keys to values using a **hash function** to compute a bucket index, giving average O(1) insert/lookup/delete.

```python
h = {}
h["apple"] = 3     # hash("apple") -> bucket index -> store
print(h["apple"])  # O(1) average lookup
```

**Collision handling strategies:**
- **Open addressing (probing)** — on collision, probe subsequent slots (linear, quadratic, or double hashing) until an empty one is found. CPython's `dict`/`set` use open addressing with pseudo-random probing.
- **Separate chaining** — each bucket holds a linked list (or small dynamic array) of entries that hashed to it; on collision, append to the chain. Java's `HashMap` historically uses this (with tree-ification for long chains in modern versions).

**Load factor** = number of entries / number of buckets. High load factor increases collision probability and degrades performance, so hash tables **resize (rehash)** once load factor crosses a threshold (Python dicts resize around 2/3 full), amortizing the O(n) rehash cost across many O(1) insertions (same idea as list resizing — amortized O(1)).

```python
# Custom hash table sketch (separate chaining) — illustrates internals
class HashMap:
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.buckets = [[] for _ in range(capacity)]
        self.size = 0

    def _index(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        idx = self._index(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx][i] = (key, value)   # update
                return
        self.buckets[idx].append((key, value))
        self.size += 1
        if self.size / self.capacity > 0.7:
            self._resize()

    def get(self, key):
        idx = self._index(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        raise KeyError(key)

    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in old_buckets:
            for k, v in bucket:
                self.put(k, v)
```

**Use cases:** deduplication, counting/frequency maps, caching (memoization, LRU cache), fast lookups replacing O(n) linear scans, implementing sets/graphs (adjacency lists keyed by node).

**Pitfalls:** using mutable objects as dict keys (unhashable → `TypeError`); relying on dict ordering in old Python (guaranteed insertion order only since 3.7+, an implementation detail promoted to language spec); poor `__hash__` implementations causing clustering and degraded O(n) worst-case behavior.

### 4.4 Trees: Binary Trees, BST, Balanced Trees, Tries, Heaps/Priority Queues

**Binary tree** — each node has at most two children.

```python
class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def inorder(root):                 # O(n) time, O(h) space (recursion stack, h = height)
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)
```

**Binary Search Tree (BST)** — left subtree values < node < right subtree values, enabling O(log n) search/insert/delete *if balanced*, degrading to O(n) if the tree becomes a skewed chain (e.g., inserting sorted data in order).

```python
def bst_insert(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = bst_insert(root.left, val)
    else:
        root.right = bst_insert(root.right, val)
    return root

def bst_search(root, val):
    if root is None or root.val == val:
        return root
    return bst_search(root.left, val) if val < root.val else bst_search(root.right, val)
```

**Balanced trees (AVL / Red-Black — concept level):** self-balancing BSTs that guarantee O(log n) height via rotation rules after insert/delete, preventing the skewed-chain worst case.
- **AVL tree**: maintains the invariant that for every node, the height difference between left and right subtrees is at most 1; rebalances via rotations after every insert/delete. Stricter balance → faster lookups, slightly more expensive writes.
- **Red-Black tree**: maintains balance via node "color" invariants (red/black coloring rules) with looser balance than AVL, giving faster insert/delete at the cost of slightly slower lookups. Used internally in C++ `std::map`, Java `TreeMap`, and Linux kernel schedulers.
- You are rarely asked to implement these from scratch in an interview, but you must be able to explain *why* they guarantee O(log n) (bounded height) and name real-world usages.

**Tries (prefix trees)** — tree where each path from root represents a string prefix; each node has up to 26 (or alphabet-size) children. Used for autocomplete, spell-check, IP routing tables.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):                 # O(L), L = word length
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def search(self, word):                  # O(L)
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):           # O(L)
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True
```

**Heaps / Priority Queues** — a complete binary tree stored as an array, satisfying the heap property (min-heap: parent ≤ children; max-heap: parent ≥ children). Gives O(1) peek of the min/max, O(log n) insert/extract.

```python
import heapq

min_heap = []
heapq.heappush(min_heap, 5)
heapq.heappush(min_heap, 1)
heapq.heappush(min_heap, 3)
print(heapq.heappop(min_heap))   # 1 — smallest first

# Python's heapq is min-heap only; simulate max-heap by negating values
max_heap = []
for val in [5, 1, 3]:
    heapq.heappush(max_heap, -val)
print(-heapq.heappop(max_heap))  # 5

# heapq also provides O(n) heapify and efficient k-largest/k-smallest
nums = [5, 1, 9, 3, 7]
heapq.heapify(nums)               # O(n), in-place
print(heapq.nlargest(2, nums))    # [9, 7]
```

Array-based heap index math: for node at index `i`, parent = `(i-1)//2`, left child = `2i+1`, right child = `2i+2` — this is why heaps need no explicit pointers.

| Structure | Search | Insert | Delete | Min/Max | Space |
|---|---|---|---|---|---|
| BST (balanced) | O(log n) | O(log n) | O(log n) | O(log n) | O(n) |
| BST (unbalanced, worst) | O(n) | O(n) | O(n) | O(n) | O(n) |
| Trie | O(L) | O(L) | O(L) | N/A | O(total chars) |
| Binary Heap | O(n) (not designed for search) | O(log n) | O(log n) (root) | O(1) peek | O(n) |

### 4.5 Graphs: Representations, Directed/Undirected, Weighted

A graph `G = (V, E)` consists of vertices and edges.

**Representations:**

```python
# Adjacency list — space O(V + E), efficient for sparse graphs; most common in interviews
adj_list = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A"],
    "D": ["B"],
}

# Adjacency matrix — space O(V^2), O(1) edge-existence check, wasteful for sparse graphs
adj_matrix = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
]

# Weighted graph — adjacency list of (neighbor, weight) pairs
weighted_adj = {
    "A": [("B", 4), ("C", 1)],
    "B": [("D", 2)],
    "C": [("B", 1)],
    "D": [],
}
```

| Representation | Space | Edge lookup | Iterate neighbors | Best for |
|---|---|---|---|---|
| Adjacency list | O(V + E) | O(degree) | O(degree) | sparse graphs (most real-world graphs) |
| Adjacency matrix | O(V²) | O(1) | O(V) | dense graphs, or when O(1) edge checks matter |

**Directed vs undirected:** directed graphs (digraphs) have edges with direction (`A -> B` doesn't imply `B -> A`); undirected graphs treat edges symmetrically. Representing an undirected graph via adjacency list requires adding the edge to both nodes' lists.

```python
def add_undirected_edge(adj, u, v):
    adj.setdefault(u, []).append(v)
    adj.setdefault(v, []).append(u)
```

Graphs may also have **weighted** edges (cost/distance/capacity) vs **unweighted** (all edges equal cost), which determines whether BFS (unweighted shortest path) or Dijkstra/Bellman-Ford (weighted shortest path) is the correct algorithm — see §5.6.

### 4.6 Design Problems: LRU Cache and Min-Stack (Full Implementations)

"Design a data structure that supports X, Y, Z in O(1)" is one of the most common ML/AI Engineer interview formats because it tests whether you can combine two simple structures (usually a hash map plus something else) to hit a complexity target neither structure alone can achieve. The two most frequently asked are below; §4.4's Trie (with the autocomplete walk in this section's Interview Questions) is the third classic member of this family.

**LRU (Least Recently Used) Cache** — supports `get(key)` and `put(key, value)` in O(1), evicting the least-recently-used entry once capacity is exceeded. The standard-library shortcut uses `collections.OrderedDict`, which is internally a hash map plus a doubly linked list — exactly the combination the "from scratch" version builds explicitly.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()      # preserves insertion/access order

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)     # mark as most-recently-used -- O(1)
        return self.cache[key]

    def put(self, key, value) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)   # evict least-recently-used (front) -- O(1)
```

Complexity: O(1) for both `get` and `put` — `OrderedDict.move_to_end` and `popitem` are both O(1) because the structure maintains an internal doubly linked list alongside the hash map.

**From-scratch version (hash map + doubly linked list)** — shows what `OrderedDict` is doing internally, useful when an interviewer disallows the standard-library shortcut:

```python
class _Node:
    __slots__ = ("key", "val", "prev", "next")
    def __init__(self, key=None, val=None):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCacheManual:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.map = {}                 # key -> _Node, O(1) lookup
        self.head = _Node()           # dummy sentinel, head.next = most-recently-used
        self.tail = _Node()           # dummy sentinel, tail.prev = least-recently-used
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node) -> None:
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)
        self._add_to_front(node)
        return node.val

    def put(self, key, value) -> None:
        if key in self.map:
            self._remove(self.map[key])
        node = _Node(key, value)
        self.map[key] = node
        self._add_to_front(node)
        if len(self.map) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]
```

Complexity: O(1) for `get`/`put` — the hash map gives O(1) node lookup by key, and the doubly linked list gives O(1) splice (remove + re-insert at the front) with no shifting, which is exactly why a singly linked list or a plain array would not work here (removing an arbitrary middle node needs the `prev` pointer, and an array's middle removal is O(n)).

**Min-Stack** — a stack supporting `push`, `pop`, `top`, and `get_min`, all in O(1). The trick is caching the running minimum alongside every element instead of recomputing it.

```python
class MinStack:
    def __init__(self):
        self.stack = []              # each entry: (value, min_so_far_including_this_value)

    def push(self, val: int) -> None:
        current_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, current_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def get_min(self) -> int:
        return self.stack[-1][1]

ms = MinStack()
ms.push(3); ms.push(1); ms.push(2)
print(ms.get_min())   # 1
ms.pop()
print(ms.get_min())   # 1 -- still 1, popping the 2 didn't change the running minimum
```

Complexity: O(1) time for all four operations; O(n) space — one extra cached minimum per stack entry, versus the naive approach of scanning the whole stack for `get_min()` (O(n) per call). A more memory-frugal variant stores only the minimum *when it changes* on a second stack, trading a small amount of extra logic for O(1) amortized extra space instead of O(n) always, but the paired-tuple version above is the simplest to derive and explain under interview time pressure.

**Pitfalls:** for LRU Cache, forgetting to move an entry to most-recently-used on `get` (not just `put`) — a `get` is itself a "use" and must refresh recency, a detail interviewers specifically probe; for Min-Stack, recomputing the minimum by scanning on every `get_min()` call silently reintroduces O(n) per call and defeats the point of the exercise.

### Interview Questions

1. **Q: Why is `list.pop(0)` O(n) but `deque.popleft()` O(1)?**
   A: A Python `list` is backed by a contiguous array; removing the first element requires shifting every remaining element left by one slot — O(n). `collections.deque` is implemented as a doubly-linked list of fixed-size blocks, so both ends support O(1) insertion/removal without shifting.

2. **Q: Implement a stack-based solution to check if a string of brackets `()[]{}` is balanced.**
   ```python
   def is_valid(s: str) -> bool:
       pairs = {')': '(', ']': '[', '}': '{'}
       stack = []
       for ch in s:
           if ch in "([{":
               stack.append(ch)
           elif ch in pairs:
               if not stack or stack.pop() != pairs[ch]:
                   return False
       return not stack
   print(is_valid("{[()]}"))   # True
   print(is_valid("{[(])}"))   # False
   ```
   Complexity: O(n) time, O(n) space (worst case, all opening brackets).

3. **Q: What is the load factor of a hash table, and what happens when it gets too high?**
   A: Load factor = number of stored entries / number of buckets. As it rises, collisions become more frequent, degrading average-case O(1) operations toward O(n) (long chains or probe sequences). Hash tables mitigate this by **resizing** (typically doubling capacity) once load factor crosses a threshold, then rehashing all existing entries — an O(n) one-time cost that's amortized to O(1) per insertion over the sequence of insertions.

4. **Q: Design a data structure supporting `insert`, `remove`, and `getRandom` all in average O(1). (This is a classic "design" interview question.)**
   ```python
   import random
   class RandomizedSet:
       def __init__(self):
           self.arr = []
           self.idx = {}          # value -> index in arr

       def insert(self, val) -> bool:
           if val in self.idx:
               return False
           self.idx[val] = len(self.arr)
           self.arr.append(val)
           return True

       def remove(self, val) -> bool:
           if val not in self.idx:
               return False
           i = self.idx[val]
           last = self.arr[-1]
           self.arr[i] = last               # swap with last element
           self.idx[last] = i
           self.arr.pop()
           del self.idx[val]
           return True

       def get_random(self):
           return random.choice(self.arr)
   ```
   A: The trick is combining a hash map (value -> array index) with a dynamic array; removal swaps the target with the last element before popping, avoiding an O(n) shift. All three operations are O(1) average.

5. **Q: Given a Trie populated with a dictionary, how would you implement autocomplete (return all words with a given prefix)?**
   ```python
   def autocomplete(trie, prefix):
       node = trie.root
       for ch in prefix:
           if ch not in node.children:
               return []
           node = node.children[ch]
       results = []
       def dfs(n, path):
           if n.is_end:
               results.append(prefix + path)
           for ch, child in n.children.items():
               dfs(child, path + ch)
       dfs(node, "")
       return results
   ```
   Complexity: O(L) to walk the prefix, plus O(k) to collect all matching words where k is the total length of characters in matches — much faster than scanning the whole dictionary (O(n·L)).

6. **Q: Explain why an unbalanced BST can degrade to O(n) operations, and name two structures that prevent this.**
   A: If elements are inserted in sorted order, a naive BST degenerates into a linked list (each node has only one child), making search/insert/delete O(n) instead of O(log n). AVL trees and Red-Black trees prevent this by enforcing balance invariants and performing rotations after insert/delete to keep height O(log n).

7. **Q: Implement a min-heap-based solution to find the k-th largest element in an unsorted array.**
   ```python
   import heapq
   def kth_largest(nums, k):
       heap = nums[:k]
       heapq.heapify(heap)                 # O(k)
       for num in nums[k:]:
           if num > heap[0]:
               heapq.heapreplace(heap, num) # O(log k)
       return heap[0]
   ```
   Complexity: O(n log k) time (better than sorting the whole array, O(n log n), when k << n), O(k) space. Alternative one-liner: `heapq.nlargest(k, nums)[-1]`, same asymptotic complexity.

8. **Q: When would you choose an adjacency matrix over an adjacency list?**
   A: When the graph is dense (E close to V²) and you need O(1) edge-existence checks frequently (e.g., "is there an edge between u and v?"), or when V is small enough that O(V²) space is acceptable and matrix operations (e.g., via linear algebra / matrix powers for path counting) are useful. For sparse real-world graphs (social networks, road networks), adjacency lists are almost always preferred for their O(V+E) space.

9. **Q: What's the difference between a stack-based DFS traversal and a queue-based BFS traversal of a tree/graph, in terms of the order nodes are visited?**
   A: DFS (via stack or recursion) explores as deep as possible along one branch before backtracking, visiting nodes in depth order — good for path-finding, topological sort, cycle detection. BFS (via queue) explores all neighbors at the current depth before moving deeper, visiting nodes in level order — good for shortest path in unweighted graphs and finding the minimum number of steps/hops.

10. **Q: Two dict keys `d[(1,2)]` and a mutable key attempt `d[[1,2]]` — explain what happens and why.**
    A: `(1, 2)` is a tuple — immutable and hashable (assuming its elements are hashable), so it works fine as a dict key. `[1, 2]` is a list — mutable, and Python deliberately makes mutable built-in containers unhashable (`list.__hash__` is `None`) to prevent a key's hash from changing after insertion, which would corrupt the hash table's internal bucket placement. Attempting `d[[1,2]] = ...` raises `TypeError: unhashable type: 'list'`.

11. **Q: Explain the array-based representation of a binary heap and derive the parent/child index formulas.**
    A: A complete binary tree can be stored without pointers in a flat array by placing nodes in level order. For a 0-indexed array, the node at index `i` has parent at `(i-1)//2`, left child at `2i+1`, right child at `2i+2`. This works because a complete tree has no "gaps," so level-order indexing perfectly captures the tree structure implicitly.

12. **Q: Write a function to detect a cycle in a singly linked list, and explain why it's O(n) time and O(1) space.**
    ```python
    def has_cycle(head) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False
    ```
    A: This is Floyd's cycle detection ("tortoise and hare"). The fast pointer moves 2 steps for every 1 step of the slow pointer; if there's a cycle, the fast pointer eventually laps the slow pointer inside the cycle, guaranteeing a meeting within O(n) steps. Space is O(1) since only two pointers are used, versus an O(n)-space hash-set-of-visited-nodes alternative.

13. **Q: You need a cache that evicts the least-recently-used item once it exceeds capacity, with O(1) get and put. Sketch the data structure design.**
    A: Combine a hash map (key -> node) with a doubly linked list ordered by recency (most-recently-used at head, least at tail). `get` moves the accessed node to the head (O(1) via pointer rewiring using the hash map to locate the node), `put` inserts at head and evicts the tail node when over capacity — all O(1) because both the hash map lookup and the doubly-linked-list splice are O(1). (Python's `collections.OrderedDict` — or plain `dict` in 3.7+ — with `move_to_end` and `popitem(last=False)` implements this directly.)

14. **Q: Implement an LRU cache from scratch (no `OrderedDict`) with O(1) `get` and `put`, and explain why a singly linked list wouldn't work as well as a doubly linked list here.**
    A: See §4.6 `LRUCacheManual`. A singly linked list only stores a `next` pointer, so removing an arbitrary node (the one being accessed via `get`, which could be anywhere in the recency order) requires walking from the head to find its predecessor — O(n). A doubly linked list's `prev` pointer lets you splice any node out in O(1) directly, which combined with an O(1) hash-map lookup by key is what makes both `get` and `put` O(1) overall.

15. **Q: Design a stack that supports push, pop, top, and retrieving the minimum element, all in O(1). Why is the naive approach of scanning the stack for `get_min()` insufficient?**
    A: See §4.6 `MinStack`. Naively scanning the stack for the minimum on every `get_min()` call is O(n) per call, which defeats an O(1) requirement outright — even though push/pop/top would still be O(1), a single O(n) operation breaks the overall guarantee. The fix caches the running minimum alongside each pushed value (or on a parallel min-stack that only records new minimums), so `get_min()` becomes a plain O(1) lookup of the top's cached value.

16. **Q: Your min-stack from the previous question is pushed 10 million integers, mostly decreasing. What's the space complexity of the paired-tuple version, and can you do better?**
    A: The paired-tuple version (`(value, min_so_far)` per entry) is O(n) space regardless of the data pattern — one extra integer per element, always. You can reduce the *extra* space to O(m) where m is the number of times the minimum actually changes (which could be as low as O(1) for constant data, or as high as O(n) for strictly decreasing data, as in this question's mostly-decreasing case) by keeping a second, smaller stack that only records a new minimum when one occurs, popping it off the min-stack only when the corresponding value is popped off the main stack. For this specific "mostly decreasing" input, the improved version offers little savings since nearly every push sets a new minimum — worst case still O(n).

---

## 5. Core Algorithms

### 5.1 Sorting

```python
# Bubble sort — O(n^2) time, O(1) space, stable. Repeatedly swaps adjacent out-of-order pairs.
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:      # early exit if already sorted -> best case O(n)
            break
    return arr

# Insertion sort — O(n^2) worst, O(n) best (nearly sorted input), O(1) space, stable.
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# Selection sort — O(n^2) always (even best case), O(1) space, NOT stable.
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Merge sort — O(n log n) always, O(n) extra space, stable. Divide and conquer.
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

# Quick sort — O(n log n) average, O(n^2) worst (bad pivots), O(log n) space (in-place, stack), NOT stable.
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    return quick_sort(less) + equal + quick_sort(greater)
# Note: this functional version builds new lists at every level, so it actually
# uses O(n) extra space, not the O(log n) of a true in-place quicksort. The O(log n)
# space claim in the table below refers to the classic in-place partition version
# (swap elements around a pivot index, recurse on sub-ranges instead of allocating
# new lists) — see the in-place Lomuto-partition implementation in the Interview
# Questions below for the space-efficient variant.

# Heap sort — O(n log n) always, O(1) extra space (in-place), NOT stable.
import heapq
def heap_sort(arr):
    heapq.heapify(arr)                       # O(n)
    return [heapq.heappop(arr) for _ in range(len(arr))]   # O(n log n)
```

**Comparison table:**

| Algorithm | Best | Average | Worst | Space | Stable | In-place |
|---|---|---|---|---|---|---|
| Bubble sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
| Insertion sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
| Selection sort | O(n²) | O(n²) | O(n²) | O(1) | No | Yes |
| Merge sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No |
| Quick sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No | Yes |
| Heap sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No | Yes |

**Stability** matters when sorting records by one key but needing ties to preserve their original relative order (e.g., sort employees by department, keeping original name order within each department). Python's built-in `sorted()`/`list.sort()` use **Timsort** — a hybrid stable merge/insertion sort, O(n log n) worst case, O(n) best case for nearly-sorted data.

**When to use which:** merge sort for guaranteed O(n log n) and stability (external sorting, linked lists); quicksort for in-place average-case speed on random data; heap sort when O(1) extra space is critical and stability doesn't matter; insertion sort for small or nearly-sorted arrays (also used as the base case inside Timsort/introsort for small partitions).

### 5.2 Searching: Binary Search and Variants

Binary search requires a **sorted** (or monotonic) sequence and repeatedly halves the search space — O(log n) time, O(1) space (iterative).

```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# Variant: leftmost insertion point (lower_bound) — first index where arr[i] >= target
def lower_bound(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

# Python's bisect module provides this directly
import bisect
bisect.bisect_left(arr, target)    # same as lower_bound
bisect.bisect_right(arr, target)   # first index where arr[i] > target
```

**Binary search on the answer** — a powerful pattern for optimization problems where the *answer space* (not the input array) is monotonic with respect to a feasibility check.

```python
# Example: minimum capacity to ship packages within `days` days
def ship_within_days(weights, days):
    def can_ship(capacity):
        needed_days, current = 1, 0
        for w in weights:
            if current + w > capacity:
                needed_days += 1
                current = 0
            current += w
        return needed_days <= days

    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = (lo + hi) // 2
        if can_ship(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

Complexity: O(n log(sum(weights))) — binary search over the capacity range, each feasibility check is O(n).

**Pitfall:** off-by-one errors in loop bounds (`<` vs `<=`, `mid` vs `mid+1`/`mid-1`); infinite loops from forgetting to shrink the search space on every iteration.

### 5.3 Recursion and Backtracking

**Recursion** solves a problem by reducing it to smaller instances of itself, requiring a **base case** (stopping condition) and a **recursive case** that makes progress toward it.

```python
def factorial(n):
    if n <= 1:              # base case
        return 1
    return n * factorial(n - 1)   # recursive case
```

**Backtracking** — a refinement of recursion for exploring all candidate solutions incrementally, abandoning ("pruning") a partial solution as soon as it's determined invalid, then trying the next candidate.

```python
def permutations(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()                     # undo choice ("backtrack")
    backtrack([], nums)
    return result

print(permutations([1, 2, 3]))
# [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

Complexity: O(n! · n) time (n! permutations, each O(n) to build/copy), O(n) auxiliary space for the recursion depth (excluding the output).

**Classic backtracking template:**

```python
def backtrack(state):
    if is_solution(state):
        record(state)
        return
    for choice in get_candidates(state):
        if is_valid(choice, state):
            apply(choice, state)
            backtrack(state)
            undo(choice, state)   # critical: revert before trying next candidate
```

Applications: N-Queens, Sudoku solver, subset/combination generation, generating valid parentheses, word search in a grid.

**Pitfalls:** forgetting to undo state mutations (leads to corrupted subsequent branches); not pruning early enough (leads to unnecessary exponential blowup — always check validity as early as possible rather than only at full-solution time); Python's default recursion limit (~1000, `sys.setrecursionlimit`) can be hit on deep recursion — consider converting to iterative with an explicit stack for very deep problems.

### 5.4 Dynamic Programming: Memoization vs Tabulation

DP solves problems with **optimal substructure** (optimal solution built from optimal solutions to subproblems) and **overlapping subproblems** (naive recursion recomputes the same subproblem many times).

**Memoization (top-down)** — recursive solution + a cache to avoid recomputation.

```python
import functools

@functools.lru_cache(maxsize=None)
def fib_memo(n):
    if n < 2:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)
# O(n) time (each of n subproblems computed once), O(n) space (cache + recursion stack)
```

**Tabulation (bottom-up)** — iterative solution building a table from the base cases upward, usually avoiding recursion-stack overhead entirely.

```python
def fib_tab(n):
    if n < 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
# O(n) time, O(n) space — can be reduced to O(1) space by keeping only last two values
def fib_optimal(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

**Classic problem: 0/1 Knapsack** — maximize value within a weight capacity, each item used at most once.

```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]                       # don't take item i
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    return dp[n][capacity]
```

Complexity: O(n · capacity) time and space; space reducible to O(capacity) with a 1-D rolling array iterated in reverse over `w`.

**Classic problem: Longest Common Subsequence (LCS)**

```python
def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

Complexity: O(m·n) time and space.

**Classic problem: Edit Distance (Levenshtein)**

```python
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # delete
                    dp[i][j-1],     # insert
                    dp[i-1][j-1],   # replace
                )
    return dp[m][n]
```

Complexity: O(m·n) time and space.

**Classic problem: Coin Change (fewest coins to make amount)**

```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a-c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

Complexity: O(amount · len(coins)) time, O(amount) space.

**Memoization vs tabulation tradeoffs:**

| | Memoization (top-down) | Tabulation (bottom-up) |
|---|---|---|
| Style | Recursive + cache | Iterative table fill |
| Computes only needed subproblems | Yes (lazy) | No (fills entire table, even unused states) |
| Recursion stack overhead | Yes | No |
| Easier to derive from brute force | Yes — minimal changes to recursive solution | Requires figuring out iteration order |
| Risk | Stack overflow on deep recursion | None (usually) |

### 5.5 Greedy Algorithms

Greedy algorithms make the locally optimal choice at each step, hoping it leads to a globally optimal solution. Correct only when the problem exhibits the **greedy-choice property** (local optimum leads to global optimum) — must be proven, not assumed.

```python
# Activity selection: maximum number of non-overlapping intervals
def max_non_overlapping(intervals):
    intervals.sort(key=lambda x: x[1])   # greedy: sort by end time
    count, last_end = 0, float('-inf')
    for start, end in intervals:
        if start >= last_end:
            count += 1
            last_end = end
    return count
```

Complexity: O(n log n) for the sort, O(n) for the scan.

```python
# Coin change with canonical coin systems (greedy works ONLY for systems like US coins)
def greedy_coin_change(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for c in coins:
        count += amount // c
        amount %= c
    return count if amount == 0 else -1
```

**Pitfall:** greedy coin change fails for non-canonical coin systems, e.g. coins `[1, 3, 4]` and amount `6` — greedy picks `4 + 1 + 1 = 3 coins`, but the optimal is `3 + 3 = 2 coins`. This is exactly why the general coin-change problem requires DP, not greedy — always verify the greedy-choice property holds before applying it, and be ready to explain a counterexample in interviews.

Other classic greedy problems: Huffman coding, Dijkstra's algorithm (greedy + priority queue), Kruskal's/Prim's minimum spanning tree, interval scheduling/partitioning, job sequencing with deadlines.

### 5.6 Graph Algorithms: BFS, DFS, Topological Sort, Dijkstra, Bellman-Ford, Union-Find

**BFS (Breadth-First Search)** — level-by-level traversal using a queue; finds shortest paths in **unweighted** graphs.

```python
from collections import deque

def bfs(adj, start):
    visited = {start}
    order = []
    q = deque([start])
    while q:
        node = q.popleft()
        order.append(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return order
```

Complexity: O(V + E) time, O(V) space.

**DFS (Depth-First Search)** — explores as deep as possible before backtracking; via recursion or an explicit stack.

```python
def dfs(adj, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for neighbor in adj[start]:
        if neighbor not in visited:
            order.extend(dfs(adj, neighbor, visited))
    return order
```

Complexity: O(V + E) time, O(V) space (recursion stack + visited set).

**Topological sort** — a linear ordering of nodes in a **Directed Acyclic Graph (DAG)** such that for every edge `u -> v`, `u` comes before `v`. Used for build/dependency ordering, course scheduling.

```python
# Kahn's algorithm (BFS-based, using in-degrees)
from collections import deque

def topo_sort(adj, num_nodes):
    in_degree = [0] * num_nodes
    for u in adj:
        for v in adj[u]:
            in_degree[v] += 1
    q = deque([n for n in range(num_nodes) if in_degree[n] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    if len(order) != num_nodes:
        raise ValueError("graph has a cycle — no valid topological order")
    return order
```

Complexity: O(V + E) time and space.

**Dijkstra's algorithm** — single-source shortest paths on a graph with **non-negative** edge weights, using a min-heap priority queue.

```python
import heapq

def dijkstra(adj, start):
    dist = {node: float('inf') for node in adj}
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue    # stale entry, already found a shorter path
        for v, weight in adj[u]:
            nd = d + weight
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
```

Complexity: O((V + E) log V) with a binary heap. **Fails with negative edge weights** (a negative edge can invalidate the greedy "once popped, finalized" assumption).

**Bellman-Ford** — single-source shortest paths that correctly handles **negative** edge weights and can detect negative-weight cycles, at the cost of higher complexity.

```python
def bellman_ford(edges, num_nodes, start):
    dist = [float('inf')] * num_nodes
    dist[start] = 0
    for _ in range(num_nodes - 1):            # relax all edges V-1 times
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for u, v, w in edges:                      # one more pass to detect negative cycles
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            raise ValueError("graph contains a negative-weight cycle")
    return dist
```

Complexity: O(V·E) time, O(V) space — slower than Dijkstra but strictly more general.

**Union-Find (Disjoint Set Union)** — tracks a partition of elements into disjoint sets, supporting near-O(1) `union` and `find` via **union by rank/size** and **path compression**. Used for cycle detection in undirected graphs, Kruskal's MST, connected components.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # path compression
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False               # already connected -> would form a cycle
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra           # union by rank
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True
```

Complexity: O(α(n)) amortized per operation (α = inverse Ackermann function, effectively constant for all practical n) with both path compression and union by rank applied together.

| Algorithm | Handles negative weights? | Time complexity | Use case |
|---|---|---|---|
| BFS | N/A (unweighted) | O(V + E) | shortest path, unweighted |
| DFS | N/A | O(V + E) | connectivity, cycle detection, topo sort |
| Dijkstra | No | O((V+E) log V) | shortest path, non-negative weights |
| Bellman-Ford | Yes | O(V·E) | shortest path with negative weights, cycle detection |
| Union-Find | N/A | O(α(n)) amortized | connectivity, MST (Kruskal), cycle detection |

### 5.7 Common Interview Patterns

**Two pointers** — use two indices moving through a structure (often from both ends, or at different speeds) to avoid nested loops.

```python
def two_sum_sorted(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        s = arr[lo] + arr[hi]
        if s == target:
            return [lo, hi]
        elif s < target:
            lo += 1
        else:
            hi -= 1
    return []
```
Complexity: O(n) time, O(1) space — versus O(n²) brute force.

**Sliding window** — maintain a window `[left, right]` over a sequence, expanding/shrinking it to satisfy a constraint, avoiding recomputation from scratch.

```python
def longest_substring_no_repeat(s):
    seen = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        best = max(best, right - left + 1)
    return best
```
Complexity: O(n) time (each character visited a constant number of times), O(min(n, alphabet)) space.

**Fast/slow pointers** — two pointers moving at different speeds through a linked list or array, used for cycle detection, finding the middle, or finding the n-th-from-end node.

```python
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow   # slow is at the middle when fast reaches the end
```
Complexity: O(n) time, O(1) space.

**Binary search on the answer** — covered in §5.2; use when the answer space is monotonic w.r.t. a feasibility predicate.

**Monotonic stack** — a stack kept strictly increasing or decreasing, used to answer "next greater/smaller element" style queries in O(n) total instead of O(n²).

```python
def next_greater_element(nums):
    result = [-1] * len(nums)
    stack = []                      # holds indices, values decreasing bottom-to-top... 
    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            result[stack.pop()] = num
        stack.append(i)
    return result

print(next_greater_element([2, 1, 2, 4, 3]))   # [4, 2, 4, -1, -1]
```
Complexity: O(n) time — each element is pushed and popped at most once, despite the nested `while` loop, due to amortized analysis. O(n) space for the stack.

### Interview Questions

1. **Q: Compare quicksort and merge sort. When would you choose one over the other?**
   A: Merge sort guarantees O(n log n) in all cases and is stable, but uses O(n) extra space and has more overhead for small arrays — preferred for linked lists, external sorting, or when stability/worst-case guarantees matter. Quicksort is in-place (O(log n) space) and typically faster in practice due to better cache locality and lower constant factors, but has O(n²) worst case on adversarial/already-sorted input with poor pivot choice — mitigated with randomized pivots or median-of-three, and preferred for general-purpose in-memory sorting.

2. **Q: Implement binary search iteratively and explain why the loop condition is `lo <= hi`, not `lo < hi`.**
   A: See §5.2. `lo <= hi` is required because when `lo == hi`, there's still exactly one candidate element left to check; using `lo < hi` would skip checking that last element, causing an off-by-one bug that misses valid targets at the boundary.

3. **Q: Solve: given an array, find two numbers that sum to a target. Provide both the brute-force and optimal solutions with complexities.**
   ```python
   # Brute force: O(n^2) time, O(1) space
   def two_sum_brute(nums, target):
       for i in range(len(nums)):
           for j in range(i+1, len(nums)):
               if nums[i] + nums[j] == target:
                   return [i, j]

   # Optimal: O(n) time, O(n) space using a hash map
   def two_sum_optimal(nums, target):
       seen = {}
       for i, n in enumerate(nums):
           if target - n in seen:
               return [seen[target - n], i]
           seen[n] = i
   ```

4. **Q: Explain the difference between memoization and tabulation, and rewrite the naive exponential Fibonacci function using each.**
   A: See §5.4 for full explanation and code. Memoization is top-down recursion plus a cache (lazy — only computes needed subproblems); tabulation is bottom-up iteration filling a table (usually more space/stack efficient, no recursion overhead).

5. **Q: Why does greedy coin-change fail for coin denominations `[1, 3, 4]` with target `6`, and how does DP fix it?**
   A: Greedy (always take the largest coin ≤ remaining amount) picks 4, then 1, then 1 → 3 coins, but the optimal answer is 3+3 → 2 coins. Greedy fails because this coin system lacks the "canonical" property required for the greedy-choice property to hold. DP fixes it by exploring *all* combinations via `dp[a] = min(dp[a], dp[a-c] + 1)` for every coin `c`, guaranteeing the true optimum rather than a locally-greedy one. (Full code in §5.5.)

6. **Q: Given a list of meeting time intervals, determine the minimum number of conference rooms required. Outline your approach and complexity.**
   A: Separate start and end times into two sorted arrays; sweep through using two pointers — increment a room counter when a meeting starts, decrement when one ends (processing ends before starts at equal times); track the maximum concurrent count.
   ```python
   def min_meeting_rooms(intervals):
       starts = sorted(i[0] for i in intervals)
       ends = sorted(i[1] for i in intervals)
       rooms = max_rooms = 0
       s = e = 0
       while s < len(starts):
           if starts[s] < ends[e]:
               rooms += 1
               s += 1
               max_rooms = max(max_rooms, rooms)
           else:
               rooms -= 1
               e += 1
       return max_rooms
   ```
   Complexity: O(n log n) for sorting, O(n) for the sweep.

7. **Q: Detect a cycle in a directed graph. Give the algorithm and complexity.**
   A: Use DFS with three states per node — unvisited, in-progress (on current recursion stack), done. A cycle exists if DFS encounters a node that is currently "in-progress" (a back edge).
   ```python
   def has_cycle(adj, num_nodes):
       WHITE, GRAY, BLACK = 0, 1, 2
       color = [WHITE] * num_nodes
       def dfs(u):
           color[u] = GRAY
           for v in adj[u]:
               if color[v] == GRAY:
                   return True
               if color[v] == WHITE and dfs(v):
                   return True
           color[u] = BLACK
           return False
       return any(color[n] == WHITE and dfs(n) for n in range(num_nodes))
   ```
   Complexity: O(V + E) time, O(V) space.

8. **Q: Explain why Dijkstra's algorithm fails with negative edge weights, with a concrete counterexample.**
   A: Dijkstra greedily finalizes a node's shortest distance once popped from the priority queue, assuming no future edge could produce a shorter path to it. A negative edge violates this: e.g., nodes A→B (weight 5), A→C (weight 2), C→B (weight -10). Dijkstra finalizes B at distance 5 before exploring C, missing the true shortest path A→C→B = 2 + (-10) = -8. Bellman-Ford handles this correctly by relaxing all edges V-1 times rather than assuming greedy finality.

9. **Q: What is the time complexity of Union-Find with both path compression and union by rank, and why is it effectively constant in practice?**
   A: O(α(n)) amortized per operation, where α is the inverse Ackermann function — grows so slowly that for any n representable in the physical universe, α(n) ≤ 4, making it "effectively constant" for all practical purposes, even though it is not formally O(1) for all n.

10. **Q: Solve "longest substring without repeating characters" and explain the sliding window technique used.**
    A: See §5.7 for the full solution. The window `[left, right]` expands by moving `right` forward; whenever a repeated character is found within the current window, `left` jumps past the previous occurrence of that character. Each index is visited a constant number of times, giving O(n) time instead of the O(n²) brute-force approach of checking every substring.

11. **Q: You need to find the k-th smallest element in an unsorted array. Compare a full-sort approach, a heap-based approach, and Quickselect in terms of complexity.**
    A: Full sort: O(n log n) time, simplest. Heap-based (max-heap of size k): O(n log k) time, O(k) space — better when k << n. Quickselect (partition-based, like quicksort but recursing into only one side): O(n) average case, O(n²) worst case, O(1) extra space (in-place) — the best average-case choice when only a single order statistic is needed, not a full sort.

12. **Q: Write a recursive and an iterative version of computing the n-th Fibonacci number, and state which you'd use in production and why.**
    A: See §5.4 for memoized/tabulated versions. In production, prefer the iterative O(1)-space version (`fib_optimal`) — no recursion-depth risk, no cache memory overhead, fastest and simplest for a problem with no branching subproblem structure beyond the immediate previous two values.

13. **Q: Explain the classic N-Queens backtracking approach and its worst-case complexity.**
    A: Place queens row by row; for each row, try every column, check if it conflicts with previously placed queens (same column, or same diagonal via `abs(row1-row2) == abs(col1-col2)`); recurse if valid, backtrack (remove and try next column) otherwise. Worst-case time complexity is O(n!) in the naive bound (n choices for row 1, at most n-1 for row 2, etc.), though pruning invalid branches early in practice makes it much faster than the naive bound for typical board sizes.

14. **Q: Given an array of stock prices, find the maximum profit from a single buy/sell (buy before sell). Give an O(n) solution.**
    ```python
    def max_profit(prices):
        min_price = float('inf')
        best = 0
        for p in prices:
            min_price = min(min_price, p)
            best = max(best, p - min_price)
        return best
    ```
    Complexity: O(n) time, O(1) space — track the minimum price seen so far and the best profit achievable by selling at the current price, in a single pass, versus O(n²) brute force checking every buy/sell pair.

15. **Q: What's the total time complexity of a monotonic-stack "next greater element" solution, and why isn't it O(n²) despite the nested loop?**
    A: O(n) total, via amortized analysis: although there's a `while` loop nested inside the `for` loop, every index is pushed onto the stack exactly once and popped at most once across the *entire* run of the algorithm — so the total number of push/pop operations across all iterations is bounded by 2n, not n² — the same "aggregate method" amortized-analysis argument used for dynamic array resizing.

---

## Rapid-Fire Interview Q&A

1. **Q: Is Python interpreted or compiled?** A: Both — CPython compiles source to bytecode (`.pyc`), then the bytecode is interpreted by the Python Virtual Machine at runtime.
2. **Q: What is duck typing?** A: An object's suitability is determined by the presence of methods/behavior it supports ("if it walks like a duck and quacks like a duck..."), not by its explicit type/class.
3. **Q: `str` is mutable or immutable?** A: Immutable — every "modification" creates a new string object.
4. **Q: What does `__init__.py` do?** A: Marks a directory as a Python package and can run package-level initialization code; not strictly required since Python 3.3 (namespace packages) but still standard practice.
5. **Q: Difference between `deepcopy` and `copy`?** A: `copy` duplicates only the top-level container (nested mutables are shared); `deepcopy` recursively duplicates all nested objects.
6. **Q: What does `*` do in `def f(a, *, b):`?** A: Forces `b` to be keyword-only; it cannot be passed positionally.
7. **Q: What is a generator's memory advantage?** A: It yields one value at a time (lazy evaluation), using O(1) memory regardless of sequence length, instead of materializing the whole sequence.
8. **Q: What's the output of `0.1 + 0.2 == 0.3`?** A: `False` — floating-point representation error; use `math.isclose()` for float comparisons.
9. **Q: What is PEP 8?** A: Python's official style guide (naming conventions, indentation, line length, etc.).
10. **Q: What's the time complexity of `dict` lookup?** A: O(1) average case, O(n) worst case (pathological collisions).
11. **Q: What does `__slots__` do?** A: Restricts instance attributes to a fixed set, saving memory by avoiding a per-instance `__dict__`.
12. **Q: What's a metaclass?** A: A "class of a class" — `type` is the default metaclass; custom metaclasses let you control class creation itself (rare, advanced usage).
13. **Q: Difference between `@property` and a plain attribute?** A: `@property` lets you run code (validation, computation) on attribute access while keeping attribute-like syntax.
14. **Q: What is the time complexity of building a heap from an unsorted array (`heapify`)?** A: O(n), not O(n log n) — a tighter amortized bound than naively inserting n elements one at a time.
15. **Q: What is tail recursion, and does Python optimize it?** A: Recursion where the recursive call is the last operation; Python does **not** perform tail-call optimization, so deep tail recursion can still hit the recursion limit/stack overflow.
16. **Q: What's the difference between `==` and `.equals()` in Python?** A: Python has no `.equals()` method — `==` universally invokes `__eq__`.
17. **Q: What does `sys.getsizeof()` measure?** A: The memory size in bytes of a single object (not including referenced sub-objects for containers).
18. **Q: What's a race condition?** A: A bug where the outcome depends on the non-deterministic timing/interleaving of concurrent operations on shared state.
19. **Q: What is a semaphore, briefly?** A: A concurrency primitive that limits the number of threads/processes that can access a resource simultaneously, via a counter.
20. **Q: What's the space complexity of an adjacency matrix for a graph with V vertices?** A: O(V²), regardless of how many edges actually exist.
21. **Q: What is the difference between `break` and `continue`?** A: `break` exits the loop entirely; `continue` skips to the next iteration.
22. **Q: What's a stable sort?** A: A sort that preserves the relative order of elements with equal keys.
23. **Q: What's the worst-case complexity of Quickselect?** A: O(n²), though O(n) on average.
24. **Q: What is a B-tree used for, briefly?** A: A self-balancing tree optimized for disk-based storage (databases, filesystems) with high branching factor to minimize disk seeks.
25. **Q: What is memoization, in one sentence?** A: Caching the results of expensive function calls to avoid recomputation on repeated identical inputs.
26. **Q: What's the difference between `is not` and `not ... is`?** A: They're equivalent; `is not` is just the more idiomatic/readable form.
27. **Q: What's a hash collision?** A: When two distinct keys map to the same hash bucket/index.
28. **Q: What is the GIL's effect on `asyncio`?** A: Irrelevant to `asyncio`'s concurrency model directly — asyncio achieves concurrency via a single-threaded event loop and cooperative multitasking, not by bypassing the GIL; it helps I/O-bound work regardless of the GIL because it doesn't need parallel bytecode execution.
29. **Q: What's the difference between an abstract class and an interface (conceptually, in Python)?** A: Python has no formal `interface` keyword; ABCs serve a similar role by declaring required methods that subclasses must implement, enforced at instantiation time.
30. **Q: What is the time complexity of `sorted()` on an already-sorted list using Timsort?** A: O(n) best case — Timsort detects existing order ("runs") and exploits it.
31. **Q: What's the pytest convention for test file/function discovery?** A: Files named `test_*.py` or `*_test.py`, containing functions named `test_*` (or methods in `Test*` classes) — pytest auto-discovers and runs them with no registration boilerplate.
32. **Q: What does `unittest.mock.patch` do, in one sentence?** A: Temporarily replaces a named object (function, method, class) in a target module's namespace with a `Mock` for the duration of a test, then restores the original automatically afterward.
33. **Q: cProfile vs line_profiler — what's the key difference?** A: `cProfile` reports timing per *function*; `line_profiler` reports timing per *line* inside a function you've decorated, needed when you know which function is slow but not which statement inside it.
34. **Q: What does `mypy` actually run — at what point in the development lifecycle?** A: A static analysis pass over source code (no program execution), typically run in CI/pre-commit, checking annotated types for consistency; it never runs as part of the program's own runtime behavior.
35. **Q: What's the difference between `pip freeze` output and a `poetry.lock`/`uv.lock` file?** A: `pip freeze` is an unstructured snapshot of currently installed package versions with no hashes or dependency-graph metadata; a lockfile is the deterministic, hash-pinned output of a real dependency resolver covering the full transitive graph, guaranteeing reproducible installs.
36. **Q: In `collections.OrderedDict`, what are the two operations that make it usable as the backbone of an O(1) LRU cache?** A: `move_to_end(key)` (relocate an existing entry to the front/back in O(1)) and `popitem(last=False)` (pop and return the oldest entry in O(1)) — together these give O(1) "mark as recently used" and O(1) "evict least-recently-used."

