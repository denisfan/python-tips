
## A guide to some of the Python fetaures that can be useful

## 1 Lambda, map, filter reduce

```python
def triple_fn(x):
    return x * x * x

triple_ld = lambda x: x * x * x

for n in range(10):
    assert triple_fn(n) == triple_ld(n)
```

The lambda keyword is used to created inline functions. The functions `triple_fn` and `triple_ld` above are identical. The `lambda` functions is ideal for use in callbacks, as well as when functions are to be passed as arguments to other functions, e.g. when use in conjunction with functions like `maps`, `filter` and `reduce`.

`map(fn, iterable)` applies the `fn` to all elements of the `iterable`, e.g. list, set, dictionary, tuple and string, and returns a map object.

```python
seqs = [1/2, 90/4, 5678/3493, 778/34, 5/7, 9999/288]
seqs_tripled = [seq * seq * seq for seq in seqs]
print(f'{seqs_tripled}')

=> [0.125, 11390.625, 4.295269396182883, 11981.247506615104, 0.3644314868804665, 41849.68966674805]
```

This is the same when using `map` with a callback function.

```python
seqs_tripled_with_fn = map(triple_fn, seqs)
seqs_tripled_with_ld = map(lambda x: x * x * x, seqs)

print(f'{list(seqs_tripled_with_fn)}')

=> [0.125, 11390.625, 4.295269396182883, 11981.247506615104, 0.3644314868804665, 41849.68966674805]
```

```python
print(f'{list(seqs_tripled_with_ld)}')

=> [0.125, 11390.625, 4.295269396182883, 11981.247506615104, 0.3644314868804665, 41849.68966674805]
```

You can also use `map` with more than one iterable, e.g. if you want to calculate the mean squared error of a simple linear function `f(x) = ax + b` with the true label labels, these two methods are equivalent:

```python
a, b = 3, -0.5
xs = [2, 3, 4, 5]
labels = [6.4, 8.9, 10.9, 15.3]

# Method 1: using a loop
errors = []
for i, x in enumerate(xs):
    errors.append((a * x + b - labels[i]) ** 2)
result1 = sum(errors) ** 0.5 / len(xs)

# Method 2: using map
diffs = map(lambda x, y: (a * x + b - y) ** 2, xs, labels)
result2 = sum(diffs) ** 0.5 / len(xs)

print(f'{result1, result2}')

=> (0.35089172119045514, 0.35089172119045514)
```

Note that objects returned by `map` and `filter` are iterators, which means that their values aren't stored but generated as needed. After you've called `sum(diffs)`, `diffs` becomes empty. If you want to keep all elements in `diffs`, convert it to a list using `list(diffs)`.

`filter(fn, iterable)` works the same way as `map`, except that `fn` returns a boolean value and `filter` returns all the elements of the `iterable` for which the `fn` returns True.

```python
not_nice_prediction = filter(lambda x: x > 0.5, errors)
print(f'{list(not_nice_prediction)}')

=> [0.8100000000000006, 0.6400000000000011]
```

`reduce(fn, iterable, initializer)` is used when we want to iteratively apply an operator to all elements in a list. For example, if we want to calculate the product of all elements in a list:

```python
product = 1
for seq in seqs:
    product *= seq

print(f'{product}')

=> 10377.34009153511
```

This is equivalent to:

```python
from functools import reduce

product = reduce(lambda x, y: x * y, seqs)
print(f'{product}')

=> 10377.34009153511
```

### Note on the performance of lambda functions
Lambda functions are meant for one time use. Each time `lambda x: dosomething(x)` is called, the function has to be created, which hurts the performance if you call `lambda x: dosomething(x)` multiple times, e.g. when you pass it inside `reduce`.

When you assign a name to the lambda function as in `fn = lambda x: dosomething(x)`, its performance is slightly slower than the same function defined using `def`, but the difference is negligible

## 2 List manipulation

### 2.1 Unpacking
Unpacking can be done like this:

```python
nums = [2, 4, 6, 8]
a, b, c, d = nums
print(nums)

=> [2, 4, 6, 8]
```

It can also be unpacked a list like this:

```python
a, *new_nums, d = nums
print(f'{a}')
print(f'{new_nums}')
print(f'{d}')

=> 2
=> [4, 6]
=> 8
```

### 2.2 Slicing
We can reverse a list like using `[::-1]`

```python
nums = list(range(10))
print(f'{nums}')

=> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(f'{nums[::-1]}')

=> [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

The syntax `[x:y:z]` means "take every `z`th element of a list from index `x` to index `y`". When `z` is negative, it indicates going backwards. When `x` isn't specified, it defaults to the first element of the list in the direction you are traversing the list. When `y` isn't specified, it defaults to the last element of the list. So if we want to take every 2th element of a list, we use `[::2]`

```python
evens = nums[::2]
print(f'{evens}')

=> [0, 2, 4, 6, 8]

reversed_evens = nums[-2::-2]
print(f'{reversed_evens}')

=> [8, 6, 4, 2, 0]
```

We also can use slicing to delete ll the even numbers in the list

```python
del nums[::2]
print(f'{nums}')

=> [1, 3, 5, 7, 9]
```

### 2.4 Flatttening
A list of lists can be flattened using `sum`

```python
list_of_lists = [[1], [2, 3], [4, 5, 6]]
sum(list_of_lists, [])

=> [1, 2, 3, 4, 5, 6]
```

For nested lists, we can recursively flatten it, through together with the lambda functions

```python
nested_lists = [[1, 2], [[3, 4], [5, 6], [[7, 8], [9, 10], [[11, [12, 13]]]]]]
flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
flatten(nested_lists)

=> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
```

### 2.5 List vs generator

```python
tokens = ['we', 'have', 'a', 'nice', 'weather', 'today']

def ngrams(token, n):
    length = len(tokens)
    grams = []
    for i in range(length - n + 1):
        grams.append(tokens[i:i+n])
    
    return grams

print(f'{ngrams(tokens, 3)}')

=> [['we', 'have', 'a'], 
    ['have', 'a', 'nice'], 
    ['a', 'nice', 'weather'], 
    ['nice', 'weather', 'today']]
```

In the above example, we have to store all the n-grams at the same time. If the text has m tokens, then the memory requirement is `O(nm)`, which can be problematic when m is large

Instead of using a list to store all n-grams, we can use a generator that generates the next n-gram when it's asked for. This is known as lazy evaluation. We can make the function `ngrams` returns a generator using the keyword `yield`. Then the memory requirement is `O(m+n)`

```python
def ngrams(tokens, n):
    length = len(tokens)
    for i in range(length - n + 1):
        yield tokens[i:i+n]

ngrams_generator = ngrams(tokens, 3)
print(f'{ngrams_generator}')

=> <generator object ngrams at 0x7ff417d9b200>

for ngram in ngrams_generator:
    print(f'{ngram}')

=> ['we', 'have', 'a']
   ['have', 'a', 'nice']
   ['a', 'nice', 'weather']
   ['nice', 'weather', 'today']
```

Another option to generate n-grams is to use slices to create lists: `[0, 1, ..., -n]`, `[1, 2, ..., -n+1]`, ..., `[n-1, n, ..., -1]`, and then zip them together.

```python
def ngrams(tokens, n):
    length = len(tokens)
    slices = (tokens[i:length-n+i+1] for i in range(n))
    return zip(*slices)

ngrams_generator = ngrams(tokens, 3)
print(f'{ngrams}')

=> <function ngrams at 0x7ff4181859d0>

for ngram in ngrams_generator:
    print(f'{ngram}')

=> ('we', 'have', 'a')
   ('have', 'a', 'nice')
   ('a', 'nice', 'weather')
   ('nice', 'weather', 'today')
```

To create slices, we use `(tokens[...] for i in range(n))` instead of `[tokens[...] for i in range(n)]`. `[]` is the normal list comprehension that returns a list. `()` returns a generator

## 3 local namespace, object's attributes
The `locals()` function returns a dictionary containing the variables defined in the local namespace

```python
class Model1:
    def __init__(self, hidden_size=10, num_of_layers=2, learning_rate=2):
        print(f'{locals()}')
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.learning_rate = learning_rate

model1 = Model1()

=> { 'self': <__main__.Model1 object at 0x7ff417dad4f0>,  
     'hidden_size': 10, 
     'num_of_layers': 2, 
     'learning_rate': 2}
```

All attributes of an object are stored in its `__dict__`

```python
print(f'{model1.__dict__}')

=> {'hidden_size': 10, 'num_of_layers': 2, 'learning_rate': 2}
```

To manually assign each of the arguments to an attribute can be quite tiring when the list of the arguments is large. To avoid this, we can directly assign the list of arguments to the object's `__dict__`.

```python
class Model2:
    def __init__(self, hidden_size=10, num_of_layers=2, learning_rate=2):
        params = locals()
        del params['self']
        self.__dict__ = params

model2 = Model2()
print(f'{model2.__dict__}')

=> {'hidden_size': 10, 'num_of_layers': 2, 'learning_rate': 2}
```

This can be especially convenient when the object is initiated using the catch-all `**kwargs`, though the use of `**kwargs` should be reduced to the minimum.

```python
class Model3:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

model3 = Model3(hidden_size=10, num_layers=2, learning_rate=2)
print(f'{model3.__dict__}')

=> {'hidden_size': 10, 'num_layers': 2, 'learning_rate': 2}
```

## 4 Wild import

Often, you run into this wild import `*` that looks something like this:

`file.py`

```python
from parts import *
```

This will import everything in the module, even the imports of the module, which prove may not be a good idea. Alternatively, if we intend that only ClassA, ClassB, and MethodA are ever to be imported and used in another module, we should specify that in parts.py using the __all__ keyword.

`parts.py`

```python
 __all__ = ['ClassA', 'ClassB', 'MethodA']
import numpy
import tensorflow

class ClassA:
    ...
```

Now, if someone does a wild import with `parts`, they can only import `ClassA`, `ClassB`, and `MethodA`. Personally, I also find __all__ helpful as it gives me an overview of the module

## 5 Decorator to time your functions

It is useful to know how long it takes a function to run, and one way to do this is to use `time.time()` at the begining and end of each function and print out the difference

For example, compare two algorithms to calculate the nth Fibonacci number, one uses memoization and one doesn't

```python
def fib_helper(n):
    if n < 2:
        return n
    return fib_helper(n - 1) + fib_helper(n - 2)

def fib(n):
    # a wrapper function so that any changes won't affect every reursive step
    return fib_helper(n)

def fib_m_helper(n, computed):
    if n in computed:
        return computed[n]
    computed[n] = fib_m_helper(n - 1, computed) + fib_m_helper(n - 2, computed)
    return computed[n]

def fib_m(n):
    return fib_m_helper(n, {0: 0, 1: 1})
```

Ensure `fib` and `fib_m` are functionally equivalent

```python
for n in range(20):
    assert fib(n) == fib_m(n)
```

```python
import time

start = time.time()
fib(30)
print(f'Without memoization, it takes {time.time() - start:7f} seconds.')

=> Without memoization, it takes 0.449216 seconds.
```

```python
start = time.time()
fib_m(30)
print(f'With memoization, it takes {time.time() - start:.7f} seconds.')

=> With memoization, it takes 0.0000992 seconds.
```

It can be inconvenience to write the same code over and over again if you want to time multiple functions. It'd be nice to have a way to specify how to change any function in the same way. In this case would be to call time.time() at the beginning and the end of each function, and print out the time difference

We can use decorators to do this. Here is an example to create a decorator timeit

```python
def timeit(fn): 
    # *args and **kwargs are to support positional and named arguments of fn
    def get_time(*args, **kwargs): 
        start = time.time() 
        output = fn(*args, **kwargs)
        print(f"Time taken in {fn.__name__}: {time.time() - start:.7f}")
        return output  # make sure that the decorator returns the output of fn
    return get_time
```

Add the decorator `@timeit` to your functions

```python
@timeit
def fib(n):
    return fib_helper(n)

@timeit
def fib_m(n):
    return fib_m_helper(n, {0: 0, 1: 1})

fib(30)
fib_m(30)

=> Time taken in fib: 0.2929940
=> Time taken in fib_m: 0.0000172
```

## 6 Caching with @functools.lru_cache

Memoization is a form of cache: we cache the previously calculated Fibonacci numbers so that we don't have to calculate them again.

Caching is such an important technique that Python provides a built-in decorator to give your function the caching capacity. If you want `fib_helper` to reuse the previously calculated Fibonacci numbers, you can just add the decorator `lru_cache` from `functools.lru` stands for "least recently used"

```python
import functools

@functools.lru_cache()
def fib_helper(n):
    if n < 2:
        return n
    return fib_helper(n - 1) + fib_helper(n - 2)

@timeit
def fib(n):
    # a wrapper function so that any changes won't affect every reursive step
    return fib_helper(n)

fib(50)
fib_m(50)

=> Time taken in fib: 0.0000420
=> Time taken in fib_m: 0.0001299
```