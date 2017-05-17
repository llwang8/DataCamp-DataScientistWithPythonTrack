# DataCamp


# Write a generator function which returns the Fibonacci series.
def fib():
    a, b = 1, 1
    while 1:
        yield a
        a, b = b, a+b

import types
if type(fib()) == types.GeneratorType:
    print("The fib function is a generator.")

    counter = 0
    for n in fib():
        print(n)
        counter += 1
        if counter == 10:
            break

# List Comprehensions
# create a list of integers which specify the length of each word
# in a certain sentence, but only if the word is not the word "the".

sentence = "the quick brown fox jumps over the lazy dog"
words_lengths = [len(word) for word in sentence.split() if word != 'the']

print(word_lengths)


#Using a list comprehension, create a new list called "newlist" out of the
#list "numbers", which contains only the positive numbers from the list, as integers.



# Exercise: make a regular expression that will match an email
import re
def test_email(your_pattern):
    pattern = re.compile(your_pattern)
    emails = ["john@example.com", "python-list@python.org", "wha.t.`1an?ug{}ly@email.com"]
    for email in emails:
        if not re.match(pattern, email):
            print("You failed to match %s" % (email))
        elif not your_pattern:
            print("Forgot to enter a pattern!")
        else:
            print("Pass")
# Your pattern here!
pattern = r"\"?([-a-zA-Z0-9.`?{}]+@\w+\.\w+)\"?"
test_email(pattern)


import json

# fix this function, so it adds the given name
# and salary pair to salaries_json, and return it
def add_employee(salaries_json, name, salary):
    salaries = json.loads(salaries_json)
    salaries[name] = salary
    return json.dumps(salaries)

# test code
salaries = '{"Alfred" : 300, "Jane" : 400 }'
new_salaries = add_employee(salaries, "Me", 800)
decoded_salaries = json.loads(new_salaries)
print(decoded_salaries["Alfred"])
print(decoded_salaries["Jane"])
print(decoded_salaries["Me"])

# set()
a = set(["Jake", "John", "Eric"])
b = set(["John", "Jill"])

set(['John'])
a.intersection(b)
set(['John'])
b.intersection(a)
set(['John'])

a.symmetric_difference(b)
set(['Jill', 'Jake', 'Eric'])
b.symmetric_difference(a)
set(['Jill', 'Jake', 'Eric'])

a.difference(b)
set(['Jake', 'Eric'])
b.difference(a)
set(['Jill'])

a.union(b)
set(['Jill', 'Jake', 'John', 'Eric'])


# partial funtion
from functools import partial
def func(u,v,w,x):
    return u*4 + v*3 + w*2 + x

p = partial(func,5,6,7)
print(p(8))

# code introspection

help()
dir()
hasattr()
id()
type()
repr()
callable()
issubclass()
isinstance()
__doc__
__name__

# Closures
def multiplier_of(n):
    def multiplier(number):
        return number*n
    return multiplier

multiplywith5 = multiplier_of(5)
print(multiplywith5(9))


