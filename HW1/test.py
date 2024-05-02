import collections
import math
from typing import Any, DefaultDict, List, Set, Tuple

def gen_perm(perm: list, words: list, temp, count: int, counter):
    if count == 0:
        perm.append(temp)
        return
    elif temp == "":
        for word in words:
            if counter[word] > 0:
                counter[word] -= 1
                gen_perm(perm, words, word, count - 1, counter)
                counter[word] += 1
    else:
        for word in words:
            if counter[word] > 0:
                counter[word] -= 1
                gen_perm(perm, words, word + " " + temp, count - 1, counter)
                counter[word] += 1

    return perm

sentence = "a a a a a"
words = sentence.split()
counter = collections.Counter(words)
perm = []
count = len(words)
temp = ""

perm = set(gen_perm(perm, words, temp, count, counter))

pairs = []
for i in range(count-1):
    pairs.append(words[i]+" "+words[i+1])

result=[]
for data in perm:
    for pair in pairs:
        if pair in data: result.append(data)

print(set(result))
