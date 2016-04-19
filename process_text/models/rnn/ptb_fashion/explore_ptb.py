
import collections
import os
print os.getcwd()
file_name = '../../../../simple-examples/data/ptb.train.txt'

with open(file_name, 'r') as f:
    text = f.read().replace('\n', "<eos>")

data = text.split()

counter = collections.Counter(data)
count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

words, _ = list(zip(*count_pairs))
word_to_id = dict(zip(words, range(len(words))))

# len(words)  887,521

# after replacing \n by eos, we are left with 929,589

print "here"