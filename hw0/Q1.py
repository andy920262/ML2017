import sys
import numpy as np

A = []
B = []

with open(sys.argv[1], 'r') as file1:
    for line in file1:
        A.append([int(x) for x in line.split(',')])

with open(sys.argv[2], 'r') as file2:
    for line in file2:
        B.append([int(x) for x in line.split(',')])

for i in np.sort(np.dot(np.array(A), np.array(B)).ravel()):
    print(i)

