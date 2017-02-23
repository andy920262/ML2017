import sys
import numpy as np

file1 = open(sys.argv[1], 'r')
file2 = open(sys.argv[2], 'r')

A = [int(x) for x in file1.readline().split(',')]
B = []

for i in range(50):
    B.append([int(x) for x in file2.readline().split(',')])

for i in np.sort(np.dot(np.array(A), np.array(B))):
    print(i)


