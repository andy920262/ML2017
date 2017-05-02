import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
import matplotlib.pyplot as plt
c = np.loadtxt(sys.argv[1], delimiter = ',')
plt.figure()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(np.transpose(c)[0], np.transpose(c)[1], label = 'train_acc')
plt.plot(np.transpose(c)[0], np.transpose(c)[3], label = 'valid_acc')
plt.legend()
plt.savefig(sys.argv[2])
