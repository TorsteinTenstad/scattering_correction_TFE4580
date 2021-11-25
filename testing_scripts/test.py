import numpy as np

a = [1, 2, 3, 4, 5, 12, 34, 12, 45, 23, 6, 23, 4, 23]

x = np.argpartition(a, -4)[-4:]

print(x)
print([a[i] for i in x])