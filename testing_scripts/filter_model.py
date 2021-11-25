from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np

x = image.imread('dog_small.jpg')
x = np.average(x, axis=2)
x_flat = x.flatten()
m = x_flat.shape[0]
h = np.random.standard_normal((m,m))
y_flat = np.matmul(h,x_flat)
y = y_flat.reshape(x.shape)

x_hat = y
h_hat = np.eye(m)




fig, axs = plt.subplots(1, 2)
axs[0].imshow(x, cmap='gray')
axs[1].imshow(y, cmap='gray')
plt.show()