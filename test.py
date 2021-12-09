import matplotlib.pylab as plt
import numpy as np

print(0.8e-3*2*np.pi/(1e-9*952.341181))
print(0.07e-3*2*np.pi/(1e-9*2515.672563))

exit()
x = np.linspace(0,10,100)
y = np.sin(x)

f, ax = plt.subplots()

ax.plot(x, y)

def draw_brakes(ax, x_pos, d=0.015):
    for a in [-0.01, 0.01]:
        pos = (x_pos + a)
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot(((pos-d),(pos+d)), (-d,+d), **kwargs)
        ax.plot(((pos-d),(pos+d)),(1-d,1+d), **kwargs)

draw_brakes(ax, 0.5)
plt.show()