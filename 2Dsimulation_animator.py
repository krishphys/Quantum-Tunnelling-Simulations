# Krishnan Ganesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

# Initialise values:
sigma = 0.2
x_0 = -5
y_0 = -5
h_bar = 1
m = 1
k_0x = 3
k_0y = 3
k_0 = np.array([k_0x , k_0y])

N = 100
xval = np.linspace(-20 , 20 , N)
yval = np.linspace(-20 , 20 , N)
x , y = np.meshgrid(xval , yval , sparse = True)
delta = xval[1]-xval[0]

exponent = -((x - x_0)**2 + (y - y_0)**2) / (4*(sigma)**2)

# Gaussian wavepacket, t = 0:
psi_0 = (1/(sigma*(2*np.pi)**0.25))*np.exp(exponent)*np.exp(1j*(k_0[0]*x + k_0[1]*y))

# Load data:
all_data = np.load('Anim_Data_200frame_UBarrier_U=3.npy')
num_frame = 200
fig, ax = plt.subplots(1, figsize=(10, 10))
fig.subplots_adjust(0, 0, 1, 1)
ax.axis("off")

# Using matplotlib animation routine to stitch up the frames together:
def init():
    wave.set_array([])
    return wave

def animate(frame):
    data = all_data[frame]
    wave.set_array(data.ravel())
    return wave

# Plotting the frames as heatmaps of probability density:
wave = ax.pcolormesh(x , y , (np.abs(psi_0))**2 , cmap = cm.hot , vmin = 0, vmax = 0.5, shading = 'gouraud')

animation = FuncAnimation(fig, animate, np.arange(num_frame), fargs=[],interval=1000 / 25)   
animation.save("2dwavepacket_200frame_UBarrier_U=3.mp4", dpi=100)  
    















