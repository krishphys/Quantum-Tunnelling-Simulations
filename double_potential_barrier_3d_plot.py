# Krishnan Ganesh 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.collections import LineCollection
from scipy.sparse import spdiags
from scipy.integrate import ode

# Setting fig properties:
rc('text' , usetex = True)
sns.set_context('paper', rc={'font.size': 14,'axes.titlesize': 14,'axes.labelsize': 14, \
                             'xtick.labelsize' : 11, 'ytick.labelsize': 11})
sns.set_style('ticks')

#%% Fundamental constants
h_bar = 1.055e-34
m = 9.11e-31 # We shall consider an electron
alph =  (h_bar**2) / (2*m*(1e-10))
# Initialise values:
#Length Scale set to be 1 Angstrom:
L = h_bar**2 / (2*m*alph)
#Sigma in metres:
sigma = 1e-10
# Sigma in Angstroms
sigma_s = sigma / L
# Initial position of wave packet: 
x_i = -(0 * 1e-8)
x_i_s = x_i / L
# Initial wave-vector in reciprocal metres:
# =============================================================================
# # Change 'k' to vary the average number of the wave-packet:
# =============================================================================
k = 3
k_0 = k*1e10
# Initial wave-vector in reciprocal angstroms:
k_0_s = k_0 * L
# Energy scale:
e = (2*m*(alph**2))/(h_bar**2)
# Introduce a time scale:
t_0 = (2*np.pi*h_bar)/e

#%% Defining Potential barrier params:
# Width of each barrier in metres:
d = 1e-10
# Width of barrier in Angstroms:
d_s = d/L
# Separation between centres of barriers:
w = 20e-10
# Separation in Angstroms:
w_s = w / L
# Average energy of particle:
E_0 = ((h_bar*(1e10))**2)/(2*m)
# Potential energy of barrier:
# =============================================================================
# #Change 'f' to change the potential barrier height:
# =============================================================================
f = 10
U = (f) * E_0
# Potential in units of energy scale 'e':
U_s = U / e

#%% X-Domain:
N = 7000
x_bound = 5e-8
x = np.linspace(-x_bound , x_bound , N)
# Scaling X-Domain to angstrom units:
x_s = x/L

delta_x = x_s[1]-x_s[0]
# Logical arrays to define the potential energy at each point on the x axis
x_U1 = np.logical_and( x>(-1*(w+d)/2) , x<(-1*(w-d)/2) )
x_U2 = np.logical_and(x> ((w-d)/2) , x< ((w+d)/2) )
x_U = x_U1 + x_U2

#%% Gaussian wavepacket, t = 0:
exponent = -((x_s - x_i_s)**2) / (4*(sigma_s)**2)
psi_0 = (1/(np.sqrt(sigma_s)*(2*np.pi)**0.25))*np.exp(exponent)*np.exp(1j*k_0_s*x_s)

#%% Matrices for Hamiltonian:
# 2nd Derivative Matrix:
diagonals = np.array([np.repeat(1 , N), np.repeat(-2 , N) , np.repeat(1 , N)])
positions = np.array([-1 , 0 ,1 ])
M = spdiags(diagonals , positions , N ,N)/(delta_x**2)

# Kinetic Contribution:
T = (2j*np.pi)*M

# Potential Contribution:
V = (-2*1j*np.pi)*spdiags(U_s *x_U , 0 , N , N)
# 'Hamiltonian' Matrix:
H = T + V


#%% Discretized Schrodinger equation:
def f(t , psi):
    return H.dot(psi)

r = ode(f).set_integrator('zvode', nsteps = 5000)
r.set_initial_value(psi_0 , 0)

#%% Setting final time and time steps:
tf = (40*1e-16) / t_0
dt = (4*1e-16) / t_0

#%% Plotting wave function at final time, and 3D time evolution plot:
fig = plt.figure(figsize=(8,8))
ax1 = plt.axes()
psi_0_wavfunc = (np.abs(psi_0))**2
axes, = plt.plot(x_s , psi_0_wavfunc , color = 'red')


ax1.set_xlabel(r'$\tilde{x}$')
ax1.set_ylabel(r'$|\psi(\tilde{x})|^2$')
ax1.set_title('Wave Packet on Double Potential Barrier')
ax1.set_xlim([-0.5*x_bound/L,0.5*x_bound/L])
ax1.set_ylim(bottom = 0)
plt.axvline(x = -(w_s+d_s)/2 , linestyle='--')
plt.axvline(x = -(w_s-d_s)/2 , linestyle= '--')
plt.axvline(x =  (w_s+d_s)/2 , linestyle='--')
plt.axvline(x =  (w_s-d_s)/2 , linestyle= '--')

plt.axvspan(-(w_s+d_s)/2 , -(w_s-d_s)/2 , color = 'blue' , alpha = 0.3 )
plt.axvspan((w_s-d_s)/2 , (w_s+d_s)/2 , color = 'blue' , alpha = 0.3 )

# Storing line data for each time slice:
verts = []
# Truncating x array so that lines dont extend beyong the axes in the 3D plot:
x_trunc = x_s[ np.logical_and(x_s>-0.1*x_bound/L , x_s < 0.1*x_bound/L )]
vert1 = list(zip(x_trunc , psi_0_wavfunc[np.nonzero(np.logical_and(x_s>-0.1*x_bound/L , x_s < 0.1*x_bound/L ))] ))
verts.append(vert1)
while r.successful() and r.t < tf and plt.fignum_exists(1):
    r.integrate(r.t+dt)
    phi_sqrd = (np.abs(r.y))**2
    axes.set_data(x_s , phi_sqrd )
    verts.append(list(zip(x_trunc,phi_sqrd[np.nonzero(np.logical_and(x_s>-0.1*x_bound/L , x_s < 0.1*x_bound/L ))])))

# Plotting the potential barriers at each time slice in the 3D plot:
x_U1tr = np.logical_and( x_trunc>(-1*(w_s+d_s)/2) , x_trunc<(-1*(w_s-d_s)/2) )
x_U2tr = np.logical_and(x_trunc> ((w_s-d_s)/2) , x_trunc< ((w_s+d_s)/2) )
x_Utr = x_U1tr + x_U2tr
pot = 0.4* x_Utr
vert_pot = [list(zip(x_trunc , pot))]*12

fig = plt.figure(figsize = (8 , 8))
t = np.linspace(0 , 11 , 12)*dt
ax = fig.gca(projection='3d')
poly1 = LineCollection(verts, colors = 'k' , linewidth = 0.9 , alpha = 0.9)
poly2 = PolyCollection(vert_pot, facecolors = 'w' , alpha = 0.8 , edgecolor = 'grey')

ax.add_collection3d(poly1, zs=t, zdir='y')
ax.add_collection3d(poly2 , zs = t , zdir = 'y')

ax.set_xlabel(r'$\tilde{x}$')
ax.set_xlim3d(-0.1*x_bound/L,0.1*x_bound/L)

ax.set_zlabel(r'$|\psi(\tilde{x})|^2$', labelpad = 4)
ax.set_zlim3d(0 , 0.45)

ax.set_ylabel(r'$\tilde{t}$')
ax.set_ylim3d(0 , np.max(t))

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.5, 2, 0.5, 1]))    

plt.show()








        
    
    
    
    
    

