# Krishnan Ganesh 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rc
from matplotlib import gridspec
from scipy.sparse import spdiags
from scipy.integrate import ode

# Set the figure properties for clear graphs:
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
x_i = -(0.05 * 1e-8)
x_i_s = x_i / L
# Initial wave-vector in reciprocal metres:
k_0 = 3*1e10
# Initial wave-vector in reciprocal angstroms:
k_0_s = k_0 * L
# Energy scale:
e = (2*m*(alph**2))/(h_bar**2)
# Introduce a time scale:
t_0 = (2*np.pi*h_bar)/e

#%% Defining Potential barrier params:

# Width of barrier in metres:
d = 1e-10
# Width of barrier in Angstroms:
d_s = d/L
# Avergae energy of particle:
E_0 = ((h_bar*(1e10))**2)/(2*m)
# Potential energy of barrier:
U = (10) * E_0
# Potential in units of energy scale 'e':
U_s = U / e


#%% X- Domain:

N = 2000
x_bound = 1e-8
x = np.linspace(-x_bound , x_bound , N)
# Scaling x domain to angstrom units:
x_s = x / L
delta_x = x_s[1]-x_s[0]
# Define logical array for all points before barrier:
x_before = np.nonzero(x_s <= -d_s/2)

# Defining indicies that fall within the potential barrier region: 
x_U = np.logical_and(x_s>-d_s/2 , x_s<d_s/2)

#%% Gaussian wavepacket, t = 0:

exponent = -((x_s - x_i_s)**2) / (4*(sigma_s)**2)
psi_0 = (1/((sigma_s**0.5)*(2*np.pi)**0.25))*np.exp(exponent)*np.exp(1j*k_0_s*x_s)

#%% Matrices for Hamiltonian:

# 2nd Derivative Matrix:
diagonals = np.array([np.repeat(1 , N), np.repeat(-2 , N) , np.repeat(1 , N)])
positions = np.array([-1 , 0 ,1 ])
M = spdiags(diagonals , positions , N ,N)/(delta_x**2)

# Kinetic contribution:
T = (2j*np.pi)*M

# Potential contribution:
V = (-2*1j*np.pi)*spdiags((U_s)*x_U , 0 , N , N)                                                                                                                                                                        
# 'Hamiltonian' Matrix:
H = T + V

#%% Discretized Schrodinger equation:

def f(t , psi):
    return H.dot(psi)


r = ode(f).set_integrator('zvode')
r.set_initial_value(psi_0 , 0)

# Setting the animation time, and time steps:
tf = (20*1e-16) / t_0
dt = (0.05*1e-16) / t_0

#%% Plotting and animating the result:

fig = plt.figure(figsize=(8,8))
gs  = gridspec.GridSpec(2, 1, height_ratios=[0.75 , 0.25])
ax1 = plt.subplot(gs[0])
psi_0_wavfunc = (np.abs(psi_0))**2
axes, = plt.plot(x_s , psi_0_wavfunc , color = 'red')
ax1.set_xlabel(r'$\tilde{x}$')
ax1.set_ylabel(r'$|\psi(\tilde{x})|^2$')
ax1.set_title('Gaussian Wave Packet impinging on potential barrier')
ax1.set_xlim([-0.5*x_bound/L,0.5*x_bound/L])
ax1.set_ylim(bottom = 0, top = 0.5)
plt.axvline(x = -d_s/2 , linestyle='--')
plt.axvline(x = +d_s/2 , linestyle= '--')

# Changine the color of the potential barrier region based on whether the potential 
# is negative or positive:
if U_s > 0:
    plt.axvspan(-d_s/2 , + d_s/2 , color = 'blue' , alpha = 0.3 )
elif U_s == 0:
    plt.axvspan(-d_s/2 , + d_s/2 , color = 'green' , alpha = 0 )
else:
    plt.axvspan(-d_s/2 , + d_s/2 , color = 'green' , alpha = 0.3 )

ax2 = plt.subplot(gs[1])
potential = (U_s) * x_U
ax2.plot(x_s , potential , label = r'$\tilde{V}$')
ax2.set_xlim([-0.5*x_bound/L , 0.5*x_bound/L])
ax2.set_ylabel(r'$\tilde{V}$')
E = (1/(2*m))*(h_bar*k_0)**2
plt.axhline(y = E/e , color= 'r' , linestyle = '-' , label = r'$\tilde{E_0}$')
plt.legend()

# Storing values for the probability to the left of the barrier as a function of time:
P_before = np.zeros(int(tf//dt)+2)
val = int(tf//dt)
t = 0
P_before[0] = np.sum(psi_0_wavfunc[x_before]*delta_x)
while r.successful() and r.t < tf and plt.fignum_exists(1):
    r.integrate(r.t+dt)
    phi_sqrd = (np.abs(r.y))**2
    axes.set_data(x_s , phi_sqrd )
    t+=1
    P_before[t] = np.sum(phi_sqrd[x_before]*delta_x) 
    plt.pause(0.01)
# Plotting probability to the left of barrier as function of time:
tval = np.linspace(0 , tf , int(tf//dt)+2)
plt.figure(figsize = (5,4))
plt.plot(tval , P_before)
plt.xlabel(r'$\tilde{t}$')
plt.ylabel(r'P$_{before}$')
plt.grid()

        
    
    
    
    
    


