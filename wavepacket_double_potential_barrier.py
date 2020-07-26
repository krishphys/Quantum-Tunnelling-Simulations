#Krishnan 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rc
from matplotlib import gridspec
from scipy.sparse import spdiags
from scipy.integrate import ode

# Setting figure properties:
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
x_i = -(0.1 * 1e-8)
x_i_s = x_i / L
# Initial wave-vector in reciprocal metres:

# =============================================================================
# #Change 'k' to change the average wavenumber:
# =============================================================================
k = 1
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
w = 10e-10
# Separation in Angstroms:
w_s = w / L
# Average energy of particle:
E_0 = ((h_bar*(1e10))**2)/(2*m)
# Potential energy of barrier:

# =============================================================================
# #Change 'f' to change the barrier height:
# =============================================================================
f = 2
U = (f) * E_0

# Potential in units of energy scale 'e':
U_s = U / e

#%% X-Domain:
N = 1000
x_bound = 1e-8
x = np.linspace(-x_bound , x_bound , N)
# Scaling X-Domain to angstrom units:
x_s = x/L

delta_x = x_s[1]-x_s[0]
# Logical arrays to define the potential energy at each point on the x axis
x_U1 = np.logical_and( x>(-1*(w+d)/2) , x<(-1*(w-d)/2) )
x_inbetwn = np.logical_and(x_s > -1*(w_s-d_s)/2 , x_s< (w_s-d_s)/2)
x_U2 = np.logical_and(x> ((w-d)/2) , x< ((w+d)/2) )
x_U = x_U1 + x_U2


#%% Gaussian wavepacket, t = 0:
exponent = -((x_s - x_i_s)**2) / (4*(sigma_s)**2)
psi_0 = (1/(np.sqrt(sigma_s)*(2*np.pi)**0.25))*np.exp(exponent)*np.exp(1j*k_0_s*x_s)

#%% Matrices for Hamiltonian:
# 2nd Derivative:
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


r = ode(f).set_integrator('zvode')
r.set_initial_value(psi_0 , 0)

# Setting final time and time steps:
tf = (30*1e-16) / t_0
dt = (0.05*1e-16) / t_0

#%% Animating and plotting
fig = plt.figure(figsize=(8,8))
gs  = gridspec.GridSpec(2, 1, height_ratios=[0.75 , 0.25])
ax1 = plt.subplot(gs[0])
psi_0_wavfunc = (np.abs(psi_0))**2
axes, = plt.plot(x_s , psi_0_wavfunc , color = 'red')
ax1.set_xlabel(r'$\tilde{x}$')
ax1.set_ylabel(r'$|\psi(\tilde{x})|^2$')
ax1.set_title('Wave Packet on Double Potential Barrier')
ax1.set_xlim([-0.5*x_bound/L,0.5*x_bound/L])
ax1.set_ylim(bottom = 0)
# Plotting lines to mark the potential barrier region
plt.axvline(x = -(w_s+d_s)/2 , linestyle='--')
plt.axvline(x = -(w_s-d_s)/2 , linestyle= '--')
plt.axvline(x =  (w_s+d_s)/2 , linestyle='--')
plt.axvline(x =  (w_s-d_s)/2 , linestyle= '--')

plt.axvspan(-(w_s+d_s)/2 , -(w_s-d_s)/2 , color = 'blue' , alpha = 0.3 )
plt.axvspan((w_s-d_s)/2 , (w_s+d_s)/2 , color = 'blue' , alpha = 0.3 )

# Separate plot to show the potential barrier function:
ax2 = plt.subplot(gs[1])
potential = U_s * x_U
ax2.plot(x_s , potential , label = r'$\tilde{V}$')
ax2.set_xlim([-0.5*x_bound/L , 0.5*x_bound/L])
ax2.set_ylabel(r'$\tilde{V}$')
E = (1/(2*m))*(h_bar*k_0)**2
plt.axhline(y = E/e , color= 'r' , linestyle = '-' , label = r'$\tilde{E_0}$')
plt.legend()

while r.successful() and r.t < tf and plt.fignum_exists(1):
    r.integrate(r.t+dt)
    phi_sqrd = (np.abs(r.y))**2
    axes.set_data(x_s , phi_sqrd )
    plt.pause(0.01)









        
    
    
    
    
    

