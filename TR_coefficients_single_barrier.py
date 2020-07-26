# Krishnan Ganesh 14305448
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rc
from scipy.sparse import spdiags
from scipy.integrate import ode
import time

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
sigma = (1)*1e-10
# Sigma in Angstroms
sigma_s = sigma / L
# Initial position of wave packet: 
x_i = -(0.05 * 1e-8)
x_i_s = x_i / L
# Initial wave-vectors in reciprocal metres:
# =============================================================================
# #Change 'k' to change the maximum wavenumber the program will calculate for:
# =============================================================================
k = 10
k_0 = np.linspace(0 , k , 100)*1e10

# Initial wave-vector in reciprocal angstroms:
k_0_s = k_0 * L
# Energy scale:
e = (2*m*(alph**2))/(h_bar**2)
# Introduce a time scale:
t_0 = (2*np.pi*h_bar)/e

#%% Defining Potential barrier params:
# Width of barrier in metres:
d = 5e-10
# Width of barrier in Angstroms:
d_s = d/L
# Select the average energy given by the maximum value in k_0 array:
E_0_max = ((h_bar*np.amax(k_0))**2)/(2*m)
# Set Potential energy barrier height in terms of maximum wave-packet energy:
# =============================================================================
# # Change 'f' to vary the potential barrie height:
# =============================================================================
f  = 0.1
U = f*E_0_max
# Potential in units of energy scale 'e':
U_s = U / e

#%% X- Domain:
N = 5000
x_bound = 5*1e-8
x = np.linspace(-x_bound , x_bound , N)
# Scaling x domain to angstroms units:
x_s = x / L
delta_x = x_s[1]-x_s[0]

# Defining indicies that fall within the potential barrier region: 
x_U = np.logical_and(x_s>-d_s/2 , x_s<d_s/2)
# Indices before and after barrier:
x_before = np.nonzero(x_s <= -d_s/2)
x_after = np.nonzero(x_s >= d_s/2)


#%% Gaussian wavepacket, t = 0:
def psi_0(k_0):
    exponent = -((x_s - x_i_s)**2) / (4*(sigma_s)**2)
    psi_0 = (1/(np.sqrt(sigma_s)*(2*np.pi)**0.25))*np.exp(exponent)*np.exp(1j*k_0*x_s)
    return(psi_0)

#%% Matrices for Hamiltonian:
# 2nd Derivative Matrix:
diagonals = np.array([np.repeat(1 , N), np.repeat(-2 , N) , np.repeat(1 , N)])
positions = np.array([-1 , 0 ,1 ])
M = spdiags(diagonals , positions , N ,N)/(delta_x**2)

# Kinetic contribution:
T = 2*1j*np.pi*M

# Potential contribution:
V = -2*1j*np.pi*spdiags(U_s*x_U , 0 , N , N)   

# 'Hamiltonian' Matrix:
H = T + V

#%% Discretized Schrodinger equation:
def f(t , psi):
    return H.dot(psi)

# Setting integration time:
tf = (50*1e-16)/t_0

    
#%% Probability arrays:
P_reflect = np.zeros(len(k_0_s))
P_transmit = np.zeros(len(k_0_s))
# Probability of wave-packet inside barrier region:
P_barrier = np.zeros(len(k_0_s))

#%% Calculating transmission, reflection and barrier probabilities for each k_0 value:
for i in range(len(k_0_s)):   
    start = time.time()                                                                                                                                                                
    r = ode(f).set_integrator('zvode' , nsteps = 20000)
    r.set_initial_value(psi_0(k_0_s[i]) , 0)
    phi = r.integrate(tf)
    phi_sqrd = (np.abs(phi))**2
    P_reflect[i] = np.sum(phi_sqrd[x_before] * delta_x)
    P_transmit[i] = np.sum(phi_sqrd[x_after] * delta_x)
    P_barrier[i]  = np.sum(phi_sqrd[x_U.nonzero()] * delta_x)
    end = time.time()
    print(end - start, "Done" , i)

#%% Plotting the probabiities
plt.figure(figsize = (6 , 5))
E_ratio = ((h_bar*k_0)**2)/(2*m)/U
P_tot = P_reflect + P_transmit + P_barrier
plt.plot(k_0_s , P_reflect , label = '$P_{reflect}$')
plt.plot(k_0_s , P_transmit , label = '$P_{transmit}$')
plt.plot(k_0_s , P_barrier , label = '$P_{barrier}$')
plt.plot(k_0_s , P_tot , label = '$\Sigma P$')
plt.axvline(x = 1 , linestyle = '--' , color = 'grey')
plt.ylabel('Probability')
plt.xlabel('$k_0 / \AA ^{-1}$')
plt.grid()
plt.title('Transmission probabilities against $k_0$')
plt.legend(fontsize = 11)






