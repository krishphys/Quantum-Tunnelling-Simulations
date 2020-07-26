# Krishnan Ganesh 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.pyplot import rc
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
k_0 = np.linspace(0 , 3 , 5)*1e10
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
# Set an energy of the potential barrier using the average energy of a particle:
E_0 = ((h_bar*(1e10))**2)/(2*m)
# Potential energy of barrier:
U = 10*E_0
# Potential in units of energy scale 'e':
U_s = U / e

#%% X-Domain:
N = 8000
x_bound = 5e-8
x = np.linspace(-x_bound , x_bound , N)
# Scaling X-Domain to angstrom units:
x_s = x/L

delta_x = x_s[1]-x_s[0]

# Creating logical arrays to definethe extent of the potential barriers:
x_U1 = np.logical_and(x_s>(-1*(w_s+d_s)/2) , x_s<(-1*(w_s-d_s)/2) )
x_U2 = np.logical_and(x_s> ((w_s-d_s)/2) , x_s< ((w_s+d_s)/2) )
x_inbetwn = np.logical_and(x_s > -1*(w_s-d_s)/2 , x_s< (w_s-d_s)/2)
x_U = x_U1 + x_U2
x_before = np.nonzero(x_s <= -1*(w_s+d_s)/2)
x_after = np.nonzero(x_s >= (w_s+d_s)/2)


#%% Gaussian wavepacket, t = 0:
exponent = -((x_s - x_i_s)**2) / (4*(sigma_s)**2)
def psi_0(k):
    psi_0 = (1/(np.sqrt(sigma_s)*(2*np.pi)**0.25))*np.exp(exponent)*np.exp(1j*k*x_s)
    return(psi_0)


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

#%% Polynomial fitting and plotting of decay profiles:

tf = (200*1e-16) / t_0
t = np.linspace(0.01 , tf , 100)

# Coefficients for polynomial fitting routine:
b_array = np.zeros(len(k_0))
c_array = np.zeros(len(k_0))

plt.figure(figsize = (6 , 5))  
for j in range(len(k_0)):
    r.set_initial_value(psi_0(k_0_s[j]), 0)
    P_barrier = np.zeros(len(t))
    for i in range(len(t)):
        start = time.time()
        final_time = t[i]
        phi = r.integrate(final_time)
        phi_sqrd = (np.abs(phi))**2
        P_barrier[i] = np.sum(phi_sqrd[x_inbetwn.nonzero()]*delta_x)
        end = time.time()
        print('Integrating time:' , end - start)
    text = str(r'$k_0$ =' +  str(k_0_s[j]) + r'\AA$^{-1}$')
    logprob = np.log(P_barrier)
    # Applying quadratic order polynomial fitting to each decay profile:
    P_fit = np.polyfit(t , logprob , 2)
    b_array[j] = P_fit[1]
    c_array[j] = P_fit[2]
    plt.plot(t , logprob , label = text )

# Plotting the log of the probability against time:        
plt.xlabel(r'$\tilde{t}$')
plt.ylabel('ln(P$_{gap}$)')
plt.legend(fontsize = 11)
plt.grid()

# Plotting coeffieicnt values against incident wave-packet wavenumber:
plt.figure(figsize = (6 , 5))
plt.plot(k_0_s , b_array, label = r'$b$')
plt.plot(k_0_s , c_array , label = r'$c$')
plt.xlabel(r'$k_0$/ \AA $^{-1}$')
plt.ylabel('Coefficient values')
plt.grid()
plt.legend()







