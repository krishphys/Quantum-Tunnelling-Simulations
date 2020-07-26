# Krishnan Ganesh 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from scipy.sparse import spdiags
from scipy.integrate import ode

# Setting figure properties for nice plots:
rc('text' , usetex = True)
sns.set_context('paper', rc={'font.size': 14,'axes.titlesize': 14,'axes.labelsize': 14, \
                             'xtick.labelsize' : 11, 'ytick.labelsize': 11})
sns.set_style('ticks')

#%% Fundamental constants

h_bar = 1.055e-34
m = 9.11e-31
alph =  (h_bar**2) / (2*m*(1e-10))
# Initialise values:
#Length Scale set to be 1 Angstrom:
L = h_bar**2 / (2*m*alph)
#Sigma in metres:
sigma = 1e-10
# Sigma in Angstroms
sigma_s = sigma / L
# Initial position of wave packet: 
x_i = -(0.25 * 1e-8)
x_i_s = x_i / L
# Initial wave-vector in reciprocal metres:
k_0 = 4*1e10
# Initial wave-vector in reciprocal angstroms:
k_0_s = k_0 * L
# Energy scale:
e = (2*m*(alph**2))/(h_bar**2)
# Introduce a time scale:
t_0 = (2*np.pi*h_bar)/e

#%% X-Domain

N = 3000
x_bound = 1e-8
x = np.linspace(-1*x_bound, x_bound , N)
# Dimensionless x values, scaled by the length of the domain:
x_s = x / L
delta_x = x_s[1]-x_s[0]

# Gaussian wavepacket, t = 0:
exponent = -(((x_s - x_i_s))**2) / (4*(sigma_s)**2)
psi_0 = (1/((sigma_s**0.5)*(2*np.pi)**0.25))*np.exp(exponent)*np.exp((1j*k_0_s*x_s))

#%% Matrices for Hamiltonian:
diagonals = np.array([np.repeat(1 , N), np.repeat(-2 , N) , np.repeat(1 , N)])
positions = np.array([-1 , 0 ,1 ])
M = spdiags(diagonals , positions , N ,N)/(delta_x**2)
# Hamiltonian Matrix after being non-dimensionalised:
H = 2j*np.pi*M

#%% Discretized Schrodinger equation:

def f(t , psi):
    return H.dot(psi)

r = ode(f).set_integrator('zvode' , nsteps = 5000)
r.set_initial_value(psi_0 , 0)

t_f = (20*1e-16)/t_0
dt = (0.05*1e-16) / t_0

# Analytical decay time of maximum probability amplitude:
tau = ((2*m*(sigma**2))/h_bar) / t_0

#%%PLotting and Animating
fig1= plt.figure(1 , figsize = (6 ,5))

psi_0_wavfunc = (np.abs(psi_0))**2
axes, = plt.plot(x_s , psi_0_wavfunc)
plt.xlabel(r'$\tilde{x}$')
plt.ylabel(r'$|\psi(\tilde{x})|^2$')
plt.title('Free Gaussian Wave Packet')
plt.xlim([-x_bound/L,x_bound/L])
plt.ylim(bottom = 0)

# Probability values to check normalisation:
P = np.zeros(int(np.floor(t_f / dt))+1)
P[0] = np.sum(psi_0_wavfunc*delta_x)

# Peak values of probability density:
Peak = np.zeros(int(np.floor(t_f / dt))+1)
Peak[0] = np.amax(psi_0_wavfunc)

i = 0
while r.successful() and r.t < t_f and plt.fignum_exists(1):
    r.integrate(r.t+dt)
    phi_sqrd = (np.abs(r.y))**2
    axes.set_data(x_s , phi_sqrd)
    P[i] = np.sum(phi_sqrd * delta_x)
    Peak[i] = np.amax(phi_sqrd)
    i +=1
    plt.pause(0.01)

# Plotting the total probability against time:
t = np.linspace(0 , t_f , int(np.floor(t_f/dt)+1))
plt.figure(figsize = (6,5))
plt.plot(t , P)
plt.xlim([0 , t_f])
plt.ylim(bottom = 0)
plt.xlabel(r'$\tilde{t}$')
plt.ylabel(r'$\int|\psi(\tilde{x})|^2 d\tilde{x}$')
plt.ylim([0 , 2])
txt = r'$\int|\psi(\tilde{x})|^2 d\tilde{x}$'
plt.title( txt + ' against time')

# Plotting the numerical and analytic values of the peak of the wave-packet against time:
plt.figure(figsize = (7,5))
plt.tight_layout()
plt.subplot(1 , 2 , 1)
for i in range(6):
    t_mark = 2 * i * tau
    txt = str(str(2*(i)) +  r'$\tilde{\tau}$')
    plt.axvline(x = t_mark , color = 'r' , linestyle = '--')
    plt.annotate(txt , xy=(t_mark , 0.02), xycoords = 'data' , xytext=(t_mark + 0.03 , 0.01) )
plt.plot(t , Peak )
plt.xlim([0 , t_f])
plt.ylim(bottom = 0)
plt.xlabel(r'$\tilde{t}$') 
plt.ylabel(r'$|\psi(\tilde{x})|^2 _{max}$' )
plt.title(r'Numerical')  
plt.grid()

plt.subplot(1 , 2 , 2)
Peak_0 = 1 /( sigma_s * np.sqrt(2*np.pi))
y = (1 / (np.sqrt(1 + (t/tau)**2))) * Peak_0 
plt.plot(t , y)
for i in range(6):
    t_mark = 2 * i * tau
    txt = str(str(2*(i)) +  r'$\tilde{\tau}$')
    plt.axvline(x = t_mark , color = 'r' , linestyle = '--')
    plt.annotate(txt , xy=(t_mark , 0.02), xycoords = 'data' , xytext=(t_mark + 0.03 , 0.01) )

plt.xlabel(r'$\tilde{t}$')
plt.xlim([0 , t_f])
plt.ylim(bottom = 0)
plt.title(r'Analytical')
plt.grid()    
    
    
