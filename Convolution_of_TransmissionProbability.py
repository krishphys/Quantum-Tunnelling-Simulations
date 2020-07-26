# Krishnan Ganesh 14305448
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rc
from scipy.ndimage.filters import gaussian_filter

# Setting fig properties
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

#%% Initial wave-vectors in reciprocal metres:
k_0 = np.linspace(0.0001 , 10 , 400)*1e10
# Initial wave-vector in reciprocal angstroms:
k_0_s = k_0 * L

#%% Defining Potential barrier params:
# Width of barrier in metres:
d = 5e-10
# Width of barrier in Angstroms:
d_s = d/L
# Select the average energy given by the maximum value in k_0 array:
E_0_max = ((h_bar*np.amax(k_0))**2)/(2*m)
# Set Potential energy of barrier to 0.1*E_0 to study bound states and scattering states:
U = 0.1*E_0_max
#%% Plotting the theoretical transmission function:
E_ratio = ((h_bar*k_0)**2)/(2*m)/U

# Splitting the plot into E_ratio < 1 and E_ratio> 1:
T = np.zeros(len(E_ratio))

for i in range(len(E_ratio)):
    if E_ratio[i]<1:
        Kappa2 = (np.sqrt(2*m*U*(1-E_ratio[i])))/h_bar
        T[i] = 1/(1 +((np.sinh(Kappa2*d))**2)/(4*E_ratio[i]*(1-E_ratio[i])))
    else :
        Kappa1 = (np.sqrt(2*m*U*(E_ratio[i]-1)))/h_bar
        T[i] = 1/(1 +((np.sin(Kappa1*d))**2)/(4*E_ratio[i]*(E_ratio[i]-1)))
        

plt.figure(figsize = (7.4,4.4))
plt.subplot( 1 , 2 , 1)
plt.plot(k_0_s , T)
plt.xlim([0 , np.max(k_0_s)])
plt.ylim(bottom = 0)
plt.ylabel('$T(k)$')
plt.xlabel('$k/\AA ^{-1}$')
plt.title('Not Convolved')
plt.grid()

#%% Gaussian Wave-packet in k-space and transmission function upon applying Gaussian filter:
k = np.linspace(0.0001 , 10 , 400)
gauss_k = (2*sigma_s/np.sqrt(2*np.pi))*np.exp(-2*(sigma_s*(k-2))**2)
convolved = gaussian_filter(T , sigma = 17)

# Plotting the convolved curves
plt.subplot(1 , 2, 2)
plt.plot(k , convolved , label= '$P_{transmit}$')
plt.plot(k , gauss_k , label = '$|\Phi(k)|^2$')
plt.xlim([0 , np.max(k)])
plt.ylabel('Probability')
plt.xlabel("$k_0/ \AA ^{-1}$")
plt.title("Convolved")
plt.legend(fontsize  = 11)
plt.grid()

plt.subplots_adjust(left = None , bottom = None , right = None , top = None , wspace = 0.25 , hspace= None)


