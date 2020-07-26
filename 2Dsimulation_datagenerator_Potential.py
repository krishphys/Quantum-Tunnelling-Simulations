# Krishnan Ganesh 
import numpy as np
from scipy.integrate import ode
import time


# Initialise values:
sigma = 0.5
x_0 = -5
y_0 = -5
h_bar = 1
m = 1
k_0x = 3
k_0y = 3
k_0 = np.array([k_0x , k_0y])
U = 5
d = 2

# Create Meshgrid:
N = 100
xval = np.linspace(-20 , 20 , N)
yval = np.linspace(-20 , 20 , N)
x , y = np.meshgrid(xval , yval , sparse = True)

# Define potential wall params:
y_U = np.nonzero((yval > -(d/2)) & (yval < (d/2)))
U_matrix = np.zeros(shape = (N , N))
U_matrix[y_U,:] = U

delta = xval[1]-xval[0]

# Gaussian wavepacket, t = 0:
exponent = -((x - x_0)**2 + (y - y_0)**2) / (4*(sigma)**2)
psi_0 = (1/(sigma*(2*np.pi)**0.25))*np.exp(exponent)*np.exp(1j*(k_0[0]*x + k_0[1]*y))


kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])

# Applying the kernel to every point in the meshgrid:
def M(psi):
    if psi.shape != (N, N):
        psi = psi.reshape((N,N))
    
    out = np.zeros((N , N) , dtype = complex)
    for a in range(N):
        for b in range(N):
            for i in range(-1*min([a , 1]) , min([N-1-a ,1])+1):
                for j in range(-1*min([b , 1]) , min([N-1-b , 1])+1):
                    out[a , b] += kernel[i+1 , j+1] * psi[a+i , b+j]
    return(out.ravel())
# The out put mesh grid must be 'ravelled' in order to zvode to solve the diff equation.           

# Hamiltonian:
def f(t , psi):
    flattened_array = np.ravel(M(psi))
    potential_term = np.multiply(psi.reshape(N,N) , U_matrix)
    Lap = flattened_array/delta**2
    Ham = -((h_bar**2) /(2*m))*Lap + potential_term.ravel()
    return ((-1*1j)/h_bar)*Ham
    
 
r = ode(f).set_integrator('zvode')

r.set_initial_value(psi_0.ravel() , 0)

# Final time and time steps:
tf = 10
dt = 0.05

# Create an empty matrix to store all the data:
all_data = []

# Generate data for each time step and save to all_data:
while r.successful() and r.t<tf :
    start3 = time.time()
    r.integrate(r.t+dt)
    end3 = time.time()
    print('Integrating time:' , end3 - start3)
    matrix = np.reshape(r.y,(N, N))
    phi_sqrd = np.square(np.abs(matrix))
    all_data.append(phi_sqrd)

# Save all_data to file
np.save('Anim_Data_200frame_UBarrier_U=3.npy',all_data)

















