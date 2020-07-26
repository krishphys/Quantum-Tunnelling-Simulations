# Quantum-Tunnelling-Simulations
Computational Study of Quantum Tunnelling and Reflection

Filename: freewavepacket_animation.py
Function: Displays an animation of the time evolution for a free gaussian wave-packet. At the end of the script, total probability and peak value of the wave-packet
	  are plotted against time. The number of x-axis points 'N' , has been chosen to preserve simulation accuracy
          whilst providing a smooth animation, with little buffer time. The same applies for the final integration time, 'tf', and time step 'dt'.

Filename: wavepacket_single_potential_barrier.py
Function: -Displays an animation of a Gaussian wave-packet impinging on a potential barrier. The average wavenumber,'k_0',can be changed by changing variable 'k'. 
          The potential barrier height can be changed by changing the variable 'f'. The number of x-axis points 'N' , has been chosen to preserve simulation accuracy
          whilst providing a smooth animation, with little buffer time. The same applies for the final integration time, 'tf', and time step 'dt'.
	  - At the end of the animation, the program produces a figure of the total probability of finding the particle to the left of the barrier as a function of time.

Filename: single_potential_barrier_3d_plot.py
Function: - Displays a 3D time-evolution/ time-series plot of the wave-packet impinging on potential barrier at equally spaced time steps.
	  - Plots the wavefunction at the final integration time.
	  - The average wavenumber,'k_0',can be changed by changing variable 'k'. The potential barrier height can be changed by changing the variable 'f'.
	    The number of x-axis points 'N' , has been chosen to preserve simulation accuracy whilst mainting a speedy run-time.  
	    The  final integration time, 'tf', and time step 'dt', have been chosen to preserve clarity in the final 3D plot image. If you make 'dt' too small, or
	    'tf' too big, the plot looks more cluttered.
	  - The code is set up to provide an aesthetically pleasing plot.

Filename: TR_coefficients_single_barrier.py
Function: - Calculates the transmission and reflection probabilities and plots them against average wavenumber k_0. 
	  - This script is time consuming to run since, as supplied, it is cycling through 100 different k_0 values and each probability integration sums 5000 points 
	    on the wave-function. The script takes less time if you sample fewer k_0 values (i.e. make the k_0 array smaller) and decrease the number of x-axis points
	    (N). However, this comes at the expense of less accurate integrals for the probability, and the plots will be less smooth.

Filename: Convolution_of_TransmissionProbability.py
Function: - Plots the analytical curve for the transmission function of a plane wave.
	  - Applies a Gaussian filter on the curve and plots the convolved transmission coefficient, which reporduces the features of Gaussian wave-packet scattering.

Filename: wavepacket_double_potential_barrier.py
Function: -Displays an animation of a Gaussian wave-packet impinging on a double potential barrier. The average wavenumber,'k_0',can be changed by changing variable 'k'. 
          The potential barrier height can be changed by changing the variable 'f'. The number of x-axis points 'N' , has been chosen to preserve simulation accuracy
          whilst providing a smooth animation, with little buffer time. The same applies for the final integration time, 'tf', and time step 'dt'.
	  - At the end of the animation, the program produces a figure of the total probability of finding the particle to the left of the barrier as a function of time.


Filename: double_potential_barrier_3d_plot.py
Function: - Displays a 3D time-evolution/ time-series plot of the wave-packet impinging on double potential barrier, at equally spaced time steps.
	  - Plots the wavefunction at the final integration time.
	  - The average wavenumber,'k_0',can be changed by changing variable 'k'. The potential barrier height can be changed by changing the variable 'f'.
	    The number of x-axis points 'N' , has been chosen to preserve simulation accuracy whilst mainting a speedy run-time.  
	    The  final integration time, 'tf', and time step 'dt', have been chosen to preserve clarity in the final 3D plot image. If you make 'dt' too small, or
	    'tf' too big, the plot looks more cluttered.
	  - The code is set up to provide an aesthetically pleasing plot.

Filename: Probability_Decay_Double_Barrier_Gap.py
Function: - Produces a plot of the confinement probability against time for a wave-packet between two potential barriers. 
	  - Works out the polynomial fitting parameters for the decay for different values of average wavenumber k_0.
	  - Typically took 5 minutes on my laptop.

Filename: 2Dsimulation_datagenerator_Potential.py
Function: - Generates data for each frame of an animation for a Gaussian wave-packet scattering of a potential barrier in 2D
	  - This script is very time-consuming. The data files have been provided (Anim_Data_200frame_UBarrier_U=3.npy) , which allows the Animation script 
            (2Dsimulation_animator.py) to animate the simulation.
	    
Filename: 2Dsimulation_animator.py
Function: - Reads the data generated by the 2Dsimulation_datagenerator_Potential.py and stitches the frames together to play an animation. It also saves
	    animation as an mp4 file.

Filename: Anim_Data_200frame_UBarrier_U=3.npy
Function: Animation data, pre-baked, using 2Dsimulation_datagenerator_Potential.py. This is read by 2Dsimulation_animator.py to produce an animation and 
	  save as an mp4.

Filename: 2dwavepacket_200frame_UBarrier_U=3.mp4
Function: Here's an animation made earlier using the the previous 3 scripts. 



