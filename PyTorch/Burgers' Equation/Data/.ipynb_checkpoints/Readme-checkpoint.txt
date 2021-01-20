The Burgers PDE solver has been created using the Chebfun GUI 

Run the Chebfun_Burgers_PDE.m file in the Chebfun library using MATLAB or add the library to path

To my best knowledge, currently no python implementation of Chebfun can solve PDEs and therefore MATLAB is a prerequisite. Here, there are 3 solution files available with different parameters 


1. burgers_shock_mu_01_pi.mat == burgers_shock.mat
- mu = 0.01/pi, IC: -sin(pi*x), BC: u(−1, t) = u(1, t) = 0

2. burgers_shock_mu_005_pi.mat
- mu = 0.005/pi, IC = -sin(pi*x), BC: u(−1, t) = u(1, t) = 0

3. burgers_shock_IC_sin2pi.mat
- mu = 0.01/pi, IC = -sin(2*pi*x), BC: u(−1, t) = u(1, t) = 0
