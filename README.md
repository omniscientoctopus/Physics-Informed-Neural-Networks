# Physics-Informed-Neural-Networks (PINNs)

PINNs were proposed by Raissi et al. in [1] to solve PDEs by incorporating the physics (i.e the PDE) and the boundary conditions in the loss function. The loss is the Mean-Squared Error of the PDE and boundary residual measured on 'collocation points' distributed across the domain. 

This repository contains implementation of PINNs in TensorFlow 2 (w and w/o Keras) for the Burgers' and Poissons' PDE. 

# Work Summary

1. Solving stiff PDEs with the L-BFGS optimizer

PINNs are studied with the L-BFGS optimizer and compared with the Adam optimizer to observe the gradient imbalance reported in [2]  for stiff PDEs. It was observed that the gradient imbalance is not as stark with the L-BFGS optimizer when solving stiff PDEs. However, the convergence of PINNs is still slow due to the ill-conditioning of the optimization landscape. 

2. Bottom Up learning in PINNs

It was reported in [3] that PINNs tend to learn all spectral frequencies of the solution simalteneously due to the presence of derivatives in the loss function. In order to understand if there are any other changes in the learning mechanics of PINNs, bottom-up learning was reinvestigated. Bottom-up learing implies that the lower layers, i.e layers close to the input, converge faster than the upper layers, i.e layers closer to the output. A heuristic proof of bottom-up learning was given in [4], the same methodology is followed here while training the PINN to solve the Burgers' PDE.  No changes in this mechanism was observed and it was confirmed that PINNs also learn bottom-up. A video of this observation can be found here (https://youtu.be/LmaSPoBVOrA). 

3. Transfer Learning in PINNs

The effect of transfer learning in PINNs was studied to understand its effects on solution error. The general observation was that transfer learning helps us find better local minima when compared to a random Xavier initialization. 

[1] Maziar Raissi, Paris Perdikaris, and George Em Karniadakis.Physics InformedDeep Learning (Part I): Data-driven Solutions of Nonlinear Partial     DifferentialEquations.url:http://arxiv.org/pdf/1711.10561v1

[2] Sifan Wang, Yujun Teng, Paris Perdikaris.UNDERSTANDING AND MITIGAT-ING GRADIENT PATHOLOGIES IN PHYSICS-INFORMED NEURAL NET-WORKS. 2020.url:https://arxiv.org/pdf/2001.04536.pdf.

[3] Lu Lu et al.DeepXDE: A deep learning library for solving differential equations.2019.url:https://arxiv.org/abs/1907.04502

[4] Maithra Raghu et al.SVCCA: Singular Vector Canonical Correlation Analysis forDeep Learning Dynamics and Interpretability.url:http://arxiv.org/pdf/1706.05806v2
