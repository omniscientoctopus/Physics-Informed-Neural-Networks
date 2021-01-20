# Physics-Informed-Neural-Networks (PINNs)

PINNs were proposed by Raissi et al. in [1] to solve PDEs by incorporating the physics (i.e the PDE) and the boundary conditions in the loss function. The loss is the Mean-Squared Error of the PDE and boundary residual measured on 'collocation points' distributed across the domain. 

PINNs are summarised in the following schematic:

<p align="center">
<img src="https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/blob/main/Images/PINN_schematic.png" width="600">
</p>

This repository currently contains implementation of PINNs in TensorFlow 2 and PyTorch for the Burgers' and Helmholtz PDE.

Currently working to incorporate SIREN (paper from NeurIPS 2020).

# Installation

### TensorFlow 

```javascript
pip install numpy==1.19.2 scipy==1.5.3 tensorflow==2.0.0 matplotlib==3.3.2 pydoe==0.3.8 seaborn==0.9.0
```

### PyTorch 

```javascript
pip install numpy==1.19.2 scipy==1.5.3 matplotlib==3.3.2 pydoe==0.3.8 torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
For GPU installations, check for compatible PyTorch versions on the [official website](https://pytorch.org/get-started/locally/).   

**NOTE**: Newer versions of seaborn do not support sns.distplot and can problematic when ploting gradient histograms

# Work Summary

1. Solving stiff PDEs with the L-BFGS optimizer

PINNs are studied with the L-BFGS optimizer and compared with the Adam optimizer to observe the gradient imbalance reported in [2]  for stiff PDEs. It was observed that the gradient imbalance is not as stark with the L-BFGS optimizer when solving stiff PDEs. However, the convergence of PINNs is still slow due to the ill-conditioning of the optimization landscape. 

2. Bottom Up learning in PINNs

It was reported in [3] that PINNs tend to learn all spectral frequencies of the solution simalteneously due to the presence of derivatives in the loss function. In order to understand if there are any other changes in the learning mechanics of PINNs, bottom-up learning was reinvestigated. Bottom-up learing implies that the lower layers, i.e layers close to the input, converge faster than the upper layers, i.e layers closer to the output. A heuristic proof of bottom-up learning was given in [4], the same methodology is followed here while training the PINN to solve the Burgers' PDE.  No changes in this mechanism was observed and it was confirmed that PINNs also learn bottom-up. A video of this observation can be found here (https://youtu.be/LmaSPoBVOrA). 

3. Transfer Learning in PINNs

The effect of transfer learning in PINNs was studied to understand its effects on solution error. The general observation was that transfer learning helps us find better local minima when compared to a random Xavier initialization. 

# Bibliography

[1] Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations. http://arxiv.org/pdf/1711.10561v1

[2] Sifan Wang, Yujun Teng, Paris Perdikaris. UNDERSTANDING AND MITIGATING GRADIENT PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS. 2020. https://arxiv.org/pdf/2001.04536.pdf.

[3] Lu Lu et al. DeepXDE: A deep learning library for solving differential equations.2019. https://arxiv.org/abs/1907.04502

[4] Maithra Raghu et al. SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. http://arxiv.org/pdf/1706.05806v2

[5] Levi McClenny and Ulisses Braga-Neto. Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism. 2020. https://arxiv.org/abs/2009.04544
