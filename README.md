# Physics-informed neural network (PINNs) for inverse problems

Implementation of PINNs (Raissi et al., J. Comput. Phys., 2019) for solving the inverse problem. All examples are implemented using **PyTorch**.

## Van der Pol oscillator

We implement PINNs for the van der Pol oscillator -- a highly nonlinear second order ODE that gives rise to relaxation oscillations (multiple time scales). To deal with the nonlinearities in this model, and in contrast to the original PINNs implementation by Raissi et al., we here propose to train the network by using dynamic weighting and sequential learning for the physics loss function. These extensions not only stabilize PINNs, but also allow for precise inference of model parameters beyond the data range.

