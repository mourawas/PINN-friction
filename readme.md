# Physics-Informed Neural Network for Friction Modeling & Parameter Discovery

A PyTorch implementation of a Physics-Informed Neural Network (PINN) for friction

## Overview

This PINN combines data-driven learning with the LuGre friction model physics to predict friction forces. The LuGre model, introduced by Canudas de Wit et al. (1995), captures complex friction phenomena including the Stribeck effect, and stick-slip behavior.

## LuGre Model

The implementation follows the LuGre friction model:
- **State equation**: `F = σ₀z + σ₁(dz/dt) + σ₂v`
- **State dynamics**: `dz/dt = v - |v|z/g(v)`  
- **Stribeck function**: `g(v) = (Fc + (Fs - Fc)exp(-(v/vs)²))/σ₀`

Where `z` is the internal friction state, `v` is velocity, and `F` is friction force.

## Requirements

```bash
pip install torch numpy matplotlib
```

## Usage

```python
# Initialize and train
pinn = Friction(time_data, velocity_data, friction_data)
loss_history, data_loss_history, physics_loss_history, param_history = pinn.train(epochs=10000)

# Predict
z_pred, F_pred, dzdt_pred, g_v_pred, position, masked_position, peak_idx = pinn.predict(time_test, velocity_test, friction_test)
```

## Run Complete Pipeline

```bash
python pinn.py
```

Loads M4 (training) and M6 (test) datasets, trains the model, and generates plots in `plots pinn/` directory.

## Model Features

- **Neural Network**: 3-layer network (64→128→128) with velocity, direction, and position inputs
- **Learnable Parameters**: Six LuGre parameters (σ₀, σ₁, σ₂, Fc, Fs, vs) with physical constraints
- **Combined Loss**: Weighted data fitting + physics residual loss
- **Regularization**: Layer normalization, dropout, and weight decay

## Reference

Canudas de Wit, C., Olsson, H., Astrom, K. J., & Lischinsky, P. (1995). A new model for control of systems with friction. *IEEE Transactions on automatic control*, 40(3), 419-425.
