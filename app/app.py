import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from utils.math_tools import (
    f1, f1_grad,
    f2, f2_grad,
    f3, f3_grad,
    f4, f4_grad,
    f5, f5_grad
)

def dynamic_plot_hyperparameters(beta1, beta2, num_iterations, function, ax1, ax2):
    f, f_grad = function_dict[function]
    params = torch.tensor([1.5, -1.5], requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=0.1, betas=(beta1, beta2))

    x = np.linspace(-4, 4, 600)
    y = np.linspace(-4, 4, 600)
    X, Y = np.meshgrid(x, y)
    Z = f(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)).detach().numpy()

    ax1.clear()
    ax2.clear()
    ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.6)
    ax2.contourf(X, Y, Z, levels=50, cmap=cm.coolwarm)

    trajectory = []
    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss = f(params[0], params[1])
        loss.backward()
        optimizer.step()
        trajectory.append(params.detach().numpy().copy())

    trajectory = np.array(trajectory)
    trajectory_tensor = torch.tensor(trajectory, dtype=torch.float32)
    z_values = f(trajectory_tensor[:, 0], trajectory_tensor[:, 1]).detach().numpy() 
    ax1.plot(trajectory[:, 0], trajectory[:, 1], z_values, color='black', marker='o')
    ax2.plot(trajectory[:, 0], trajectory[:, 1], color='black', marker='o')


st.title("Visualization of the Optimization by the Adam Algorithm")

beta1 = st.sidebar.slider(f"Select β₁", 0.0, 0.999, 0.9, 0.001)
beta2 = st.sidebar.slider(f"Select β₂", 0.0, 0.999, 0.999, 0.001)
iterations = st.sidebar.slider('Iterations', 20, 100, 50)
function_selected = st.sidebar.radio("Select a function", ['f1', 'f2', 'f3', 'f4', 'f5'])

function_dict = {
    'f1': (f1, f1_grad),
    'f2': (f2, f2_grad),
    'f3': (f3, f3_grad),
    'f4': (f4, f4_grad),
    'f5': (f5, f5_grad)
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

dynamic_plot_hyperparameters(beta1, beta2, iterations, function_selected, ax1, ax2)

st.pyplot(fig)