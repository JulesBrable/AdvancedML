import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_convergence_path(solutions, loss_fn, x_star, box_scale=1.5):
    # Convert solutions to torch tensors
    solutions = [torch.tensor(sol, dtype=torch.float32) for sol in solutions]

    # Evaluate loss for each solution
    losses = [loss_fn(sol) for sol in solutions]

    # Convert solutions and losses to numpy arrays for plotting
    solutions_np = np.array([sol.numpy() for sol in solutions])
    losses_np = np.array(losses)

    # Calculate box boundaries
    x_min, y_min = solutions_np.min(axis=0) - box_scale
    x_max, y_max = solutions_np.max(axis=0) + box_scale

    # Plot contour levels within the box
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = loss_fn(torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32))

    plt.contour(X, Y, Z, levels=20, cmap='viridis')

    # Plot convergence path
    plt.scatter(solutions_np[:, 0], solutions_np[:, 1], c=losses_np, cmap='Reds', marker='x')
    plt.plot(solutions_np[:, 0], solutions_np[:, 1], linestyle='-', color='blue', linewidth=2)

    # Highlight the final solution
    plt.scatter(x_star[0], x_star[1], c='green', marker='o', label='Target Solution')

    # Set plot labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Convergence Path')

    plt.legend()
    plt.show()