from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np
import torch
from src.algorithms.adam import AdamOptimizer

def optimize_with_one_optimizer(
    optimizer_cls,
    x_init: np.ndarray,
    loss_fn: Callable,
    loss_grad: Optional[Callable] = None,  # only for AdamOptimizer
    optim_kwargs: Dict[str, Any] = None,
    max_iter: int = 100,
    tol_grad: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimize a function using a specified optimizer.

    Parameters:
        optimizer_cls (class): The optimizer class to be used.
        x_init (np.ndarray): The initial input for optimization.
        loss_fn (Callable): The loss function to be minimized.
        loss_grad (Optional[Callable]): The gradient of the loss function 
            (only for AdamOptimizer).
        optim_kwargs (Dict[str, Any]): Additional keyword arguments for the 
            optimizer.
        max_iter (int): Maximum number of iterations.
        tol_grad (float): Tolerance for the gradient to determine convergence.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the optimized input 
            and corresponding function values.
    """
    if optimizer_cls == AdamOptimizer:
        if loss_grad is None:
            raise ValueError("AdamOptimizer requires `loss_grad`.")

        adam_optimizer = AdamOptimizer(**optim_kwargs)
        adam_optimizer.minimize(x_init, f=loss_fn, f_grad=loss_grad, max_iter=max_iter)

        return adam_optimizer.all_x_k, adam_optimizer.all_f_k

    x = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
    optimizer = optimizer_cls([x], **optim_kwargs)
    all_x_k, all_f_k = [x_init], [loss_fn(x).item()]

    for _ in range(max_iter):
        optimizer.zero_grad()
        loss = loss_fn(x)
        loss.backward()

        with torch.no_grad():
            optimizer.step()
            x_k_np = x.detach().numpy().copy()
            all_x_k.append(x_k_np)
            all_f_k.append(loss.item())

            if np.min(np.abs(x.grad.detach().numpy().copy())) < tol_grad:
                break

    return np.array(all_x_k), np.array(all_f_k)
