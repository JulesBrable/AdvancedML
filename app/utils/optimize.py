from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np
import torch
from utils.adam import AdamOptimizer
from utils.config import get_optimizer, get_optimizer_choices

def optimize_with_one_optimizer(
    optimizer_cls_name,
    x_init: np.ndarray,
    loss_fn: Callable,
    loss_grad: Optional[Callable] = None,  # only for AdamOptimizer
    lr: float = 0.01,
    optim_kwargs: Dict[str, Any] = None,
    max_iter: int = 100,
    tol_grad: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, Any]:
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
    
    x = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
        
    if optimizer_cls_name == "Adam":
        optim_kwargs['lr'] = lr
        adam_optimizer = AdamOptimizer(**optim_kwargs)
        adam_optimizer.minimize(x, f=lambda x: loss_fn(x.clone().detach()), f_grad=lambda x: loss_grad(x.clone().detach()))
        return np.array(adam_optimizer.all_x_k), np.array(adam_optimizer.all_f_k), adam_optimizer
            

    optimizer = get_optimizer(optimizer_cls_name, [x], lr, **optim_kwargs)
    all_x_k, all_f_k = [x.detach().numpy()], [loss_fn(x).item()]
    
    for _ in range(max_iter):
        optimizer.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x_k_np = x.detach().numpy().copy()  
            all_x_k.append(x_k_np)
            all_f_k.append(loss_fn(x).item())
    return np.array(all_x_k)[1:], np.array(all_f_k), optimizer