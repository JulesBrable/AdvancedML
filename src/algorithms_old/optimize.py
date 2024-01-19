"""
"""
from itertools import product
from typing import Callable, Dict, Any, Optional, Tuple, Literal

import numpy as np
import torch
from joblib import Parallel, delayed
from src.algorithms_old.adam import AdamOptimizer


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


def optimize_with_multiple_optimizers(
    x_init: np.ndarray,
    loss_fn: Callable,
    loss_grad: Callable,
    optimizers_config: Dict[str, Tuple[Any, Dict[str, Any]]],
    max_iter: int = 1000,
    tol_grad: float = 1e-6
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Optimize a given objective function using multiple optimizers.

    Parameters:
    - x_init (np.ndarray): Initial guess for the optimization.
    - loss_fn (Callable): Objective function to minimize.
    - loss_grad (Callable): Gradient of the objective function.
    - optimizers_config (Dict[str, Tuple[Any, Dict[str, Any]]]): Dictionary 
        where keys are optimizer names and values are tuples containing 
        optimizer class and its configuration.
    - max_iter (int): Maximum number of optimization iterations 
        (default: 1000).
    - tol_grad (float): Tolerance for convergence based on the gradient norm
        (default: 1e-6).

    Returns:
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Tuple containing 
        dictionaries of solutions and corresponding objective values for each 
        optimizer.
    """
    solutions, values = {}, {}

    for optim_name, (optim_cls, optim_kwargs) in optimizers_config.items():
        if optim_cls == AdamOptimizer:
            adam_optim = optim_cls(**optim_kwargs, tol_grad=tol_grad)
            adam_optim.minimize(
                theta_init=x_init,
                f=loss_fn,
                f_grad=loss_grad,
                f_grad_args=(),
                max_iter=max_iter
            )
            solutions[optim_name] = np.array(adam_optim.all_x_k)
            values[optim_name] = np.array(adam_optim.all_f_k)
        else:
            s, v = optimize_with_one_optimizer(
                optim_cls, x_init, loss_fn, loss_grad, optim_kwargs, max_iter, tol_grad
            )
            solutions[optim_name] = s
            values[optim_name] = v

    return solutions, values


def tune_parameters(
    param_grid: Dict[str, list],
    optimizer_cls,
    x_init: np.ndarray,
    loss_fn: Callable,
    loss_grad: Optional[Callable] = None,
    max_iter: int = 1000,
    tol_grad: float = 1e-6,
    criteria: Literal["n_iter", "x_distance"] = "n_iter",
    x_star: Optional[np.ndarray] = None
) -> Dict[str, Any]:

    best_score = float('inf')
    best_params = None

    if criteria == "n_iter":
        for params in product(*param_grid.values()):
            optimizer_params = dict(zip(param_grid.keys(), params))
            result = optimize_with_one_optimizer(
                optimizer_cls=optimizer_cls,
                x_init=x_init,
                loss_fn=loss_fn,
                loss_grad=loss_grad,
                optim_kwargs=optimizer_params,
                max_iter=max_iter,
                tol_grad=tol_grad
            )
            current_n_iter = len(result[0])

            if current_n_iter < best_score:
                best_score = current_n_iter
                best_params = optimizer_params

    elif criteria == "x_distance":
        for params in product(*param_grid.values()):
            optimizer_params = dict(zip(param_grid.keys(), params))
            result = optimize_with_one_optimizer(
                optimizer_cls=optimizer_cls,
                x_init=x_init,
                loss_fn=loss_fn,
                loss_grad=loss_grad,
                optim_kwargs=optimizer_params,
                max_iter=max_iter,
                tol_grad=tol_grad
            )
            current_min = np.linalg.norm(result[0][-1] - x_star)

            if current_min < best_score:
                best_score = current_min
                best_params = optimizer_params

    return {"optimal_grid": best_params, "best_n_iter": best_score}


def tune_parameters_multiple(
    optimizers: Dict[str, type],
    param_grids: Dict[str, Dict[str, list]],
    x_init: np.ndarray,
    loss_fn: Callable,
    loss_grad: Optional[Callable] = None,
    max_iter: int = 1000,
    tol_grad: float = 1e-6,
    n_jobs: int = -1
) -> Dict[str, Dict[str, Any]]:
    """
    Tune parameters for multiple optimizers using grid search and return optimal sets.

    Parameters:
        optimizers (Dict[str, type]): Dictionary containing optimizer names as keys
            and their respective optimizer classes as values.
        param_grids (Dict[str, Dict[str, list]]): Dictionary containing optimizer names
            as keys and their respective parameter grids as values.
        x_init (np.ndarray): The initial input for optimization.
        loss_fn (Callable): The loss function to be minimized.
        loss_grad (Optional[Callable]): The gradient of the loss function.
        max_iter (int): Maximum number of iterations for each optimization run.
        tol_grad (float): Tolerance for the gradient to determine convergence.
        n_jobs (int): Number of parallel jobs to run (-1 for using all available CPUs).

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing optimal grids for each optimizer.
    """
    def tune_parameters_single(optimizer_name, optimizer_cls, param_grid):
        result = tune_parameters(
            param_grid=param_grid,
            optimizer_cls=optimizer_cls,
            x_init=x_init,
            loss_fn=loss_fn,
            loss_grad=loss_grad,
            max_iter=max_iter,
            tol_grad=tol_grad
        )
        return {optimizer_name: result}

    results = Parallel(n_jobs=n_jobs)(
        delayed(tune_parameters_single)(name, cls, param_grids[name]) 
        for name, cls in optimizers.items()
    )

    optimal_grids = {}
    for result in results:
        optimal_grids.update(result)

    return optimal_grids

def build_optimizers_config(optimal_grids, optimizer_mapping):
    optimizers_config = {}

    for optim_name, params in optimal_grids.items():
        optimizer_cls = optimizer_mapping.get(optim_name.lower())
        if optimizer_cls is not None:
            optimizer_kwargs = params['optimal_grid']
            optimizers_config[optim_name] = (optimizer_cls, optimizer_kwargs)

    return optimizers_config
