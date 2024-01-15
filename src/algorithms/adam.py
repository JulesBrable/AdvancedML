"""
Adam Optimizer Module

This module contains the implementation of the Adam optimizer for numerical optimization.

The Adam optimizer is an adaptive learning rate optimization algorithm that combines
ideas from RMSprop and Momentum. It is particularly well-suited for optimizing 
non-convex objective functions.

Classes:
- AdamOptimizer: Implementation of the Adam optimizer.

Usage:
Instantiate an AdamOptimizer object with desired parameters, initialize it with the 
`initialize` method, and then use the `minimize` method to optimize an objective function.

Example:
>>> from algorithms.adam import AdamOptimizer

>>> # Create an instance of AdamOptimizer
>>> adam_opt = AdamOptimizer(
>>> ... learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_ema=False)

>>> # Define an objective function and its gradient
>>> def objective_function(theta):
>>> ... return theta[0]**2 + theta[1]**2

>>> def gradient_function(theta):
>>> ...  return [2 * theta[0], 2 * theta[1]]

>>> # Initialize optimizer with initial guess
>>> initial_guess = np.array([1.0, 1.0])
>>> adam_opt.initialize(initial_guess)

>>> # Minimize the objective function
>>> optimized_theta, min_value, min_gradient = adam_opt.minimize(
>>> ...     initial_guess, objective_function, gradient_function)

>>> print(f"Optimized Theta: {optimized_theta}")
>>> print(f"Minimum Value: {min_value}")
>>> print(f"Gradient at Minimum: {min_gradient}")
"""
from typing import Optional, List, Callable, Tuple
import numpy as np

ArrayLike = np.ndarray

class AdamOptimizer():
    """
    Adam optimizer implementation for numerical optimization.

    Parameters:
    - lr (float): The learning rate for the optimization algorithm.
    - beta1 (float): Exponential decay rate for the first moment estimates.
    - beta2 (float): Exponential decay rate for the second moment estimates.
    - epsilon (float): Small constant to prevent division by zero.
    - use_ema (bool): Flag indicating whether to use exponential moving 
        average (EMA).

    Methods:
    - initialize(var): Initialize momentums and velocities for optimization.
    - update_step(theta, f, f_grad, f_grad_args=()): Perform a single update step
        of the optimization.
    - minimize(theta_init, f, f_grad, f_grad_args=(), max_iter=1000): Minimize the
        objective function.
    - get_config(): Get the configuration parameters of the optimizer.
    """

    def __init__(self,
        lr: float=0.001,
        beta1: float=0.9,
        beta2: float=0.999,
        epsilon: float=1e-8,
        use_ema: bool=False
    ) -> None:
        """
        Initialize the Adam optimizer with specified parameters.

        Parameters:
        - lr (float): The learning rate for the optimization algorithm.
        - beta1 (float): Exponential decay rate for the first moment estimates.
        - beta2 (float): Exponential decay rate for the second moment estimates.
        - epsilon (float): Small constant to prevent division by zero.
        - use_ema (bool): Flag indicating whether to use exponential moving 
            average (EMA).
        """
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.use_ema = use_ema
        self.momentums: Optional[ArrayLike] = None
        self.velocities: Optional[ArrayLike] = None
        self.all_x_k: Optional[List[ArrayLike]] = None
        self.all_f_k: Optional[List[float]] = None
        self.iter = 0

    def initialize(self, var: ArrayLike) -> None:
        """
        Initialize momentums and velocities for optimization.

        Parameters:
        - var (ArrayLike): The array of parameters of model variables to be 
            updated.
        """
        self.momentums = np.zeros_like(var)
        self.velocities = np.zeros_like(var)
        self.all_x_k = []
        self.all_f_k = []
        self.iter = 0
        self.initial_guess = var
        
    def update_step(
            self,
            theta: ArrayLike,
            f: Callable,
            f_grad: Callable,
            f_grad_args: Tuple = ()
    ) -> Tuple[ArrayLike, float, ArrayLike]:
        """
        Perform a single update step of the optimization.

        Parameters:
        - theta (ArrayLike): Current values of the optimization variables.
        - f (Callable): Objective function to be minimized.
        - f_grad (Callable): Gradient of the objective function.
        - f_grad_args (Tuple): Additional arguments for the gradient function.

        Returns:
        Tuple[ArrayLike, float, ArrayLike]: Updated values of the variables, 
            objective function value, and gradient.
        """
        theta_old = theta.copy()
        self.iter += 1
        beta1_power = np.power(self.beta1, self.iter, dtype=theta.dtype)
        beta2_power = np.power(self.beta2, self.iter, dtype=theta.dtype)

        # Update rule.
        grad = f_grad(theta, *f_grad_args)
        self.momentums = self.beta1 * self.momentums + (1 - self.beta1) * grad
        self.velocities = self.beta2 * self.velocities + (1 - self.beta2) * grad**2
        alpha = self.learning_rate * np.sqrt(1 - beta2_power) / (1 - beta1_power)
        theta -= alpha * self.momentums / (np.sqrt(self.velocities) + self.epsilon)

        # Exponential moving average (EMA).
        if self.use_ema:
            theta = self.beta1 * theta_old + (1 - self.beta1) * theta

        return theta, f(theta), grad

    def minimize(
            self,
            theta_init: ArrayLike,
            f: Callable,
            f_grad: Callable,
            f_grad_args: Tuple = (),
            max_iter: int = 1000
    ) -> ArrayLike:
        """
        Minimize the objective function.

        Parameters:
        - theta_init (ArrayLike): Initial values of the optimization variables.
        - f (Callable): Objective function to be minimized.
        - f_grad (Callable): Gradient of the objective function.
        - f_grad_args (Tuple): Additional arguments for the gradient function.
        - max_iter (int): Maximum number of iterations.

        Returns:
        ArrayLike: Optimized values of the variables.
        """
        theta = theta_init.copy()

        if self.iter == 0:
            self.initialize(theta)

        self.all_x_k.append(theta.copy())
        self.all_f_k.append(f(theta))

        for _ in range(max_iter):
            theta, f_theta, grad_theta = self.update_step(
                theta, 
                f=f,
                f_grad=f_grad,
                f_grad_args=f_grad_args
            )
            self.all_x_k.append(theta.copy())
            self.all_f_k.append(f_theta.copy())

            # Alternative stopping criteria.
            l_inf_norm_grad = np.max(np.abs(grad_theta))
            if l_inf_norm_grad < 1e-6:
                break

        return theta, f(theta), f_grad(theta)

    def get_config(self) -> dict:
        """
        Get the configuration parameters of the optimizer.

        Returns:
        dict: Configuration parameters of the optimizer.
        """
        return {
            "learning_rate": self.learning_rate,
            "beta_1": self.beta1,
            "beta_2": self.beta2,
            "epsilon": self.epsilon,
            "use_ema": self.use_ema
        }
