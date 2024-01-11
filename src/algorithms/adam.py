from typing import Optional, List, Callable, Tuple, Dict, Any
import numpy as np

ArrayLike = np.ndarray

class AdamOptimizer():
    """
    Adam optimizer implementation for numerical optimization.

    Parameters:
    - learning_rate (float): The learning rate for the optimization algorithm.
    - beta1 (float): Exponential decay rate for the first moment estimates.
    - beta2 (float): Exponential decay rate for the second moment estimates.
    - epsilon (float): Small constant to prevent division by zero.
    - use_ema (bool): Flag indicating whether to use exponential moving 
        average (EMA).

    Methods:
    - initialize(var): Initialize momentums and velocities for optimization.
    - update_step(x, f, f_grad, f_grad_args=()): Perform a single update step
        of the optimization.
    - minimize(x_init, f, f_grad, f_grad_args=(), max_iter=1000): Minimize the
        objective function.
    - get_config(): Get the configuration parameters of the optimizer.
    """

    def __init__(self,
        learning_rate: float=0.001,
        beta1: float=0.9,
        beta2: float=0.999,
        epsilon: float=1e-8,
        use_ema: bool=False
    ) -> None:
        """
        Initialize the Adam optimizer with specified parameters.

        Parameters:
        - learning_rate (float): The learning rate for the optimization algorithm.
        - beta1 (float): Exponential decay rate for the first moment estimates.
        - beta2 (float): Exponential decay rate for the second moment estimates.
        - epsilon (float): Small constant to prevent division by zero.
        - use_ema (bool): Flag indicating whether to use exponential moving 
            average (EMA).
        """
        self.learning_rate = learning_rate
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
            x: ArrayLike,
            f: Callable,
            f_grad: Callable,
            f_grad_args: Tuple = ()
    ) -> Tuple[ArrayLike, float, ArrayLike]:
        """
        Perform a single update step of the optimization.

        Parameters:
        - x (ArrayLike): Current values of the optimization variables.
        - f (Callable): Objective function to be minimized.
        - f_grad (Callable): Gradient of the objective function.
        - f_grad_args (Tuple): Additional arguments for the gradient function.

        Returns:
        Tuple[ArrayLike, float, ArrayLike]: Updated values of the variables, objective function value, and gradient.
        """
        x_old = x.copy()
        timestep = self.iter
        self.iter += 1
        alpha = self.learning_rate

        # Update rule.
        grad = f_grad(x, *f_grad_args)
        self.momentums = self.beta1 * self.momentums + (1 - self.beta1) * grad
        self.velocities = self.beta2 * self.velocities + (1 - self.beta2) * grad**2
        alpha *= np.sqrt(1 - self.beta2**timestep) / (1 - self.beta1**timestep)
        x -= alpha * self.momentums / (np.sqrt(self.velocities) + self.epsilon)

        # Exponential moving average (EMA).
        if self.use_ema:
            x = self.beta2 * x_old + (1 - self.beta2) * x

        # Initialization bias correction.
        x /= (1 - self.beta2**timestep)

        return x, f(x), grad

    def minimize(
            self,
            x_init: ArrayLike,
            f: Callable,
            f_grad: Callable,
            f_grad_args: Tuple = (),
            max_iter: int = 1000
    ) -> ArrayLike:
        """
        Minimize the objective function.

        Parameters:
        - x_init (ArrayLike): Initial values of the optimization variables.
        - f (Callable): Objective function to be minimized.
        - f_grad (Callable): Gradient of the objective function.
        - f_grad_args (Tuple): Additional arguments for the gradient function.
        - max_iter (int): Maximum number of iterations.

        Returns:
        ArrayLike: Optimized values of the variables.
        """
        x = x_init.copy()

        if self.iter == 0:
            self.initialize(x)

        for _ in range(max_iter):
            x, f_x, grad_x = self.update_step(
                x,
                f=f,
                f_grad=f_grad,
                f_grad_args=f_grad_args
            )
            self.all_x_k.append(x)
            self.all_f_k.append(f_x)

            # Alternative stopping criteria.
            l_inf_norm_grad = np.max(np.abs(grad_x))
            if l_inf_norm_grad < 1e-6:
                break

        return x

    def get_config(self) -> Dict[str, Any]:
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
