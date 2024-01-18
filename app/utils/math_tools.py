
import torch


def polynomial(x, y): 
    return x**4 - x**3 + y**2

def polynomial_grad(x, y):
    df_dx = 4*x**3 - 3*x**2
    df_dy = 2*y
    return torch.stack([df_dx, df_dy])


def trigonometric(x, y):
    return torch.sin(x) + torch.cos(y)

def trigonometric_grad(x, y):
    df_dx = torch.cos(x)
    df_dy = -torch.sin(y)
    return torch.stack([df_dx, df_dy])

# ====================================================================
# Non-convex Gaussian kernel functions
# ====================================================================
def gaussian(x, y):
    """
    Gaussian kernel function.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output of the Gaussian kernel function.
    """
    return 1 - torch.exp(-(x**2 + y**2))

def gaussian_grad(x, y):
    df_dx = 2 * x * torch.exp(-(x**2 + y**2))
    df_dy = 2 * y * torch.exp(-(x**2 + y**2))
    return torch.stack([df_dx, df_dy])
    
def logarithmic(x, y):
    return -torch.log(x**2 + y**2 + 1)

def logarithmic_grad(x, y):
    df_dx = -2*x/(x**2 + y**2 + 1)
    df_dy = -2*y/(x**2 + y**2 + 1)
    return torch.stack([df_dx, df_dy])

def mixed(x, y):
    return x**2 - torch.sin(y)**2

def mixed_grad(x, y):
    df_dx = 2*x
    df_dy = -2*torch.sin(y)*torch.cos(y)
    return torch.stack([df_dx, df_dy])

# ====================================================================
# Ill-condition problem: Rosenbrock function (flat region)
# ====================================================================

def rosenbrock(x, y):
    a = 1
    b = 100
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y):
    a = 1
    b = 100
    df_dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    df_dy = 2 * b * (y - x**2)
    return torch.stack([df_dx, df_dy])


def quadratic(epsilon):
    def f(x, y):
        scaled_x = x * epsilon
        scaled_y = y * epsilon**2
        return 0.5 * (scaled_x**2 + scaled_y**2)

    def f_prime(x, y):
        df_dx = epsilon**2 * x
        df_dy = epsilon**4 * y
        return torch.stack([df_dx, df_dy])

    return f, f_prime


def ackley(x, y):
    a = 20
    b = 0.2
    c = 2 * torch.pi
    sum1 = x**2 + y**2
    sum2 = torch.cos(c * x) + torch.cos(c * y)
    return -a * torch.exp(-b * torch.sqrt(sum1 / 2)) - torch.exp(sum2 / 2) + a + torch.exp(torch.tensor(1.0))

def ackley_grad(x, y):
    a = 20
    b = 0.2
    c = 2 * torch.pi
    exp_term = torch.exp(-b * torch.sqrt((x**2 + y**2) / 2))
    trig_term = torch.exp((torch.cos(c * x) + torch.cos(c * y)) / 2)
    
    df_dx = a * b * exp_term * x / torch.sqrt(2*(x**2 + y**2)) \
            + c/2 * trig_term * torch.sin(c * x)
            
    df_dy = a * b * exp_term * y / torch.sqrt(2*(x**2 + y**2)) \
            + c/2 * trig_term * torch.sin(c * y)

    return torch.stack([df_dx, df_dy])