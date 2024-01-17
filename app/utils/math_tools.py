
import torch

def f1(x, y): 
    return x**4 - x**3 + y**2

def f1_grad(x, y):
    df_dx = 4*x**3 - 3*x**2
    df_dy = 2*y
    return torch.stack([df_dx, df_dy])

def f2(x, y):
    return torch.sin(x) + torch.cos(y)

def f2_grad(x, y):
    df_dx = torch.cos(x)
    df_dy = -torch.sin(y)
    return torch.stack([df_dx, df_dy])
    
def f4(x, y):
    return -torch.log(x**2 + y**2 + 1)

def f4_grad(x, y):
    df_dx = -2*x/(x**2 + y**2 + 1)
    df_dy = -2*y/(x**2 + y**2 + 1)
    return torch.stack([df_dx, df_dy])

def f5(x, y):
    return x**2 - torch.sin(y)**2

def f5_grad(x, y):
    df_dx = 2*x
    df_dy = -2*torch.sin(y)*torch.cos(y)
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
    return torch.exp(-(x**2 + y**2))

def gaussian_grad(x, y):
    df_dx = -2 * x * torch.exp(-x**2)
    df_dy = -2 * y * torch.exp(-y**2)
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
    common_factor = -a * torch.exp(-b * torch.sqrt((x**2 + y**2) / 2)) / 2
    df_dx = common_factor * x / torch.sqrt((x**2 + y**2) / 2) +  torch.exp((torch.cos(c * x) + torch.cos(c * y)) / 2) * torch.sin(c * x) / 2
    df_dy = common_factor * y / torch.sqrt((x**2 + y**2) / 2) + torch.exp((torch.cos(c * x) + torch.cos(c * y)) / 2) * torch.sin(c * y) / 2
    return torch.stack([df_dx, df_dy])
