
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

def f3(x, y):
    return torch.exp(-x**2 - y**2)

def f3_grad(x, y):
    df_dx = -2*x*torch.exp(-x**2 - y**2)
    df_dy = -2*y*torch.exp(-x**2 - y**2)
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
