import torch
import numpy as np

class vanderpol(torch.nn.Module):

    def __init__(self):
        
        super(vanderpol, self).__init__()
        self.mu = 2.
        self.state_variables = ['x', 'y']
    
    def forward(self, X, U):
        x, y = torch.split(X[:, 0], split_size_or_sections=1, dim=1)
        xdt = self.mu * (x - 1/3 * x**3 - y)
        ydt = 1/self.mu * x
        return torch.concat((xdt, ydt), dim=1)
            
class duffing_oscillator(torch.nn.Module):

    def __init__(self):

        super(duffing_oscillator, self).__init__()
        self.gamma = 0.002
        self.omega = 1
        self.beta = 1
        self.alpha = 1
        self.delta = 0.3
        self.state_variables = ['x', 'y']
     
    def forward(self, X, U):
        x0, x1  = torch.split(X[:, 0], split_size_or_sections=1, dim=1)
        f = U[:, 0]
        d_x0__dt = x1
        d_x1__dt = -self.gamma * x1 - self.alpha * x0 - self.beta * x0**3 + f
        return torch.concat((d_x0__dt, d_x1__dt), dim=1)
      