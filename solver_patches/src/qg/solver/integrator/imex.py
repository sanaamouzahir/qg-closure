import torch
import numpy as np
from qg.solver.util import _Cache, cached_dot

def CN2(u, source, dt, linear_operator):
    # u_n+1 - dt/2 f(u_n+1) = u_n + dt/2 f(u_n)
    rhs = u + dt * (0.5 * linear_operator * u + source) 
    lhs = (1 - 0.5 * dt * linear_operator)
    return rhs / lhs

class AB(_Cache):
    def __init__(self, level=2):
        super().__init__(level,
            cached_dot({
                1: [1],
                2: [3/2, -1/2],
                3: [23/12, -4/3, 5/12],
                4: [55/24, -59/24, 37/24, -3/8],
                5: [1901/720, -1387/360, 109/30, -637/360, 251/720]
            })
        )
        
AB2 = AB(2)