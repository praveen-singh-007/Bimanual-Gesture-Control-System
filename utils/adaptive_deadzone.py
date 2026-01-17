import numpy as np

def adaptive_deadzone(v, speed, base=0.4, min_dz=0.12):
    dz = max(min_dz, base - speed * 0.08)
    if abs(v) <= dz:
        return 0.0
    return np.sign(v) * (abs(v) - dz)