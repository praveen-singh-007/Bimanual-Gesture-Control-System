import math
import numpy as np

def mcp_angle(a,b,c):

    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return None
    cos_angle = np.dot(ba,bc)/denom

    cos_angle = np.clip(cos_angle,-1.0,1.0)

    return np.degrees(np.arccos(cos_angle))