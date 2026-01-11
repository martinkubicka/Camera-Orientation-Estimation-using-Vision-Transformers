import numpy as np

# w,h is cutout 
# W, H original panorama
# top,left returns
def persp_to_equirect_deg(px, py, w, h, fovx_deg, fovy_deg, yaw_deg, pitch_deg, W, H):
    fovx = np.deg2rad(fovx_deg)
    fovy = np.deg2rad(fovy_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    nx = ((px + 0.5) / w - 0.5) * 2 * np.tan(fovx / 2)
    ny = (0.5 - (py + 0.5) / h) * 2 * np.tan(fovy / 2)
    v = np.array([nx, ny, 1.0])
    v /= np.linalg.norm(v)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    R_yaw = np.array([[ cy, 0, sy],
                        [  0, 1,  0],
                        [-sy, 0, cy]])
    R_pitch = np.array([[1, 0,  0],
                        [0, cp, -sp],
                        [0, sp,  cp]])

    v_world = R_pitch @ R_yaw @ v
    
    lon = np.arctan2(v_world[0], v_world[2])
    lat = np.arcsin (v_world[1])

    u_eq = (lon + np.pi) / (2 * np.pi) * W
    v_eq = (np.pi / 2 - lat) / np.pi * H
    return [u_eq, v_eq]
