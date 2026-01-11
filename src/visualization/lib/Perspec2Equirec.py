import os
import sys
import cv2
import numpy as np

import cv2
import numpy as np

class Perspective:
    def __init__(self, img_name , FOV, THETA, PHI, ROLL=0.0):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if self._img is None:
            raise FileNotFoundError(img_name)
        self._height, self._width, _ = self._img.shape

        self.wFOV  = FOV          # horizontálny FOV (deg)
        self.THETA = THETA        # yaw (deg)
        self.PHI   = PHI          # pitch (deg)
        self.ROLL  = ROLL         # roll (deg)

        # odporúčaná presná formula pre vertikálny FOV
        self.hFOV = np.degrees(
            2*np.arctan((self._height/self._width)*np.tan(np.radians(self.wFOV/2)))
        )

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

    def GetEquirec(self, height, width):
        # sférická mriežka (lon: -180..180, lat: 90..-90)
        x, y = np.meshgrid(np.linspace(-180, 180, width),
                           np.linspace(90,  -90, height))
        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))
        xyz = np.stack((x_map, y_map, z_map), axis=2).reshape([-1, 3])

        # --- ROTÁCIE (intrinsic: yaw -> pitch -> roll) ---
        # Zachovám tvoje pôvodné znamienka: pitch mal v kóde mínus.
        z_axis = np.array([0., 0., 1.], np.float32)  # yaw
        y_axis = np.array([0., 1., 0.], np.float32)  # pitch
        x_axis = np.array([1., 0., 0.], np.float32)  # roll

        Rz, _ = cv2.Rodrigues(z_axis * np.radians(self.THETA))      # yaw
        Ry, _ = cv2.Rodrigues(y_axis * np.radians(-self.PHI))       # pitch (mínus kvôli pôvodnému kódu)
        Rx, _ = cv2.Rodrigues(x_axis * np.radians(-self.ROLL))       # roll

        # kombinácia a inverzia (potrebujeme world->camera)
        R = (Rz @ Ry @ Rx).T    # inverse of pure rotation = transpose

        # aplikuj rotáciu
        xyz = (R @ xyz.T).T.reshape([height, width, 3])

        # viditeľnosť + perspektívna normalizácia
        forward = xyz[:, :, 0] > 0
        xyz = xyz / np.repeat(xyz[:, :, 0][:, :, None], 3, axis=2)

        in_w = (-self.w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.w_len)
        in_h = (-self.h_len < xyz[:, :, 2]) & (xyz[:, :, 2] < self.h_len)
        cond = in_w & in_h

        lon_map = np.where(cond, (xyz[:, :, 1] + self.w_len) / (2*self.w_len) * self._width, 0)
        lat_map = np.where(cond, (-xyz[:, :, 2] + self.h_len) / (2*self.h_len) * self._height, 0)

        # remap
        persp = cv2.remap(self._img,
                          lon_map.astype(np.float32),
                          lat_map.astype(np.float32),
                          cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)

        # binárna maska 0/255 (žiadne dtype problémy)
        mask_u8 = np.where(cond & forward, 255, 0).astype(np.uint8)

        # aplikuj masku (ponechaj uint8)
        persp = cv2.bitwise_and(persp, persp, mask=mask_u8)

        # vráť aj 3-kanálovú masku, ak chceš (voliteľne)
        mask3 = cv2.merge([mask_u8, mask_u8, mask_u8])

        return persp, mask3

        
##################

# import cv2
# import numpy as np

# class Perspective:
#     def __init__(self, img_name, FOV, THETA, PHI, ROLL=0):
#         """
#         Args:
#             img_name (str): input image path
#             FOV (float): horizontal field of view in degrees
#             THETA (float): yaw (left/right) in degrees
#             PHI (float): pitch (up/down) in degrees
#             ROLL (float): roll (rotation around optical axis) in degrees
#         """
#         self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
#         if self._img is None:
#             raise FileNotFoundError(f"Image not found: {img_name}")
#         self._height, self._width, _ = self._img.shape

#         self.wFOV = FOV
#         self.THETA = THETA
#         self.PHI = PHI
#         self.ROLL = ROLL

#         self.hFOV = float(self._height) / self._width * FOV
#         self.w_len = np.tan(np.radians(self.wFOV / 2.0))
#         self.h_len = np.tan(np.radians(self.hFOV / 2.0))

#     def GetEquirec(self, height, width):
#         # Create spherical grid (lon: -180°→180°, lat: 90°→-90°)
#         x, y = np.meshgrid(np.linspace(-180, 180, width), np.linspace(90, -90, height))
        
#         x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
#         y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
#         z_map = np.sin(np.radians(y))
#         xyz = np.stack((x_map, y_map, z_map), axis=2)

#         # Axis definitions
#         x_axis = np.array([1.0, 0.0, 0.0], np.float32)
#         y_axis = np.array([0.0, 1.0, 0.0], np.float32)
#         z_axis = np.array([0.0, 0.0, 1.0], np.float32)

#         # Rotation matrices for yaw (THETA), pitch (PHI), roll (ROLL)
#         [R_yaw, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA))
#         [R_pitch, _] = cv2.Rodrigues(np.dot(R_yaw, y_axis) * np.radians(-self.PHI))
#         [R_roll, _] = cv2.Rodrigues(np.dot(R_pitch, x_axis) * np.radians(self.ROLL))

#         # Inverse to map from equirect → perspective
#         R_yaw = np.linalg.inv(R_yaw)
#         R_pitch = np.linalg.inv(R_pitch)
#         R_roll = np.linalg.inv(R_roll)

#         # Combine rotation (yaw → pitch → roll)
#         R = R_roll @ R_pitch @ R_yaw

#         xyz = xyz.reshape([-1, 3]).T
#         xyz = np.dot(R, xyz).T.reshape([height, width, 3])

#         # Keep only visible points
#         inverse_mask = (xyz[:, :, 0] > 0).astype(np.float32)
#         xyz /= np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

#         # Perspective projection
#         cond = (
#             (-self.w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.w_len) &
#             (-self.h_len < xyz[:, :, 2]) & (xyz[:, :, 2] < self.h_len)
#         )
#         lon_map = np.where(cond, (xyz[:, :, 1] + self.w_len) / (2 * self.w_len) * self._width, 0)
#         lat_map = np.where(cond, (-xyz[:, :, 2] + self.h_len) / (2 * self.h_len) * self._height, 0)

#         persp = cv2.remap(
#             self._img,
#             lon_map.astype(np.float32),
#             lat_map.astype(np.float32),
#             cv2.INTER_CUBIC,
#             borderMode=cv2.BORDER_WRAP
#         )

#         mask = cond.astype(np.float32) * inverse_mask
#         mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
#         persp = (persp.astype(np.float32) * mask).astype(np.uint8)

#         return persp, mask



