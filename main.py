import cv2
import numpy as np
from svm import SingleViewM
import os

lines = {
    "x": [[(205, 280), (315, 220)], [(205, 205), (318, 150)], [(87, 121), (197, 74)]],
    "y": [[(205, 280), (90, 190)], [(205, 205), (87, 121)], [(318, 150), (197, 74)]],
    "z": [[(205, 280), (205, 205)], [(90, 190), (86, 121)], [(315, 220), (320, 150)]]
}
refpoints = {
    'x': np.array([315, 220, 1]).reshape(-1, 1),
    'y': np.array([90, 190, 1]).reshape(-1, 1),
    'z': np.array([205, 205, 1]).reshape(-1, 1),
    'o': np.array([205, 280, 1])
}
img = cv2.imread('img/sony_clie-800x600.png')
singlevm = SingleViewM(img, endpoints=lines, refpoints=refpoints)

proj_matrix = singlevm.findProjectionMatrix()
print(proj_matrix)
H_xy = proj_matrix[:, [0, 1, 3]].copy()
H_yz = -proj_matrix[:, [1, 2, 3]].copy()
H_xz = -proj_matrix[:, [0, 2, 3]].copy()

if not os.path.isdir('textures'):
    os.mkdir('textures')

frame_xy = cv2.warpPerspective(img, H_xy, img.shape[:2], flags=cv2.WARP_INVERSE_MAP)
cv2.imwrite('textures/xy.jpg', frame_xy[..., ::-1])

frame_xz = cv2.warpPerspective(img, H_xz, img.shape[:2], flags=cv2.WARP_INVERSE_MAP)
cv2.imwrite('textures/xz.jpg', frame_xz[..., ::-1])

frame_yz = cv2.warpPerspective(img, H_yz, img.shape[:2], flags=cv2.WARP_INVERSE_MAP)
cv2.imwrite('textures/yz.jpg', frame_yz[..., ::-1])
