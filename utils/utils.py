import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, List



def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def erode_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.erode(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def show_mask(ax, mask: np.ndarray, random_color=False):
    mask = mask.astype(np.uint8)
    if np.max(mask) == 255:
        mask = mask / 255
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)


def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    color_table = {0: 'red', 1: 'green'}
    for label_value, color in color_table.items():
        points = coords[labels == label_value]
        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
                   s=size, edgecolor='white', linewidth=1.25)

# def get_clicked_point(img_path):
#     img = cv2.imread(img_path)
#     cv2.namedWindow("image")
#     cv2.imshow("image", img)
#
#     last_point = []
#     keep_looping = True
#
#     def mouse_callback(event, x, y, flags, param):
#         nonlocal last_point, keep_looping, img
#
#         if event == cv2.EVENT_LBUTTONDOWN:
#             if last_point:
#                 cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
#             last_point = [x, y]
#             cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
#             cv2.imshow("image", img)
#         elif event == cv2.EVENT_RBUTTONDOWN:
#             keep_looping = False
#
#     cv2.setMouseCallback("image", mouse_callback)
#
#     while keep_looping:
#         cv2.waitKey(1)
#
#     cv2.destroyAllWindows()
#
#     return last_point


# def get_clicked_point(img_path, max_display_size=1200):
#     img = cv2.imread(img_path)
#     h, w = img.shape[:2]
#
#     # 计算缩放比例
#     scale = min(max_display_size / w, max_display_size / h, 1.0)
#
#     if scale < 1.0:
#         new_w, new_h = int(w * scale), int(h * scale)
#         img_display = cv2.resize(img, (new_w, new_h))
#         resize_factor = scale
#     else:
#         img_display = img
#         resize_factor = 1.0
#
#     cv2.namedWindow("Select Point", cv2.WINDOW_AUTOSIZE)
#     cv2.imshow("Select Point", img_display)
#
#     clicked_point = [0, 0]
#     keep_looping = True
#
#     def mouse_callback(event, x, y, flags, param):
#         nonlocal clicked_point, keep_looping
#         if event == cv2.EVENT_LBUTTONDOWN:
#             clicked_point = [x, y]
#             # 在缩放图上画点
#             cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
#             cv2.imshow("Select Point", img_display)
#             print(f"Selected point on scaled image: ({x}, {y})")
#             print(f"Original coordinates: ({int(x / resize_factor)}, {int(y / resize_factor)})")
#         elif event == cv2.EVENT_RBUTTONDOWN:
#             keep_looping = False
#
#     cv2.setMouseCallback("Select Point", mouse_callback)
#
#     print(f"Image scaled by factor: {resize_factor:.2f}")
#     print("Left click to select point, Right click to confirm and exit")
#
#     while keep_looping:
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # ESC键退出
#             break
#
#     cv2.destroyAllWindows()
#
#     # 返回原始坐标
#     original_x = int(clicked_point[0] / resize_factor)
#     original_y = int(clicked_point[1] / resize_factor)
#     return [original_x, original_y]





def get_clicked_point(img_path):
    img = cv2.imread(img_path)

    # 创建一个缩放比例，确保图片能完整显示在屏幕上
    screen_width, screen_height = 1920, 1080  # 假设屏幕分辨率
    h, w = img.shape[:2]

    # 计算缩放比例，确保图片不超过屏幕尺寸
    scale = min(screen_width / w, screen_height / h, 1.0)

    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        resize_factor = scale
    else:
        img_resized = img
        resize_factor = 1.0

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)  # 允许窗口缩放
    cv2.imshow("image", img_resized)

    last_point = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img_resized

        if event == cv2.EVENT_LBUTTONDOWN:
            # 将缩放后的坐标转换回原始坐标
            original_x = int(x / resize_factor)
            original_y = int(y / resize_factor)

            if last_point:
                cv2.circle(img_resized, (last_point[0], last_point[1]), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img_resized, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("image", img_resized)
            print(f"点击坐标 (原始图片): ({original_x}, {original_y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键退出
            break

    cv2.destroyAllWindows()

    # 返回原始坐标
    if last_point:
        original_x = int(last_point[0] / resize_factor)
        original_y = int(last_point[1] / resize_factor)
        return [original_x, original_y]
    else:
        return []


