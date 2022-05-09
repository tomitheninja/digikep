import cv2
import numpy as np


def rotate_image(image, angle):
    """Rotate the image by the given angle

    Args:
        image (np.array): 
        angle (float32): 0-360

    Returns:
        np.array: rotated image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def get_rotation(angle: float, axis=4):
    """Return the signed angle diff to the closest axis

    Args:
        angle (float): angle in radian
        axis (int, optional): Number of axis. Defaults to 4.

    Returns:
        _type_: angle to the closest axis
    """
    domain = np.pi / 2 * (axis / 4)

    theta = angle % domain
    return -theta if theta < domain / 2 else domain - theta


def get_rotation_between_points(point1, point2):
    """Rotation between two points in degrees

    Args:
        point1 ((float, float)):
        point2 ((float, float)):

    Returns:
        float32: degrees
    """
    b = point2[0] - point1[0]
    a = point2[1] - point1[1]
    return get_rotation(-np.arctan(a / b)) * 180 / np.pi if b != 0 else 0


def process_lines(lines):
    """Normalizes lines and returns the rotations and weights

    Args:
        lines (_type_): result of hough lines

    Returns:
        (np. array, np.array, np.array): lines, rotations, weights
    """
    dxs = lines[:, 2] - lines[:, 0]
    dys = lines[:, 3] - lines[:, 1]
    thetas = np.arctan2(dys, dxs)
    lens_sqrt = dxs ** 2 + dys ** 2
    max_len = np.max(lens_sqrt)
    weights = (lens_sqrt / max_len) ** 3

    rotations = np.vectorize(get_rotation)(thetas)

    # filter noise by eleminating items with small weight
    not_noise = weights > 0.25
    # print(weights[not_noise])

    return lines[not_noise, :], rotations[not_noise], weights[not_noise]


def main(original_image, blur_image=True):
    img = original_image.copy()

    if blur_image:
        cv2.GaussianBlur(img, (0, 0), 3)
        struct = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.dilate(img, struct)
        img = cv2.erode(img, struct)
        img = cv2.erode(img, struct)
        img = cv2.dilate(img, struct)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=15, maxLineGap=15)
    if lines is None:
        if not blur_image:
            return 0, original_image
        else:
            return main(original_image, False)

    lines = np.resize(lines, (lines.shape[0], 4))
    lines, rotations, weights = process_lines(lines)
    # print(f"number of lines: {rotations.shape[0]}")

    sum_rotations = np.sum(weights * rotations)
    sum_weights = np.sum(weights)

    for i, (x1, y1, x2, y2) in enumerate(lines):
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), round(weights[i] * 15))

    angle = -sum_rotations / sum_weights * 180 / np.pi
    return angle, img


original_image = cv2.imread('2.jpg', cv2.IMREAD_COLOR)
assert original_image is not None

angle, debug_img = main(original_image)
cv2.imshow('original', original_image)
cv2.imshow('rotated', rotate_image(original_image, angle))
cv2.imshow('debug', rotate_image(debug_img, angle))

point1 = None


def on_mouse(event, x, y, flags, param):
    """Rotate the image by the mouse position

    Args:
        event (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        flags (_type_): _description_
        param (_type_): _description_
    """
    global point1
    global angle
    global original_image
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        if abs(point1[0] - point2[0]) > 10 or abs(point1[1] - point2[1]) > 10:
            angle = get_rotation_between_points(point1, point2)
            cv2.imshow('rotated', rotate_image(original_image, angle))
        point1 = None
    result = original_image.copy()
    if point1 is not None:
        cv2.line(result, point1, (x, y), (0, 0, 255), 2)

    cv2.imshow('original', result)


cv2.setMouseCallback('original', on_mouse)

while True:
    k = cv2.waitKey(0)
    if k == ord('q') or k == 27:
        break

cv2.imwrite('rotated.jpg', rotate_image(original_image, angle))

cv2.destroyAllWindows()
