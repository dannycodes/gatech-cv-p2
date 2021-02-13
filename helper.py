import numpy as np
import cv2


def solve_for_intersection(lines):
    thetas = lines[:, 1]
    rhos = lines[:, 0]
    a = np.array((np.sin(thetas), np.cos(thetas))).T
    b = rhos
    return np.linalg.solve(a, b)


def mask_image(img, lower_color, upper_color):
    mask = cv2.inRange(img, lower_color, upper_color)
    return cv2.bitwise_and(img, img, mask=mask)


def blur_image(img, ksize=5):
    return cv2.medianBlur(img, 5)


def canny(img):
    return cv2.Canny(img, 50, 150, apertureSize=3)


def deduplicate_lines(lines, atol=20):
    """Takes lines from raw find_lines/find_lines_parameterized
    output"""
    centers = []
    deduped_lines = None
    for line in lines.squeeze():
        # dedup lines
        # we want 4 lines
        loc = np.linalg.norm(line)
        if len(centers) == 0 or not np.any(np.isclose(loc, centers, atol=atol)):
            centers.append(loc)
            if deduped_lines is None:
                deduped_lines = np.array([line])
            else:
                deduped_lines = np.vstack((deduped_lines, line))
    return deduped_lines


def group_parallel_lines(lines):
    """Takes squeezed lines as input"""
    parallel_bins = {}
    for line in lines:
        if len(parallel_bins) == 0:
            parallel_bins[line[1]] = [line]
        else:
            keys = list(parallel_bins.keys())
            closeness = np.isclose(line[1], np.array(
                keys), atol=np.radians(10))
            if np.any(closeness):
                k = keys[np.where(closeness)[0][0]]
                parallel_bins[k].append(line)
            else:
                parallel_bins[line[1]] = [line]
    return list(parallel_bins.values())


def parallelogram_vertices_from_grouped_lines(lines):
    """Given an array of groups of lines
        [np.array(lines), np.array(lines),...]
        Return parallelogram vertices
        Only works if there are two bins of parallel lines
    """
    if len(lines) > 2:
        raise Exception("parallelogram finder \
            called with too many lines")
    c_1 = lines[0]
    c_2 = lines[1]
    intercepts = None
    for l1, l2 in list(zip(c_1, c_2)) + list(zip(c_1, c_2[::-1])):
        x = solve_for_intersection(np.array([l1, l2]))
        if intercepts is None:
            intercepts = np.array([x])
        else:
            intercepts = np.vstack((intercepts, x))
    return intercepts


def parallelogram_centroid_from_vertices(vertices):
    return np.mean(vertices[:, 1]), np.mean(vertices[:, 0])


def find_lines(img, rho, theta, threshold, **kwargs):
    return cv2.HoughLines(img, rho, np.pi / theta, threshold)


def draw_lines(img, lines, color=(0, 0, 0)):
    img = np.copy(img)
    for line in lines.squeeze():
        _rho, _theta = line
        a = np.cos(_theta)
        b = np.sin(_theta)
        x0 = a*_rho
        y0 = b*_rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        img = cv2.line(img,
                       (x1, y1),
                       (x2, y2),
                       color,
                       3)
    return img


def find_lines_parameterized(img, draw_on_img,
                             rho, theta, threshold,
                             minLineLength=None, maxLineGap=None):

    TITLE_WINDOW = "Line Stuff"

    def find_lines_for_params():
        draw_on = np.copy(draw_on_img)
        found_lines = find_lines(img, rho, np.pi / theta, threshold,
                                 minLineLength=minLineLength,
                                 maxLineGap=maxLineGap)
        if found_lines is not None:
            draw_on = draw_lines(draw_on, found_lines)
            cv2.imshow(TITLE_WINDOW, draw_on)

        return found_lines

    def on_rho(_rho):
        nonlocal rho
        rho = _rho
        find_lines_for_params()

    def on_theta(_theta):
        nonlocal theta
        theta = _theta
        find_lines_for_params()

    def on_threshold(_threshold):
        nonlocal threshold
        threshold = _threshold
        find_lines_for_params()

    def on_minLineLength(_minLineLength):
        nonlocal minLineLength
        minLineLength = _minLineLength
        find_lines_for_params()

    def on_maxLineGap(_maxLineGap):
        nonlocal maxLineGap
        maxLineGap = _maxLineGap
        find_lines_for_params()

    cv2.namedWindow(TITLE_WINDOW)
    cv2.createTrackbar("rho", TITLE_WINDOW, rho, 10, on_rho)
    cv2.createTrackbar("theta", TITLE_WINDOW, theta, 500, on_theta)
    cv2.createTrackbar("threshold", TITLE_WINDOW, threshold, 500, on_threshold)
    cv2.createTrackbar("minLineLength", TITLE_WINDOW,
                       minLineLength, 30, on_minLineLength)
    cv2.createTrackbar("maxLineGap", TITLE_WINDOW,
                       maxLineGap, 60, on_maxLineGap)
    return find_lines_for_params()


def find_circles(img, dp, min_dist, **kwargs):
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, dp, min_dist, **kwargs)
    if circles is None:
        return circles
    return circles[0]


def find_circles_parameterized(img, draw_on_img, dp=None, min_dist=None,
                               param1=None, param2=None, minradius=None,
                               maxradius=None):
    TITLE_WINDOW = "Circle Stuff"  # A constant for param exploration plot

    ##
    # Use those params to
    # find circles using hough transform
    ##
    def circle_param_finder():
        # Draw circles onto draw_on
        draw_on = np.copy(draw_on_img)

        circles = find_circles(img, dp, min_dist,
                               param1=param1,
                               param2=param2,
                               minRadius=minradius,
                               maxRadius=maxradius)
        if circles is not None:
            for circle in circles:

                cv2.circle(draw_on, (circle[0], circle[1]),
                           circle[2], (0, 0, 0), 1)

        # TOGGLE THIS IF YOU WANT TO WORK WITH PARAMS
        # cv2.imshow(TITLE_WINDOW, draw_on)
        return circles

    def on_dp(_dp):
        nonlocal dp
        dp = _dp
        print(dp)
        circle_param_finder()

    def on_min_dist(_min_dist):
        nonlocal min_dist
        min_dist = _min_dist
        circle_param_finder()

    def on_param_1(_in):
        nonlocal param1
        param1 = _in
        circle_param_finder()

    def on_param_2(_in):
        nonlocal param2
        param2 = _in
        circle_param_finder()

    def on_min_radius(_in):
        nonlocal minradius
        minradius = _in
        circle_param_finder()

    def on_max_radius(_in):
        nonlocal maxradius
        maxradius = _in
        circle_param_finder()

    cv2.namedWindow(TITLE_WINDOW)
    cv2.createTrackbar("dp", TITLE_WINDOW, dp, 10, on_dp)
    cv2.createTrackbar("minDist", TITLE_WINDOW, min_dist, 50, on_min_dist)
    cv2.createTrackbar("param1", TITLE_WINDOW, param1, 100, on_param_1)
    cv2.createTrackbar("param2", TITLE_WINDOW, param2, 200, on_param_2)
    cv2.createTrackbar("minRadius", TITLE_WINDOW, minradius, 30, on_min_radius)
    cv2.createTrackbar("maxRadius", TITLE_WINDOW, maxradius, 60, on_max_radius)
    return circle_param_finder()
