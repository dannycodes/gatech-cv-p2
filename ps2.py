"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np


def filter_angle(lines, angles):
    """ Provide angle in degrees"""
    lines = lines.squeeze()
    filtered_lines = None

    def add(filtered_lines, line):
        if filtered_lines is None:
            filtered_lines = line
        else:
            filtered_lines = np.vstack((filtered_lines, line))
        return filtered_lines

    for line in lines:
        for angle in angles:
            if np.isclose(line[1], np.radians(angle), atol=np.radians(1)):
                filtered_lines = add(filtered_lines, line)
    return filtered_lines


def solve_for_intersection(lines):
    print(lines)
    thetas = lines[:, 1]
    rhos = lines[:, 0]
    a = np.array((np.sin(thetas), np.cos(thetas))).T
    b = rhos
    return np.linalg.solve(a, b)


def solve_all_intersections(lines):
    lines = lines.squeeze()
    vertices = None
    for idx, l1 in enumerate(lines):
        print(l1)
        for l2 in lines[idx+1:]:
            try:
                x = solve_for_intersection(np.array([l1, l2]))
            except np.linalg.LinAlgError:
                pass
            if vertices is None:
                vertices = x
            else:
                vertices = np.vstack((vertices, x))
    return vertices


def mask_image(img, lower_color, upper_color):
    mask = cv2.inRange(img, lower_color, upper_color)
    return cv2.bitwise_and(img, img, mask=mask)


def blur_image(img, ksize=5):
    return cv2.medianBlur(img, 5)


def canny(img):
    return cv2.Canny(img, 50, 150, apertureSize=3)


def deduplicate_points(vertices, dist=10):
    bins = []
    deduped_vertex = None

    def insert_vertex(deduped_vertex, vertex):
        if deduped_vertex is None:
            deduped_vertex = vertex
        else:
            deduped_vertex = np.vstack((deduped_vertex, vertex))
        return deduped_vertex

    for vertex in vertices:
        if len(bins) == 0:
            bins.append(vertex)
            deduped_vertex = insert_vertex(deduped_vertex, vertex)
        else:
            rdists = [np.sqrt((b[0] - vertex[0]) ** 2 + (b[1] - vertex[1]) ** 2)
                      for b in bins]

            if np.all(np.array(rdists) > 10):
                bins.append(vertex)
                deduped_vertex = insert_vertex(deduped_vertex, vertex)
    return deduped_vertex


def deduplicate_lines(lines, atol=20):
    """Takes lines from raw
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


def draw_lines_p(img, lines):
    lines = lines.squeeze()
    for line in lines:
        img = cv2.line(img, (line[1], line[0]),
                       (line[3], line[2]), (0, 0, 0), 1)
    return img


def threshold_hsv(color_img, lowH, lowS, lowV, highH, highS, highV):
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

    TITLE_WINDOW = "Thresholding on HSV"

    def do_threshold():
        thresholded = cv2.inRange(
            hsv, (lowH, lowS, lowV), (highH, highS, highV))
        cv2.imshow(TITLE_WINDOW, thresholded)
        return thresholded

    def on_lowH(_lowH):
        nonlocal lowH
        lowH = _lowH
        do_threshold()

    def on_lowS(_lowS):
        nonlocal lowS
        lowS = _lowS
        do_threshold()

    def on_lowV(_lowV):
        nonlocal lowV
        lowV = _lowV
        do_threshold()

    def on_highH(_highH):
        nonlocal highH
        highH = _highH
        do_threshold()

    def on_highS(_highS):
        nonlocal highS
        highS = _highS
        do_threshold()

    def on_highV(_highV):
        nonlocal highV
        highV = _highV
        do_threshold()

    cv2.namedWindow(TITLE_WINDOW)
    cv2.createTrackbar("lowH", TITLE_WINDOW, lowH, 179, on_lowH)
    cv2.createTrackbar("lowS", TITLE_WINDOW, lowS, 255, on_lowS)
    cv2.createTrackbar("lowV", TITLE_WINDOW, lowV, 255, on_lowV)
    cv2.createTrackbar("highH", TITLE_WINDOW, highH, 179, on_highH)
    cv2.createTrackbar("highS", TITLE_WINDOW, highS, 255, on_highS)
    cv2.createTrackbar("highV", TITLE_WINDOW, highV, 255, on_highV)
    return do_threshold()


def find_lines_p(img, rho, theta, threshold, **kwargs):
    return cv2.HoughLinesP(img, rho, np.radians(theta), threshold, **kwargs)


def find_lines_p_parameterized(img, draw_on_img, rho, theta, threshold,
                               minLineLength=None,
                               maxLineGap=None):
    TITLE_WINDOW = "Hough P Line Stuff"

    def find_lines_for_params():
        draw_on = np.copy(draw_on_img)
        found_lines = find_lines_p(img, rho, theta, threshold,
                                   minLineLength=minLineLength,
                                   maxLineGap=maxLineGap)
        if found_lines is not None:
            draw_on = draw_lines_p(draw_on, found_lines)
            # cv2.imshow(TITLE_WINDOW, draw_on)

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
                       minLineLength, 500, on_minLineLength)
    cv2.createTrackbar("maxLineGap", TITLE_WINDOW,
                       maxLineGap, 500, on_maxLineGap)

    return find_lines_for_params()


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
        found_lines = find_lines(img, rho, theta, threshold,
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
                           circle[2], (0, 0, 255), 1)

        # TOGGLE THIS IF YOU WANT TO WORK WITH PARAMS
        cv2.imshow(TITLE_WINDOW, draw_on)
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
    cv2.createTrackbar("minRadius", TITLE_WINDOW,
                       minradius, 100, on_min_radius)
    cv2.createTrackbar("maxRadius", TITLE_WINDOW, maxradius, 60, on_max_radius)
    return circle_param_finder()


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """

    img = np.copy(img_in)
    color_img = np.copy(img_in)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow('gaussed', img)
    draw_on = np.copy(img)
    img = blur_image(img, 5)
    cv2.imshow('blurred', img)

    lowH = 0
    lowS = 0
    lowV = 15
    highH = 148
    highS = 255
    highV = 68
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_filtered_img = threshold_hsv(
        color_img, lowH, lowS, lowV, highH, highS, highV)
    cv2.waitKey(0)

    ##
    # Hough Parameters (Found Experimentally)
    ##

    dp = 6
    min_dist = 17
    param1 = 39
    param2 = 103
    minradius = 14
    maxradius = 33

    # TESTING
    # draw_on = np.copy(img)
    # circles = find_circles_parameterized(color_filtered_img, draw_on,
    #                                      dp=dp,
    #                                      min_dist=min_dist,
    #                                      param1=param1,
    #                                      param2=param2,
    #                                      minradius=minradius,
    #                                      maxradius=maxradius)
    # cv2.waitKey(0)

    circles = find_circles(color_filtered_img, dp, min_dist,
                           param1=param1,
                           param2=param2,
                           minRadius=minradius,
                           maxRadius=maxradius)

    # Likely no traffic light in here
    if circles is None:
        return None, None

    # grab coordinates for yellow (should be middle circle)
    # TODO Do this by color of found pixel?
    yellow_light = None
    red_light = None
    green_light = None
    on_light = None

    def check_on(hsv):
        # check if light is on
        return hsv[2] > 245

    for circle in circles:
        x = int(circle[1])
        y = int(circle[0])
        hsv = hsv_img[x, y]
        if 25 < hsv[0] < 35:
            yellow_light = (y, x)
            if check_on(hsv):
                on_light = 'yellow'
        if hsv[0] < 5 or hsv[0] > 175:
            # red_light = (y, x)
            if check_on(hsv):
                on_light = 'red'
        if 55 < hsv[0] < 65:
            # green_light = (y, x)
            if check_on(hsv):
                on_light = 'green'

    if yellow_light is None:
        print("COULD NOT FIND YELLOW LIGHT")
        return None, None

    return yellow_light, on_light


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    valid_angles = [30, 150]

    img = np.copy(img_in)
    draw_on = np.copy(img_in)
    img = mask_image(img, (0, 0, 255), (0, 0, 255))
    img = blur_image(img, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    rho = 1
    theta = 180
    threshold = 38

    lines = find_lines(edges, rho, theta, threshold)
    # lines = find_lines_parameterized(edges, draw_on, rho, theta,
    #                                  threshold,
    #                                  minLineLength=minLineLength,
    #                                  maxLineGap=maxLineGap)

    if lines is None:
        return None
    sides = None
    for angle in valid_angles:
        _sides = lines[np.isclose(lines[:, 0, 1], np.radians(angle)), :, :]
        _sides = _sides.squeeze()
        # idx = np.argmin(_sides[:, 0])
        # _sides = _sides[[idx], :]

        if sides is None:
            sides = _sides
        else:
            sides = np.vstack((sides, _sides))
    intersects = None
    for i in range(len(sides)):
        for j in range(len(sides)):
            if i != j and i < j:
                _thetas = sides[(i, j), 1]
                _rhos = sides[(i, j), 0]
                a = np.array((np.sin(_thetas), np.cos(_thetas))).T
                b = _rhos
                try:
                    _intersects = np.linalg.solve(a, b)
                except np.linalg.LinAlgError:
                    continue

                if intersects is None:
                    intersects = _intersects
                else:
                    intersects = np.vstack((intersects, _intersects))
    if intersects is None:
        return None
    x_max = np.max(intersects[:, 0])
    x_min = np.min(intersects[:, 0])
    SCALE_FACTOR = 6  # Only hardcoded piece (ratio for yield sign)
    height = x_max - x_min
    x_center = x_min - height * SCALE_FACTOR * (1.0/3)
    y_center = np.mean(intersects[:, 1])

    # center = np.sum(intersects, axis=0) / 3
    for intersect in intersects:
        # print(intersects)
        # print(tuple(intersect))
        img = cv2.circle(img, (intersect[1], intersect[0]), 3, (0, 0, 0), 1)
    # print(y_center, x_center)
    img = cv2.circle(img, (int(y_center), int(x_center)), 3, (0, 0, 0), 1)
    # cv2.imshow("yield", img)
    # cv2.waitKey(0)
    return y_center, x_center


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    img = np.copy(img_in)
    draw_on = np.copy(img_in)

    img = mask_image(img,
                     np.array([0, 0, 204]),
                     np.array([0, 0, 204]))
    img = blur_image(img, 5)
    edges = canny(img)

    rho = 1
    theta = 180
    threshold = 30

    # cv2.imshow('canny', edges)

    lines = find_lines(edges, rho, theta, threshold)

    # lines = find_lines_parameterized(edges, draw_on, rho, theta,
    #                                  threshold,
    #                                  minLineLength=minLineLength,
    #                                  maxLineGap=maxLineGap)
    # deduped_lines = deduplicate_lines(lines, 2)
    deduped_lines = lines.squeeze()
    vertices = None

    for idx, l1 in enumerate(deduped_lines):
        for l2 in deduped_lines[idx+1:]:
            # print(l1, l2)
            radial_distance = np.abs(l1[1] - l2[1])
            if np.isclose(l1[1], l2[1], atol=np.radians(4)):
                continue

            if np.isclose(radial_distance,
                          np.radians(90), atol=np.radians(10)):
                x = solve_for_intersection(np.array([l1, l2]))
                if vertices is None:
                    vertices = x
                else:
                    vertices = np.vstack([vertices, x])
    filtered_vertices = deduplicate_points(vertices, 3)

    # pre_img = np.copy(img)
    # for vertex in vertices:
    #     cv2.circle(pre_img, (vertex[1], vertex[0]), 3,
    #                (255, 255, 255), 1)
    # cv2.imshow('edges', edges)
    # cv2.imshow('Pre-filter', pre_img)
    # post_img = np.copy(img)
    # for vertex in filtered_vertices:
    #     cv2.circle(post_img, (vertex[1], vertex[0]), 3,
    #                (255, 255, 255), 1)
    # cv2.imshow('Post-filter', post_img)
    # cv2.waitKey(0)
    center = parallelogram_centroid_from_vertices(filtered_vertices)
    # print(center)
    # cv2.circle(img, center, 3, (255, 255, 255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return center


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img = np.copy(img_in)
    draw_on = np.copy(img_in)

    lower_white = np.array([0, 255, 255])
    upper_white = np.array([0, 255, 255])

    # Preprocessing
    img = mask_image(img, lower_white, upper_white)
    img = blur_image(img)
    edges = canny(img)

    # find the lines
    rho = 1
    theta = 36
    threshold = 36

    lines = find_lines(edges, rho, theta, threshold)
    # lines = find_lines_parameterized(edges, draw_on, rho, theta,  threshold)
    if lines is None:
        return None
    deduped_lines = deduplicate_lines(lines)
    parallel_lines = group_parallel_lines(deduped_lines)
    vertices = parallelogram_vertices_from_grouped_lines(parallel_lines)
    # for vertex in vertices:
    #     cv2.circle(img, (vertex[1], vertex[0]), 3,
    #                (255, 255, 255), 1)

    # cv2.imshow('Warning Sign Edges', edges)
    # cv2.imshow('Warning Sign Masked', img)

    # cv2.waitKey(0)

    return parallelogram_centroid_from_vertices(vertices)


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img = np.copy(img_in)
    draw_on = np.copy(img_in)
    upper = np.array([0, 128, 255])
    lower = np.copy(upper)
    img = mask_image(img, lower, upper)
    img = blur_image(img)
    edges = canny(img)

    # find the lines
    rho = 1
    theta = 36
    threshold = 36
    lines = find_lines(edges, rho, theta, threshold)
    # lines = find_lines_parameterized(edges, draw_on, rho, theta,
    #  threshold)
    # cv2.waitKey(0)
    # print(lines)

    lines = filter_angle(lines, [45, 135])

    deduped_lines = deduplicate_lines(lines)

    grouped_lines = group_parallel_lines(deduped_lines)
    vertices = parallelogram_vertices_from_grouped_lines(grouped_lines)
    center = parallelogram_centroid_from_vertices(vertices)

    # for vertex in vertices:
    #     draw_on = cv2.circle(draw_on, (vertex[1], vertex[0]), 3, (0, 0, 0), 1)
    # draw_on = cv2.circle(draw_on, center, 3, (0, 0, 0), 1)

    # cv2.imshow('construction sign', draw_on)
    # cv2.waitKey(0)
    return center


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    img = np.copy(img_in)
    draw_on = np.copy(img_in)
    img = mask_image(img, np.array([0, 0, 255]), np.array([0, 0, 255]))
    img = blur_image(img, 101)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    # cv2.imshow("DO NOT ENTER", img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("GRAY", img_gray)
    dp = 3
    min_dist = 14
    param1 = 11
    param2 = 65
    minradius = 28
    maxradius = 40

    # circles = find_circles_parameterized(img_gray, draw_on,
    #                                      dp=dp,
    #                                      min_dist=min_dist,
    #                                      param1=param1,
    #                                      param2=param2,
    #                                      minradius=minradius,
    #                                      maxradius=maxradius)

    circles = find_circles(img_gray, dp, min_dist,
                           param1=param1,
                           param2=param2,
                           minRadius=minradius,
                           maxRadius=maxradius)

    # cv2.waitKey(0)
    x = int(circles[0][0])
    y = int(circles[0][1])
    return x, y  # reversed for numpy


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    sign_dict = {}
    img = np.copy(img_in)

    traffic_light_sign, _ = traffic_light_detection(img_in, range(7, 31))
    if traffic_light_sign is not None:
        sign_dict["traffic_light"] = traffic_light_sign

    dne_sign = do_not_enter_sign_detection(img)
    if dne_sign is not None:
        sign_dict['no_entry'] = dne_sign

    stop_sign = stop_sign_detection(img)
    if stop_sign is not None:
        sign_dict['stop'] = stop_sign

    construction_sign = construction_sign_detection(img)
    if construction_sign is not None:
        sign_dict['construction'] = construction_sign

    warning_sign = warning_sign_detection(img)
    if warning_sign is not None:
        sign_dict['warning'] = warning_sign

    # cv2.imshow("Traffic Sign", img)
    # cv2.waitKey(0)

    yield_sign = yield_sign_detection(img)
    if yield_sign:
        sign_dict['yield_sign'] = yield_sign

    return sign_dict


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    sign_dict = {}
    img = np.copy(img_in)

    traffic_light_sign, _ = traffic_light_detection(img_in, range(7, 31))
    if traffic_light_sign is not None:
        sign_dict["traffic_light"] = traffic_light_sign

    dne_sign = do_not_enter_sign_detection(img)
    if dne_sign is not None:
        sign_dict['no_entry'] = dne_sign

    stop_sign = stop_sign_detection(img)
    if stop_sign is not None:
        sign_dict['stop'] = stop_sign

    construction_sign = construction_sign_detection(img)
    if construction_sign is not None:
        sign_dict['construction'] = construction_sign

    warning_sign = warning_sign_detection(img)
    if warning_sign is not None:
        sign_dict['warning'] = warning_sign

    # cv2.imshow("Traffic Sign", img)
    # cv2.waitKey(0)

    yield_sign = yield_sign_detection(img)
    if yield_sign:
        sign_dict['yield_sign'] = yield_sign

    return sign_dict


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError


# CHANGE BELOW FOR MORE CUSTOMIZATION ##
#####################
""" The functions below are used for each individual part of
the report section.

Feel free to change the return statements but ensure that the return type
remains the same
for the autograder.

"""

# Part 2 outputs


def ps2_2_a_1(img_in):
    return do_not_enter_sign_detection(img_in)


def ps2_2_a_2(img_in):
    return stop_sign_detection(img_in)


def ps2_2_a_3(img_in):
    return construction_sign_detection(img_in)


def ps2_2_a_4(img_in):
    return warning_sign_detection(img_in)


def ps2_2_a_5(img_in):
    return yield_sign_detection(img_in)


# Part 3 outputs
def ps2_3_a_1(img_in):
    return traffic_sign_detection(img_in)


def ps2_3_a_2(img_in):
    return traffic_sign_detection(img_in)


# Part 4 outputs
def ps2_4_a_1(img_in):
    return traffic_sign_detection_noisy(img_in)


def ps2_4_a_2(img_in):
    return traffic_sign_detection_noisy(img_in)

# Part 5 outputs


def ps2_5_a(img_in):
    return traffic_sign_detection_challenge(img_in)
