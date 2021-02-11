"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np


# Helper methods I Made
def find_circles(img, dp, min_dist, **kwargs):
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, dp, min_dist, **kwargs)
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

        for circle in circles:

            cv2.circle(draw_on, (circle[0], circle[1]),
                       circle[2], (255, 255, 255), 1)

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
    cv2.createTrackbar("param1", TITLE_WINDOW, param1, 10, on_param_1)
    cv2.createTrackbar("param2", TITLE_WINDOW, param2, 100, on_param_2)
    cv2.createTrackbar("minRadius", TITLE_WINDOW, minradius, 10, on_min_radius)
    cv2.createTrackbar("maxRadius", TITLE_WINDOW, maxradius, 40, on_max_radius)
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

    img_in = np.copy(img_in)
    # draw_on = np.copy(img_in)
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    ##
    # Hough Parameters (Found Experimentally)
    ##

    dp = 1
    min_dist = 42
    param1 = 1
    param2 = 5
    minradius = 10
    maxradius = 31

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

    # Handle the sun
    # TODO: Only deletes the 4th item
    # To make more general, instead keep the smallest 3 items
    # NOTE: IF we assume that the stoplight will be aligned vertically,
    # we can dramatically improve the distance measure
    if len(circles) > 3:
        # we've detected the sun
        coords = circles[:, (0, 1)]
        circle_count = len(circles)
        farthest_object_idx = None
        farthest_object_closeness_score = None
        for i, c in enumerate(coords):
            global_closeness = 0
            for j in range(circle_count):
                if j != i:
                    global_closeness += np.sqrt(np.sum(((c - coords[j]) ** 2)))
            if farthest_object_closeness_score is None or \
                    global_closeness > farthest_object_closeness_score:
                farthest_object_closeness_score = global_closeness
                farthest_object_idx = i
        circles = np.delete(circles, (farthest_object_idx), axis=0)

    # grab coordinates for yellow (should be middle circle)
    # TODO Do this by color of found pixel?
    yellow_light = circles[1]
    coordinates = (yellow_light[0], yellow_light[1])

    # Find the "brightest" light
    # This will be the 'on' light
    # TODO This logic might be brittle
    overall_brightest = None
    overall_brightest_name = None
    for color_name, color in zip(['red', 'yellow', 'green'], circles):
        x = int(np.floor(color[0]))
        y = int(np.floor(color[1]))
        cv2.imshow(color_name, img_in[y-35:y+35, x-35:x+35, :])
        center = img_in[y, x, :]

        # Not good enough to do this naively, probably need to sum over
        # or something
        if color_name == "yellow":
            max_brightness = np.sum(center) / 2
        else:
            max_brightness = np.sum(center)

        # print(color_name, max_brightness, center)
        if overall_brightest is None or max_brightness > overall_brightest:
            overall_brightest = max_brightness
            overall_brightest_name = color_name

    # print(coordinates, overall_brightest_name)
    # cv2.imshow("traffic light original", img_in)
    # cv2.imshow("traffic light", draw_on)
    # cv2.waitKey(0)
    return coordinates, overall_brightest_name


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """

    raise NotImplementedError


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    raise NotImplementedError


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """

    # detect a stop sign
    raise NotImplementedError


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    raise NotImplementedError


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    img_in = np.copy(img_in)
    draw_on = np.copy(img_in)
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("DO NOT ENTER", img_in)

    dp = 1
    min_dist = 10
    param1 = 1
    param2 = 10
    minradius = 7
    maxradius = 37

    circles = find_circles_parameterized(img_gray, draw_on,
                                         dp=dp,
                                         min_dist=min_dist,
                                         param1=param1,
                                         param2=param2,
                                         minradius=minradius,
                                         maxradius=maxradius)

    # circles = find_circles(img_gray, dp, min_dist,
    #                        param1=param1,
    #                        param2=param2,
    #                        minRadius=minradius,
    #                        maxRadius=maxradius)

    print(circles[0])
    x = int(circles[0][0])
    y = int(circles[0][1])

    cv2.imshow('one way sign', img_in[y-40:y+40, x-40:x+40, :])
    cv2.waitKey(0)

    return y, x  # reversed for numpy


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
    raise NotImplementedError


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
    raise NotImplementedError


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
