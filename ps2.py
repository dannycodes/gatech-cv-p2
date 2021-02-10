"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np


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
    cv2.imshow("traffic light", img_in)
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    print(radii_range[-1])
    a = None
    circles = cv2.HoughCircles(
        img_gray, cv2.HOUGH_GRADIENT, 1, 20,
        circles=a,
        param1=20,
        param2=10,
        minRadius=radii_range[0],
        maxRadius=radii_range[-1] + 10)
    print(a)
    print(circles.shape)
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for circle in circles[0]:
        img = cv2.circle(img, (circle[0], circle[1]),
                         circle[2], (0, 0, 255), 1)
    cv2.imshow("traffic light w/ circles", img)

    circles = circles[0]

    green_light = circles[0]
    yellow_light = circles[1]
    red_light = circles[2]

    coordinates = (yellow_light[0], yellow_light[1])

    overall_brightest = None
    overall_brightest_name = None
    for color_name, color in zip(['green', 'yellow', 'red'], circles):

        # TODO figure out the indexes, this is pointing to some rando pixel
        center = img_in[int(np.floor(color[0])), int(np.floor(color[1])), :]
        print(center)

        # Not good enough to do this naively, probably need to sum over
        # or something

        max_brightness = np.sum(center)
        print(color_name, max_brightness)
        if overall_brightest is None or max_brightness > overall_brightest:
            overall_brightest = max_brightness
            overall_brightest_name = color_name
    print(coordinates)
    print(overall_brightest_name)
    cv2.waitKey(0)
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
    raise NotImplementedError


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


################################ CHANGE BELOW FOR MORE CUSTOMIZATION #######################
""" The functions below are used for each individual part of the report section.

Feel free to change the return statements but ensure that the return type remains the same 
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
