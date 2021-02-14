"""
CS6476: Problem Set 2 Tests

In this script you will find some simple tests that will help you
determine if your implementation satisfies the autograder
requirements. In this collection of tests your code output will be
tested to verify if the correct data type is returned. Additionally,
there are a couple of examples with sample answers to guide you better
in developing your algorithms.
"""

import cv2
import unittest
import ps2
import experiment


def check_result(label, coords, ref, tol):
    assert (abs(coords[0] - ref[0]) <= tol and
            abs(coords[1] - ref[1]) <= tol), "Wrong coordinate values. " \
                                             "Image used: {}. " \
                                             "Expected: ({}, {}), " \
                                             "Returned: ({}, {}). " \
                                             "Max tolerance: {}." \
                                             "".format(label, ref[0], ref[1],
                                                       coords[0], coords[1],
                                                       tol)


class TestTrafficLight(unittest.TestCase):
    """Test Traffic Light Detection"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_output(self):
        test_image = cv2.imread("input_images/test_images/simple_tl_test.png")
        radii_range = range(10, 30, 1)
        result = ps2.traffic_light_detection(test_image, radii_range)

        self.assertTrue(result is not None, "Output is NoneType.")
        self.assertEqual(2, len(result), "Output should be a tuple of 2 "
                                         "elements.")

        coords = result[0]
        state = result[1]

        is_tuple = isinstance(coords, (tuple))
        self.assertTrue(is_tuple, "Coordinates output is not a tuple.")

        is_string = isinstance(state, str)
        self.assertTrue(is_string, "Traffic light state is not a string.")

        if state not in ["red", "yellow", "green"]:
            raise (ValueError, "Traffic light state is not valid.")

    def test_simple_tl(self):
        tl_images = {"simple_tl_test": {"x": 45, "y": 120, "state": "green"},
                     "tl_green_299_287_blank": {"x": 287, "y": 299,
                                                "state": "green"},
                     "tl_red_199_137_blank": {"x": 137, "y": 199,
                                              "state": "red"},
                     "tl_yellow_199_237_blank": {"x": 237, "y": 199,
                                                 "state": "yellow"}}

        radii_range = range(10, 30, 1)

        for tl in tl_images:
            tl_data = tl_images[tl]
            test_image = cv2.imread("input_images/test_images/"
                                    "{}.png".format(tl))

            coords, state = ps2.traffic_light_detection(test_image,
                                                        radii_range)

            check_result(tl, coords, (tl_data["x"], tl_data["y"]), 5)
            self.assertEqual(state, tl_data["state"], "Wrong state value.")

    def test_scene_tl(self):
        tl_images = {"scene_tl_test": {"x": 338, "y": 200, "state": "red"},
                     "tl_green_299_287_background": {"x": 287, "y": 299,
                                                     "state": "green"},
                     "tl_red_199_137_background": {"x": 137, "y": 199,
                                                   "state": "red"},
                     "tl_yellow_199_237_background": {"x": 237, "y": 199,
                                                      "state": "yellow"}}

        radii_range = range(10, 30, 1)

        for tl in tl_images:
            tl_data = tl_images[tl]
            test_image = cv2.imread("input_images/test_images/"
                                    "{}.png".format(tl))

            coords, state = ps2.traffic_light_detection(test_image,
                                                        radii_range)

            check_result(tl, coords, (tl_data["x"], tl_data["y"]), 5)
            self.assertEqual(state, tl_data["state"], "Wrong state value.")


class TestTrafficSignsBlank(unittest.TestCase):
    """Test Traffic Sign Detection using a blank background"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_stop_sign(self):
        image_name = "stop_249_149_blank.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.stop_sign_detection(test_image)

        check_result(image_name, coords, (149, 249), 5)

    def test_construction_sign(self):
        image_name = "construction_150_200_blank.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.construction_sign_detection(test_image)

        check_result("construction_150_200_blank", coords, (200, 150), 5)

    def test_warning_sign(self):
        image_name = "warning_250_300_blank.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.warning_sign_detection(test_image)

        check_result(image_name, coords, (300, 250), 5)

    def test_do_not_enter_sign(self):
        image_name = "no_entry_145_145_blank.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.do_not_enter_sign_detection(test_image)

        check_result(image_name, coords, (145, 145), 5)

    def test_yield_sign(self):
        image_name = "yield_173_358_blank.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.yield_sign_detection(test_image)

        check_result(image_name, coords, (358, 173), 5)


class TestTrafficSignsScene(unittest.TestCase):
    """Test Traffic Sign Detection using a simulated street scene"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_stop_sign(self):
        image_name = "stop_249_149_background.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.stop_sign_detection(test_image)

        check_result(image_name, coords, (149, 249), 5)

    def test_construction_sign(self):
        image_name = "construction_150_200_background.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.construction_sign_detection(test_image)

        check_result("construction_150_200_blank", coords, (200, 150), 5)

    def test_warning_sign(self):
        image_name = "warning_250_300_background.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.warning_sign_detection(test_image)

        check_result(image_name, coords, (300, 250), 5)

    def test_do_not_enter_sign(self):
        image_name = "no_entry_145_145_background.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.do_not_enter_sign_detection(test_image)

        check_result(image_name, coords, (145, 145), 5)

    def test_yield_sign(self):
        image_name = "yield_173_358_background.png"
        test_image = cv2.imread("input_images/test_images/" + image_name)
        coords = ps2.yield_sign_detection(test_image)

        check_result(image_name, coords, (358, 173), 5)


class TestAllSignDetection(unittest.TestCase):

    def setUp(self):
        self.image_name = "scene_all_signs"
        self.sign_img = cv2.imread(f"input_images/{self.image_name}.png")

    def show_img(self, coords, sign_name, skip=False):
        if skip is not True:
            temp_dict = {}
            temp_dict[sign_name] = coords
            img_out = experiment.mark_traffic_signs(self.sign_img, temp_dict)
            cv2.imshow(self.image_name, img_out)
            cv2.waitKey(0)

    def test_dne_sign(self):
        coords = ps2.do_not_enter_sign_detection(self.sign_img)
        self.show_img(coords, "dne", skip=True)
        check_result(self.image_name, coords, (235, 334), 5)

    def test_stop_sign(self):
        coords = ps2.stop_sign_detection(self.sign_img)
        self.show_img(coords, "stop", skip=True)
        check_result(self.image_name, coords, (348, 348), 10)

    def test_construction_sign(self):
        coords = ps2.construction_sign_detection(self.sign_img)
        self.show_img(coords, "construction", skip=True)
        check_result(self.image_name, coords, (651, 348), 5)

    def test_warning_sign(self):
        coords = ps2.warning_sign_detection(self.sign_img)
        self.show_img(coords, "warning", skip=True)
        check_result(self.image_name, coords, (801, 349), 5)

    def test_yield_sign(self):
        coords = ps2.yield_sign_detection(self.sign_img)
        self.show_img(coords, "yield", skip=True)
        check_result(self.image_name, coords, (507, 335), 5)

    def test_traffic_light(self):
        coords, _ = ps2.traffic_light_detection(
            self.sign_img, range(10, 30, 1))
        self.show_img(coords, "light", skip=True)
        check_result(self.image_name, coords, (115, 339), 5)

    def test_all_correct(self):
        sign_dict = ps2.traffic_sign_detection(self.sign_img)

        # img_out = experiment.mark_traffic_signs(self.sign_img, sign_dict)
        # print(sign_dict)
        # cv2.imshow(self.image_name, img_out)
        # cv2.waitKey(0)

        sign_dict_empirical = {
            'traffic_light': (115.0, 339.0),
            'no_entry': (235, 334),
            'stop': (348.59302, 348.1466),
            'construction': (649.8311, 350.01782),
            'warning': (799.73773, 350.01782),
            'yield_sign': (507.20224, 337.9999694824219)}
        for key in sign_dict_empirical.keys():
            check_result(self.image_name,
                         sign_dict[key],
                         sign_dict_empirical[key],
                         5)


class TestSomeSignScene(unittest.TestCase):
    def setUp(self):
        self.image_name = "scene_some_signs"
        self.sign_img = cv2.imread(f"input_images/{self.image_name}.png")

    def test_some_correct(self):
        sign_dict = ps2.traffic_sign_detection(self.sign_img)

        # img_out = experiment.mark_traffic_signs(self.sign_img, sign_dict)
        # cv2.imshow(self.image_name, img_out)
        # cv2.waitKey(0)

        sign_dict_empirical = {
            'no_entry': (151, 451),
            'stop': (548, 148),
            'construction': (849, 350.01782)}
        for key in sign_dict_empirical.keys():
            check_result(self.image_name,
                         sign_dict[key],
                         sign_dict_empirical[key],
                         5)


class TestNoisyImage(unittest.TestCase):
    def test_some_noisy_correct(self):
        self.image_name = "scene_some_signs_noisy"
        self.sign_img = cv2.imread(f"input_images/{self.image_name}.png")
        sign_dict = ps2.traffic_sign_detection(self.sign_img)

        sign_dict_empirical = {
            'traffic_light': (872, 160),
            'no_entry': (646, 245),
            'warning': (400, 449),
            'yield_sign': (156, 335)}

        # cv2.imshow("Original Sign", self.sign_img)
        # img_out = experiment.mark_traffic_signs(self.sign_img, sign_dict)
        # print(sign_dict_empirical)
        # cv2.imshow(self.image_name, img_out)
        # cv2.waitKey(0)

        for key in sign_dict_empirical.keys():
            print(key)
            check_result(self.image_name,
                         sign_dict[key],
                         sign_dict_empirical[key],
                         5)

    def test_all_noisy_correct(self):
        self.image_name = "scene_all_signs_noisy"
        self.sign_img = cv2.imread(f"input_images/{self.image_name}.png")
        sign_dict = ps2.traffic_sign_detection(self.sign_img)

        sign_dict_empirical = {
            'traffic_light': (472, 361),
            'no_entry': (346, 444),
            'stop': (650, 197),
            'construction': (250, 198),
            'warning': (799, 350),
            'yield_sign': (157, 337)}

        # cv2.imshow("Original Sign", self.sign_img)
        # img_out = experiment.mark_traffic_signs(self.sign_img, sign_dict)
        # print(sign_dict_empirical)
        # cv2.imshow(self.image_name, img_out)
        # cv2.waitKey(0)

        for key in sign_dict_empirical.keys():
            print(key)
            check_result(self.image_name,
                         sign_dict[key],
                         sign_dict_empirical[key],
                         5)


class Part1Tests(unittest.TestCase):
    def test_simple_tl(self):
        image_name = "simple_tl"
        tl = cv2.imread(f"input_images/{image_name}.png")
        coords, state = ps2.traffic_light_detection(tl, range(10, 30, 1))
        # img_out = experiment.draw_tl_center(tl, coords, state)
        # cv2.imshow(image_name, img_out)
        # cv2.waitKey(0)
        check_result(image_name, coords, (136, 122), 5)

    def test_scene_tl_1(self):
        image_name = "scene_tl_1"
        tl = cv2.imread(f"input_images/{image_name}.png")
        coords, state = ps2.traffic_light_detection(tl, range(10, 30, 1))
        # img_out = experiment.draw_tl_center(tl, coords, state)
        # cv2.imshow(image_name, img_out)
        # cv2.waitKey(0)
        check_result(image_name, coords, (438,  249), 5)


class Part2Tests(unittest.TestCase):
    def test_scene_yld_1(self):
        image_name = "scene_yld_1"
        sign_img = cv2.imread(f"input_images/{image_name}.png")
        coords = ps2.ps2_2_a_5(sign_img)
        # temp_dict = {image_name: coords}
        # img_out = experiment.mark_traffic_signs(sign_img, temp_dict)
        # cv2.imshow(image_name, img_out)
        # cv2.waitKey(0)
        check_result(image_name, coords, (307, 182), 5)

    def test_dne_1(self):
        image_name = "scene_dne_1"
        sign_img = cv2.imread(f"input_images/{image_name}.png")
        coords = ps2.ps2_2_a_1(sign_img)
        # temp_dict = {image_name: coords}
        # img_out = experiment.mark_traffic_signs(sign_img, temp_dict)
        # cv2.imshow(image_name, img_out)
        # cv2.waitKey(0)
        check_result(image_name, coords, (246, 346), 5)


if __name__ == "__main__":
    unittest.main()
