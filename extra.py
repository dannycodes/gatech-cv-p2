
    # Handle the sun
    # TODO: Only deletes the 4th item
    # To make more general, instead keep the smallest 3 items
    # NOTE: IF we assume that the stoplight will be aligned vertically,
    # we can dramatically improve the distance measure
    # if len(circles) > 3:
    #     # we've detected the sun
    #     coords = circles[:, (0, 1)]
    #     circle_count = len(circles)
    #     farthest_object_idx = None
    #     farthest_object_closeness_score = None
    #     for i, c in enumerate(coords):
    #         global_closeness = 0
    #         for j in range(circle_count):
    #             if j != i:
    #                 global_closeness += np.sqrt(np.sum(((c - coords[j]) ** 2)))
    #         if farthest_object_closeness_score is None or \
    #                 global_closeness > farthest_object_closeness_score:
    #             farthest_object_closeness_score = global_closeness
    #             farthest_object_idx = i
    #     circles = np.delete(circles, (farthest_object_idx), axis=0)