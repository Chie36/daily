import numpy as np
import matplotlib.pyplot as plt


class OccMap:

    def __init__(self, obstacle_vec, bounds, resolution):
        self.obstacle_vec = obstacle_vec
        self.bounds = bounds
        self.resolution = resolution
        self.limit_bound()

    def limit_bound(self):
        pass

    def build_bin_map(self):
        pass


def test():
    obstacle_vec = None
    bounds = None
    resolution = 0.2
    occ_map = OccMap(obstacle_vec, bounds, resolution)


if __name__ == "__main__":
    test()
