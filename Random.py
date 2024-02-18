import random


def random_range(min_value: float, max_value: float, distribution: float = 1.):
    r = random.uniform(0, 1) ** distribution
    return r * (max_value - min_value) + min_value


def random_bool(chance: float):
    return random.uniform(0, 1) < chance
