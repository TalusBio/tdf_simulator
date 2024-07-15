import numpy as np


def is_sorted_asc(arr: np.ndarray) -> bool:
    return np.all(arr[:-1] <= arr[1:])


def is_sorted_desc(arr: np.ndarray) -> bool:
    return np.all(arr[:-1] >= arr[1:])
