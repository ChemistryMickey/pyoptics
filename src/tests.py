import numpy as np

from elements.optical_elements import (
    OpticalElement,
    FreePropagation,
    ThickLens,
    ThinLens,
    CurvedMirror,
)
from beams.ray import Ray


def test_element_focal_length():
    el = OpticalElement(np.array([[0, 0], [-1 / 2, 0]]))

    assert el.get_focal_length() == 2


def test_optical_element():
    free_prop = OpticalElement(np.array([[1, 2], [0, 1]]))
    print(f"{free_prop = }")

    ray = Ray(np.array([0, 1, 0]), th=np.pi / 4)
    print(f"{ray = }")

    (ray.tail_position[1], ray.th) = free_prop.refract_ray(ray)
    print(f"{ray = }")

    assert ray.tail_position[1] == 2


def test_free_prop():
    free_prop = FreePropagation(2)
    ray = Ray(np.array([0, 1, 0]), th=np.pi / 4)
    print(f"{ray = }")

    (ray.tail_position[1], ray.th) = free_prop.refract_ray(ray)
    print(f"{ray = }")
    assert ray.tail_position[1] == 2


def test_curved_mirror():
    curved_mirror = CurvedMirror(1)
    ray = Ray(np.array([0, 1, 0]))
    print(f"{ray = }")

    (ray.tail_position[1], ray.th) = curved_mirror.refract_ray(ray)
    print(f"{ray = }")

    assert ray.th == -2


def test_thin_lens():
    thin_lens = ThinLens(focal_length=1)
    ray = Ray(np.array([0, 1, 0]))
    print(f"{ray = }")

    (ray.tail_position[1], ray.th) = thin_lens.refract_ray(ray)
    print(f"{ray = }")

    assert ray.th == -1


def test_thick_lens():
    thick_lens = ThickLens(
        left_index=1,
        left_radius=1e-3,
        right_index=1,
        right_radius=1e-3,
        n=1,
        thickness=1,
    )  # basically a passthrough
    ray = Ray(np.array([0, 1, 0]))
    print(f"{ray = }")

    (ray.tail_position[1], ray.th) = thick_lens.refract_ray(ray)
    print(f"{ray = }")

    assert ray.th == 0

    thick_lens2 = ThickLens(
        left_index=1,
        left_radius=1,
        right_index=1,
        right_radius=1,
        n=1.5,
        thickness=1,
    )  # actually glass
    ray = Ray(np.array([0, 1, 0]))
    print(f"{ray = }")

    (ray.tail_position[1], ray.th) = thick_lens2.refract_ray(ray)
    print(f"{ray = }")

    assert round(ray.th, 3) == -0.167

    thick_lens3 = ThickLens(
        left_index=1,
        left_radius=1,
        right_index=1,
        right_radius=1,
        n=1.5,
        thickness=2,
    )  # thicker
    ray = Ray(np.array([0, 1, 0]))
    print(f"{ray = }")

    (ray.tail_position[1], ray.th) = thick_lens3.refract_ray(ray)
    print(f"{ray = }")

    assert round(ray.th, 3) == -0.333
