import numpy as np

from abstract.optical_system import OpticalSystem
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

    free_prop.refract_ray(ray)

    assert ray.tail_position[1] == 2


def test_free_prop():
    free_prop = FreePropagation(2)
    ray = Ray(np.array([0, 1, 0]), th=np.pi / 4)

    free_prop.refract_ray(ray)

    assert ray.tail_position[1] == 2
    assert ray.tail_position[2] == free_prop.d


def test_curved_mirror():
    curved_mirror = CurvedMirror(1)
    ray = Ray(np.array([0, 1, 0]))

    curved_mirror.refract_ray(ray)

    assert ray.th == -2


def test_thin_lens():
    thin_lens = ThinLens(focal_length=1)
    ray = Ray(np.array([0, 1, 0]))

    thin_lens.refract_ray(ray)

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

    thick_lens.refract_ray(ray)

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

    thick_lens2.refract_ray(ray)

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

    thick_lens3.refract_ray(ray)

    assert round(ray.th, 3) == -0.333

    thick_lens4 = ThickLens(
        left_index=1,
        left_radius=2,
        right_index=1,
        right_radius=1,
        n=1.5,
        thickness=2,
    )  # thicker
    ray = Ray(np.array([0, 1, 0]))

    thick_lens4.refract_ray(ray)

    assert round(ray.th, 3) == 0.083

    thick_lens5 = ThickLens(
        left_index=1,
        left_radius=2,
        right_index=1,
        right_radius=3,
        n=1.5,
        thickness=2,
    )  # thicker
    ray = Ray(np.array([0, 1, 0]))

    thick_lens5.refract_ray(ray)

    assert round(ray.th, 3) == -0.139


def test_optical_system():
    import json
    from random import shuffle

    lenses = [ThinLens(focal_length=1, pos=np.array([0, 0, i])) for i in range(8)]
    shuffle(lenses)
    opt_sys = OpticalSystem(lenses)  # evenly spaced thin lenses
    print(opt_sys)

    n_els = json.loads(str(opt_sys))
    assert n_els.__len__() == 15

    ray = Ray(np.array([0, 1, 0]))

    opt_sys.refract_ray(ray)
    assert round(ray.th, 3) == -1.0
    assert ray.tail_position[2] == 7

    # Empty case
    try:
        opt_sys = OpticalSystem([])
        raise Exception("Should have errored")
    except:
        pass

    # Check optical properties
    assert opt_sys.focal_length == 1

    ray2 = Ray(np.array([0, 1, 0]))
    refracted = np.matmul(opt_sys.M, np.array([ray2.tail_position[1], ray2.th]))
    assert (refracted == np.array([ray.tail_position[1], ray.th])).all()
