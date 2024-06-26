import numpy as np

from beams.ray import Ray


class OpticalElement:
    def __init__(
        self,
        M: np.ndarray = np.zeros((2, 2)),
        n: float = 1,
        thickness: float = 0,
        spherical_ab: np.ndarray = np.ones((2, 2)),
        chromatic_ab: np.ndarray = np.ones((2, 2)),
    ):
        self.M = M
        self.n = n
        self.thickness = thickness
        self.spherical_ab = spherical_ab
        self.chromatic_ab = chromatic_ab

    def refract_ray(self, ray: Ray) -> tuple[float, float]:
        return np.matmul(self.M, np.array([ray.tail_position[1], ray.th]))

    def get_focal_length(self):
        return -1 / self.M[1, 0]

    def __repr__(self):
        return f"{self.M.flatten() = }"


# ========= Individual elements


class CurvedMirror(OpticalElement):
    def __init__(
        self,
        curvature: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.curvature = curvature
        self.M = np.array([[1, 0], [-2 / curvature, 1]])


class FreePropagation(OpticalElement):
    def __init__(
        self,
        distance: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d = distance
        self.M = np.array([[1, distance], [0, 1]])


class OpticalFlat(OpticalElement):
    def __init__(
        self,
        left_n: float = 1,
        right_n: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.left_n = left_n
        self.right_n = right_n
        self.M = np.array([[1, 0], [0, np.arcsin(left_n / right_n)]])


class ThickLens(OpticalElement):
    def __init__(
        self,
        left_index: float = 1,
        left_radius: float = 1,
        right_index: float = 1,
        right_radius: float = 1,
        n: float = 1,
        thickness: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n

        ref1 = np.array(
            [
                [1, 0],
                [(n - right_index) / (right_index * right_radius), n / right_index],
            ]
        )
        free_prop = np.array([[1, thickness], [0, 1]])
        ref2 = np.array(
            [[1, 0], [(left_index - n) / (n * left_radius), left_index / n]]
        )
        self.M = np.matmul(
            np.matmul(
                ref1,
                free_prop,
            ),
            ref2,
        )


class ThinLens(OpticalElement):
    def __init__(
        self,
        focal_length: float | None = None,
        lens_radius: float | None = None,
        outside_n: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if focal_length is not None:
            self.f = focal_length
            self.M = np.array([[1, 0], [-1 / focal_length, 1]])
            return

        if lens_radius is not None and outside_n is not None:
            self.f = (self.n * lens_radius) / (outside_n - self.n)
            self.M = np.array([[1, 0], [-1 / self.f, outside_n / self.n]])
            return

        raise Exception(
            "Insufficient configuration to create a thin lens!"
            + "Either supply a focal length OR a lens radius and "
            + "outside index of refraction (and make sure the internal index of refraction is what you expect!)"
        )
