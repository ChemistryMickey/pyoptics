import json

import numpy as np

from elements.optical_elements import OpticalElement, FreePropagation
from beams.ray import Ray


class OpticalSystem:
    def __init__(self, optical_elements: list[OpticalElement]):
        if len(optical_elements) == 0:
            raise Exception(
                "Cannot have an empty optical system! Add some optical elements"
            )

        optical_elements = sorted(optical_elements, key=lambda el: el.pos[2])

        # If there are any gaps, fill in that gap with a free-propagation
        free_props = []
        for i in range(len(optical_elements[:-1])):
            pos_diff = optical_elements[i + 1].pos[2] - optical_elements[i].pos[2]
            if pos_diff != 0:
                free_props.append(
                    FreePropagation(
                        pos_diff, pos=np.array([0, 0, optical_elements[i].pos[2]])
                    )
                )
        optical_elements += free_props

        optical_elements = sorted(optical_elements, key=lambda el: el.pos[2])
        self.elements = optical_elements

        self.get_meta_optical_properties()

    def __repr__(self) -> str:
        internal_els = json.dumps([str(el) for el in self.elements], indent=3)
        return internal_els

    def refract_ray(self, ray: Ray, debug_print: bool = False) -> None:
        # WARNING: Mutates ray
        if debug_print:
            print(f"Ray in: {ray}")

        for el in self.elements:
            el.refract_ray(ray)

            if debug_print:
                print(f"Refracted ray after {el.__class__.__name__}: {ray}")

    def get_meta_optical_properties(self) -> None:
        self.M = np.array([[1, 0], [0, 1]])
        for el in self.elements:
            self.M = np.matmul(self.M, el.M)

        self.nodal_points = np.array(
            [
                (self.M[1, 1] - 1) / self.M[1, 0],
                (self.elements[0].n / self.elements[-1].n - self.M[0, 0])
                / self.M[1, 0],
            ]
        )

        self.principle_planes = np.array(
            [
                (self.M[1, 1] - self.elements[0].n / self.elements[-1].n)
                / self.M[1, 0],
                (1 - self.M[0, 0]) / self.M[1, 0],
            ]
        )

        self.focal_points = np.array(
            [self.M[1, 1] / self.M[1, 0], -self.M[0, 0] / self.M[1, 0]]
        )

        self.focal_length = -(self.elements[0].n / self.elements[-1].n) / self.M[1, 0]

        self.optical_power = -self.M[1, 0]

        return
