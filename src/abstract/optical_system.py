import json

import numpy as np

from elements.optical_elements import OpticalElement, FreePropagation
from beams.ray import Ray


class OpticalSystem:
    def __init__(self, optical_elements: list[OpticalElement]):
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

    def __repr__(self) -> str:
        return json.dumps([str(el) for el in self.elements], indent=3)

    def refract_ray(self, ray: Ray) -> None:
        # WARNING: Mutates ray
        print(f"Ray in: {ray}")

        for el in self.elements:
            el.refract_ray(ray)
            print(f"Refracted ray after {el.__class__.__name__}: {ray}")
