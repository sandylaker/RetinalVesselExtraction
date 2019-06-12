import numpy as np
from typing import NamedTuple, List, Optional
import matplotlib.pyplot as plt
import skimage
from skimage import io, filters, feature
import os

# Copy from https://github.com/sunsided/stroke-width-transform/

Image = np.ndarray
GradientImage = np.ndarray
Position = NamedTuple('Position', [('x', int), ('y', int)])
Stroke = NamedTuple('Stroke', [('x', int), ('y', int), ('width', float)])
Ray = List[Position]
Component = List[Position]
Gradients = NamedTuple('Gradients', [('x', GradientImage), ('y', GradientImage)])


def get_edges(im: Image, **kwargs) -> Image:
    edges = feature.canny(im, **kwargs)
    return edges.astype(np.float)


def get_gradients(im: Image) -> Gradients:
    grad_x = filters.scharr_h(im)
    grad_y = filters.scharr_v(im)
    return Gradients(x=grad_x, y=grad_y)


def get_gradient_directions(g: Gradients) -> Image:
    return np.arctan2(g.y, g.x)

def apply_swt(im: Image, edges: Image, gradients: Gradients, dark_on_bright: bool=True) -> Image:
    # Prepare the output map
    swt = np.squeeze(np.ones_like(im)) * np.Infinity

    # For each pixel, let's obtain the normal direction of its gradient
    norms = np.sqrt(gradients.x ** 2 + gradients.y ** 2)
    # deal with the pixels with 0 norms
    norms[norms == 0] = 1
    inv_norms = 1. / norms
    directions = Gradients(x=gradients.x * inv_norms, y=gradients.y * inv_norms)

    # we keep track of all the rays found in the image.
    rays = []

    # Find a pixel that lies on an edge
    height, width = im.shape[0:2]
    for y in range(height):
        for x in range(width):
            # Edges are either 0 or 1
            if edges[y, x] < 0.5:
                continue
            ray = swt_process_pixel(Position(x=x, y=y), edges, directions, out=swt,
                                    dark_on_bright=dark_on_bright)
            if ray:
                rays.append(ray)

    # Multiple rays may cross the same pixel and each pixel has the smallest
    # stroke width of those.
    # A problem are corners like the edge of an L. Here, two rays will be found,
    # both of which are significantly longer than the actual width of each
    # individual stroke. To mitigate, we will visit each pixel on each ray and
    # take the median stroke length over all pixels on the ray.

    for ray in rays:
        median = np.median([swt[p.y, p.x] for p in ray])
        for p in ray:
            swt[p.y, p.x] = min(median, swt[p.y, p.x])

    swt[swt == np.Infinity] = 0
    return swt


def swt_process_pixel(pos: Position, edges: Image, directions: Gradients, out: Image,
                      dark_on_bright: bool=True) -> Optional[Ray]:
    """
    Obtains the stroke width starting from the specified position.
    :param pos: The starting point
    :param edges: The edges.
    :param directions: The normalized gradients
    :param out: The output image.
    :param dark_on_bright: Enables dark-on-bright text detection.
    :return (Ray): a list of pixel positions, representing a ray
    """
    # Keep track of the image dimensions for boundary tests.
    height, width = edges.shape[0:2]

    # The direction in which we travel the gradient depends on the type of text
    # we want to find. For dark text on light background, follow the opposite
    # direction (into the dark are): for light text on dark background, follow
    # the gradient as is.
    gradient_direction = -1 if dark_on_bright else 1

    # Starting from the current pixel we will shoot a ray into the direction of the pixel's
    # gradient and keep track of all pixels in that direction that still lie on an edge.
    ray = [pos]

    # obtain the direction to the step into
    dir_x = directions.x[pos.y, pos.x]
    dir_y = directions.y[pos.y, pos.x]

    # Since some pixels have no gradient, normalization of the gradient
    # is a division by zero for them, resulting in NaN. These values
    # should not bother us since we explicitly tested for an edges before.
    assert not (np.isnan(dir_x) or np.isnan(dir_y))

    # Traverse the pixels along the direction.
    prev_pos = Position(x=-1, y=-1)
    steps_taken = 0
    while True:
        # Advance to the next pixel on the line
        steps_taken += 1
        cur_x = int(np.floor(pos.x + gradient_direction * dir_x * steps_taken))
        cur_y = int(np.floor(pos.y + gradient_direction * dir_y * steps_taken))
        cur_pos = Position(x=cur_x, y=cur_y)
        if cur_pos == prev_pos:
            continue
        prev_pos = Position(x=cur_x, y=cur_y)
        # If we reach the edge of the image without crossing a stroke edge,
        # we discard the result.
        if not ((0 <= cur_x < width) and 0 <= cur_y < height):
            return None
        # The point is either on the line or the end of it, so we register it
        ray.append(cur_pos)
        # If that pixel is not an edge, we are still on the line and need to
        # continue scanning.
        if edges[cur_y, cur_x] < 0.5:
            continue
        # If this edge is pointed in a direction approximately opposite of the
        # one we started in, it is approximately parallel. This means we
        # just found the other side of the stroke.
        # The original paper suggests the gradients need to be opposite +/- PI/6.
        # Since the dot product is the cosine of the enclosed angle and
        # cos(pi/6) = 0.8660254037844387, we can discard all values that exceed
        # this threshold.
        cur_dir_x = directions.x[cur_y, cur_x]
        cur_dir_y = directions.y[cur_y, cur_x]
        dot_product = dir_x * cur_dir_x + dir_y * cur_dir_y
        if dot_product >= -0.866:
            return None
        # Paint each of the pixels on the ray with their determined stroke width
        stroke_width = np.sqrt((cur_x - pos.x) * (cur_x - pos.x) + (cur_y - pos.y) * (cur_y -
                                                                                      pos.y))
        for p in ray:
            out[p.y, p.x] = min(stroke_width, out[p.y, p.x])

        return ray