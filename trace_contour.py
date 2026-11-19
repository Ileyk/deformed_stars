"""Trace and resample a closed contour of the user-defined scalar field."""

from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# Adjust these values to select the contour and sampling density.
Q = 0.5
# F0 is computed from the inner Lagrange point L1.
SEED_X_RANGE = (-0.1, 0.1)
SEED_Y_MIN = 0.0
SEED_SEARCH_GRID_POINTS = 501
N_POINTS = 200
OUTPUT_FILE = Path("contour_points.csv")
PLOT_FILE = Path("contour_map.png")
PLOT_X_RANGE = (-1.0, 1.0)
PLOT_Y_RANGE = (-1.0, 2.0)

# Integration controls. The curvature limit bounds the local chord error.
CURVE_TOLERANCE = 1e-4
MAX_STEP = 0.1
RTOL = 1e-9
ATOL = 1e-11
CLOSURE_TOLERANCE = 1e-7
MAX_LENGTH = 100.0


class SeedNotFoundError(ValueError):
    """Raised when the requested level has no detected seed in the search box."""


def source_locations_and_weights(q: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the singularity locations and weights for the inverse-distance terms."""
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("Q must be a finite positive number.")

    locations = np.array([[0.0, 0.0], [0.0, 1.0]])
    weights = np.array([q / (1.0 + q), 1.0 / (1.0 + q)])
    return locations, weights


def f(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    q: float = Q,
) -> Union[float, np.ndarray]:
    """Evaluate the user's scalar field."""
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("Q must be a finite positive number.")

    x_values, y_values = np.broadcast_arrays(x, y)
    y_g = 1.0 / (1.0 + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            -q / (1.0 + q) / np.sqrt(x_values**2 + y_values**2)
            -1.0 / (1.0 + q) / np.sqrt(x_values**2 + (y_values - 1.0) ** 2)
            -0.5 * (x_values**2 + (y_values - y_g) ** 2)
        )


def potential_gradient_y(y: float, q: float) -> float:
    """Evaluate the y-derivative of f on the line x=0."""
    locations, weights = source_locations_and_weights(q)
    offsets = y - locations[:, 1]
    y_g = 1.0 / (1.0 + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(weights * offsets / np.abs(offsets) ** 3) - (y - y_g))


def find_lagrange_points(q: float) -> dict[str, np.ndarray]:
    """Find L3, L1, and L2 on x=0, in order from y<0 to y>1."""
    source_locations_and_weights(q)
    offset = 1e-4

    for _ in range(60):
        y_l3_near = -offset
        if potential_gradient_y(y_l3_near, q) < 0.0:
            break
        offset *= 0.1
    else:
        raise RuntimeError("Could not bracket L3 next to the body at y=0.")
    y_l3_far = -1.0
    while potential_gradient_y(y_l3_far, q) <= 0.0:
        y_l3_far *= 2.0
    y_l3 = brentq(lambda y: potential_gradient_y(y, q), y_l3_far, y_l3_near)

    offset = 1e-4
    for _ in range(60):
        y_l1_left = offset
        y_l1_right = 1.0 - offset
        if (
            potential_gradient_y(y_l1_left, q) > 0.0
            and potential_gradient_y(y_l1_right, q) < 0.0
        ):
            break
        offset *= 0.1
    else:
        raise RuntimeError("Could not bracket L1 between the two bodies.")
    y_l1 = brentq(lambda y: potential_gradient_y(y, q), y_l1_left, y_l1_right)

    offset = 1e-4
    for _ in range(60):
        y_l2_near = 1.0 + offset
        if potential_gradient_y(y_l2_near, q) > 0.0:
            break
        offset *= 0.1
    else:
        raise RuntimeError("Could not bracket L2 next to the body at y=1.")
    y_l2_far = 2.0
    while potential_gradient_y(y_l2_far, q) >= 0.0:
        y_l2_far *= 2.0
    y_l2 = brentq(lambda y: potential_gradient_y(y, q), y_l2_near, y_l2_far)

    return {
        "L3": np.array([0.0, y_l3]),
        "L1": np.array([0.0, y_l1]),
        "L2": np.array([0.0, y_l2]),
    }


LAGRANGE_POINTS = find_lagrange_points(Q)
SEED_Y_RANGE = (SEED_Y_MIN, 0.99 * LAGRANGE_POINTS["L1"][1])
F0 = float(f(*LAGRANGE_POINTS["L1"], Q))


def gradient_and_curvature(x: float, y: float, q: float) -> tuple[np.ndarray, float]:
    """Return grad(f) and the absolute curvature of its regular level set."""
    grad = gradient_f(x, y, q)
    grad_norm = np.linalg.norm(grad)
    if grad_norm <= np.finfo(float).eps:
        raise ValueError("The contour reached a critical point where its tangent is undefined.")

    locations, weights = source_locations_and_weights(q)
    displacement = np.array([x, y]) - locations
    radii = np.linalg.norm(displacement, axis=1)
    hessian = np.zeros((2, 2))
    identity = np.eye(2)
    for delta, radius, weight in zip(displacement, radii, weights):
        hessian += weight * (
            identity / radius**3 - 3.0 * np.outer(delta, delta) / radius**5
        )
    hessian -= identity
    tangent = np.array([grad[1], -grad[0]]) / grad_norm
    curvature = abs(tangent @ hessian @ tangent) / grad_norm
    return grad, curvature


def gradient_f(x: float, y: float, q: float) -> np.ndarray:
    """Return the gradient of f at one point."""
    locations, weights = source_locations_and_weights(q)
    displacement = np.array([x, y]) - locations
    radii = np.linalg.norm(displacement, axis=1)
    if np.any(radii == 0.0):
        raise ValueError("The gradient is undefined at a singularity of the scalar field.")

    grad = np.sum(weights[:, np.newaxis] * displacement / radii[:, np.newaxis] ** 3, axis=0)
    y_g = 1.0 / (1.0 + q)
    return grad - np.array([x, y - y_g])


def find_contour_start(
    q: float,
    f0: float,
    x_range: tuple[float, float] = SEED_X_RANGE,
    y_range: tuple[float, float] = SEED_Y_RANGE,
) -> np.ndarray:
    """Find a point on f=f0 inside the configured rectangular search region."""
    if not np.isfinite(f0):
        raise ValueError("F0 must be a finite number.")
    if f0 >= 0.0:
        raise SeedNotFoundError(
            "This field is strictly negative; a finite contour requires F0 < 0."
        )

    source_locations_and_weights(q)
    for name, bounds in (("x", x_range), ("y", y_range)):
        if (
            len(bounds) != 2
            or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(f"Seed search {name}-range must contain finite increasing bounds.")
    if SEED_SEARCH_GRID_POINTS < 2:
        raise ValueError("SEED_SEARCH_GRID_POINTS must be at least 2.")

    x_values = np.linspace(*x_range, SEED_SEARCH_GRID_POINTS)
    y_values = np.linspace(*y_range, SEED_SEARCH_GRID_POINTS)
    xx, yy = np.meshgrid(x_values, y_values)
    residual = f(xx, yy, q) - f0

    def find_crossing(
        first_value: float,
        second_value: float,
        first_point: tuple[float, float],
        second_point: tuple[float, float],
    ) -> Union[np.ndarray, None]:
        if np.isnan(first_value) or np.isnan(second_value):
            return None
        if first_value == 0.0:
            return np.asarray(first_point)
        if second_value == 0.0:
            return np.asarray(second_point)
        if (first_value < 0.0) == (second_value < 0.0):
            return None

        def residual_on_edge(value: float) -> float:
            point = (
                (value, first_point[1])
                if first_point[1] == second_point[1]
                else (first_point[0], value)
            )
            return float(f(point[0], point[1], q) - f0)

        root = brentq(
            residual_on_edge,
            first_point[0] if first_point[1] == second_point[1] else first_point[1],
            second_point[0] if first_point[1] == second_point[1] else second_point[1],
        )
        return (
            np.array([root, first_point[1]])
            if first_point[1] == second_point[1]
            else np.array([first_point[0], root])
        )

    for row in range(SEED_SEARCH_GRID_POINTS):
        for column in range(SEED_SEARCH_GRID_POINTS - 1):
            seed = find_crossing(
                residual[row, column],
                residual[row, column + 1],
                (x_values[column], y_values[row]),
                (x_values[column + 1], y_values[row]),
            )
            if seed is not None:
                return seed

    for row in range(SEED_SEARCH_GRID_POINTS - 1):
        for column in range(SEED_SEARCH_GRID_POINTS):
            seed = find_crossing(
                residual[row, column],
                residual[row + 1, column],
                (x_values[column], y_values[row]),
                (x_values[column], y_values[row + 1]),
            )
            if seed is not None:
                return seed

    raise SeedNotFoundError(
        f"No point with f(x, y) = {f0:g} was found in seed search box "
        f"x={x_range}, y={y_range} at grid resolution "
        f"{SEED_SEARCH_GRID_POINTS}x{SEED_SEARCH_GRID_POINTS}."
    )


def trace_contour(
    q: float,
    f0: float,
    x_range: tuple[float, float] = SEED_X_RANGE,
    y_range: tuple[float, float] = SEED_Y_RANGE,
) -> np.ndarray:
    """Trace one closed component, returning points in traversal order."""
    start = find_contour_start(q, f0, x_range, y_range)
    start_gradient, start_curvature = gradient_and_curvature(*start, q)
    start_tangent = np.array([start_gradient[1], -start_gradient[0]])
    start_tangent /= np.linalg.norm(start_tangent)
    min_return_length = (
        min(MAX_STEP, 0.1 / start_curvature)
        if start_curvature > 0.0
        else MAX_STEP
    )

    def tangent(_arc_length: float, point: np.ndarray) -> np.ndarray:
        grad, _ = gradient_and_curvature(point[0], point[1], q)
        # Rotating the gradient gives a unit tangent; its sign sets orientation.
        return np.array([grad[1], -grad[0]]) / np.linalg.norm(grad)

    def returned_to_start(arc_length: float, point: np.ndarray) -> float:
        if arc_length < min_return_length:
            return 1.0
        return np.dot(point - start, start_tangent)

    returned_to_start.terminal = True
    returned_to_start.direction = 1

    points = [start]
    arc_length = 0.0
    point = start.copy()
    while arc_length < MAX_LENGTH:
        _, curvature = gradient_and_curvature(point[0], point[1], q)
        curvature_step = (
            np.sqrt(8.0 * CURVE_TOLERANCE / curvature)
            if curvature > 0.0
            else MAX_STEP
        )
        step_limit = min(MAX_STEP, curvature_step)
        if step_limit < 1e-8:
            raise RuntimeError("Curvature requires a step smaller than the supported minimum.")

        solution = solve_ivp(
            tangent,
            (arc_length, arc_length + step_limit),
            point,
            method="RK45",
            rtol=RTOL,
            atol=ATOL,
            max_step=step_limit,
            events=returned_to_start,
        )
        if not solution.success:
            raise RuntimeError(f"Contour integration failed: {solution.message}")

        points.extend(solution.y[:, 1:].T)
        point = solution.y[:, -1]
        arc_length = solution.t[-1]
        if solution.t_events[0].size:
            if np.linalg.norm(point - start) > CLOSURE_TOLERANCE:
                raise RuntimeError("The return event did not reach the starting point closely enough.")
            points.append(start)
            return np.asarray(points)

    raise RuntimeError(
        "No return to the starting point was found before MAX_LENGTH; "
        "the selected level may not be a closed regular contour."
    )


def resample_by_arclength(
    points: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Resample an open polyline uniformly, including both endpoints."""
    if n_points < 2:
        raise ValueError("At least two output points are required.")

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative_length[-1] <= 0.0:
        raise ValueError("Cannot resample a contour with zero length.")

    distances = np.linspace(0.0, cumulative_length[-1], n_points)
    return np.column_stack(
        [np.interp(distances, cumulative_length, points[:, axis]) for axis in range(2)]
    )


def extract_positive_half_contour(
    contour: np.ndarray,
    start_point: np.ndarray,
) -> np.ndarray:
    """Return the x>=0 arc from start_point to the next crossing of x=0."""
    if not np.allclose(contour[0], contour[-1]):
        raise ValueError("The positive half can only be extracted from a closed contour.")

    unique_points = contour[:-1].copy()
    nearest_index = np.argmin(np.linalg.norm(unique_points - start_point, axis=1))
    nearest_distance = np.linalg.norm(unique_points[nearest_index] - start_point)
    snap_tolerance = 10.0 * max(CURVE_TOLERANCE, CLOSURE_TOLERANCE)
    if nearest_distance > snap_tolerance:
        raise ValueError(
            f"L1 is {nearest_distance:.3g} from the traced contour, exceeding the "
            f"{snap_tolerance:.3g} snap tolerance."
        )

    unique_points[nearest_index] = start_point
    aligned = np.concatenate(
        (unique_points[nearest_index:], unique_points[:nearest_index])
    )
    nonzero_x = aligned[1:, 0][np.abs(aligned[1:, 0]) > 1e-12]
    if nonzero_x.size == 0:
        raise ValueError("The traced contour has no points away from x=0.")
    if nonzero_x[0] < 0.0:
        aligned = np.concatenate((aligned[:1], aligned[:0:-1]))

    half = [start_point.copy()]
    for point in aligned[1:]:
        if point[0] > 1e-12:
            half.append(point)
            continue
        if abs(point[0]) <= 1e-12:
            crossing = point.copy()
        else:
            previous = half[-1]
            fraction = previous[0] / (previous[0] - point[0])
            crossing = previous + fraction * (point - previous)
            crossing[0] = 0.0
        if np.linalg.norm(half[-1] - crossing) > 0.0:
            half.append(crossing)
        return np.asarray(half)

    raise ValueError("The traced contour did not return to x=0 on its x>=0 arc.")


def plot_contour_map(
    q: float,
    f0: float,
    integration_points: Union[np.ndarray, None] = None,
    samples: Union[np.ndarray, None] = None,
    seed_point: Union[np.ndarray, None] = None,
    resampling_start: Union[np.ndarray, None] = None,
) -> None:
    """Display and save the contour map, with traced points when available."""
    x = np.linspace(*PLOT_X_RANGE, 501)
    y = np.linspace(*PLOT_Y_RANGE, 501)
    xx, yy = np.meshgrid(x, y)
    field_values = f(xx, yy, q)
    magnitudes = -field_values

    if np.isfinite(f0) and f0 < 0.0:
        color_limits = np.sort([0.1 * abs(f0), 10.0 * abs(f0)])
    else:
        finite_magnitudes = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]
        color_limits = np.array([finite_magnitudes.min(), finite_magnitudes.max()])
        if color_limits[0] == color_limits[1]:
            color_limits *= np.array([0.5, 2.0])

    levels = np.geomspace(color_limits[0], color_limits[1], 21)
    colors = plt.get_cmap("seismic")(np.linspace(0.0, 1.0, 20))
    colormap = ListedColormap(colors)
    normalization = LogNorm(vmin=color_limits[0], vmax=color_limits[1], clip=True)
    color_values = np.clip(magnitudes, color_limits[0], color_limits[1])

    figure, axes = plt.subplots()
    filled = axes.contourf(
        xx,
        yy,
        color_values,
        levels=levels,
        cmap=colormap,
        norm=normalization,
    )
    if (
        np.isfinite(f0)
        and np.min(field_values) < f0 < np.max(field_values)
    ):
        axes.contour(
            xx,
            yy,
            np.ma.masked_invalid(field_values),
            levels=[f0],
            colors="black",
            linewidths=2.5,
        )
    if integration_points is not None:
        axes.scatter(
            integration_points[:, 0],
            integration_points[:, 1],
            s=18,
            marker="o",
            color="#00bcd4",
            edgecolors="black",
            linewidths=0.35,
            label="Adaptive RK45 points",
            zorder=3,
        )
    if samples is not None:
        axes.scatter(
            samples[:, 0],
            samples[:, 1],
            s=28,
            marker="x",
            color="#ff1744",
            linewidths=1.2,
            label="Uniformly sampled points",
            zorder=4,
        )
    if seed_point is not None:
        axes.scatter(
            seed_point[0],
            seed_point[1],
            s=170,
            marker="*",
            color="#ffd700",
            edgecolors="black",
            linewidths=0.8,
            label="Seed point",
            zorder=5,
        )
    if resampling_start is not None:
        axes.scatter(
            resampling_start[0],
            resampling_start[1],
            s=75,
            marker="D",
            color="#00ff66",
            edgecolors="black",
            linewidths=0.8,
            label="L1 / resampling start",
            zorder=6,
        )
    axes.set(
        xlabel="x",
        ylabel="y",
        title=f"q = {q:g}",
        xlim=PLOT_X_RANGE,
        ylim=PLOT_Y_RANGE,
    )
    axes.set_aspect("equal")
    if (
        integration_points is not None
        or samples is not None
        or seed_point is not None
        or resampling_start is not None
    ):
        axes.legend(loc="upper right")
    ticks = np.geomspace(color_limits[0], color_limits[1], 5)
    colorbar = figure.colorbar(filled, ax=axes, ticks=ticks)
    colorbar.set_ticklabels([f"{-tick:.3g}" for tick in ticks])
    colorbar.set_label("f(x, y) (logarithmic in |f|)")
    figure.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def main() -> None:
    for name, point in LAGRANGE_POINTS.items():
        print(f"{name}: x={point[0]:.8g}, y={point[1]:.8g}")
    print(f"Using F0 = f(L1) = {F0:.8g}")

    try:
        contour = trace_contour(Q, F0, SEED_X_RANGE, SEED_Y_RANGE)
    except SeedNotFoundError:
        plot_contour_map(Q, F0)
        raise

    positive_half = extract_positive_half_contour(
        contour,
        LAGRANGE_POINTS["L1"],
    )
    samples = resample_by_arclength(positive_half, N_POINTS)
    gravity_values = np.array([np.linalg.norm(gradient_f(x, y, Q)) for x, y in samples])
    output = np.column_stack((samples, gravity_values))
    np.savetxt(
        OUTPUT_FILE,
        output,
        delimiter=",",
        header="x,y,g_eff",
        comments="",
    )
    plot_contour_map(
        Q,
        F0,
        contour,
        samples,
        contour[0],
        LAGRANGE_POINTS["L1"],
    )

    half_length = np.linalg.norm(np.diff(positive_half, axis=0), axis=1).sum()
    print(f"f-value: {F0:.8g}")
    print(f"x>=0 contour length: {half_length:.8g}")
    print(
        f"Wrote {len(samples)} approximately equally spaced x>=0 points "
        f"with g_eff to {OUTPUT_FILE}"
    )
    print(f"Saved contour map to {PLOT_FILE}")


if __name__ == "__main__":
    main()
