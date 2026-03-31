import os

import numpy as np
from joblib import Parallel, delayed
from scipy import optimize as opt
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance_matrix
from scipy.stats.mstats import theilslopes

# -----------------------------------------------------------------------------
# point_clustering
# -----------------------------------------------------------------------------


def point_clustering(X, Y, max_dist):
    """This function clusters points by distance and finds the largest
    cluster. It is to be used inside fit_circle_check().

    Parameters
    ----------
    X : numpy.ndarray
        Vector containing (x) coordinates of points belonging to a tree section.
    Y : numpy.ndarray
        Vector containing (y) coordinates of points belonging to a tree section.
    max_dist : float
        Max separation among the points to be considered as members of the same
        cluster.

    Returns
    -------
    X_g : numpy.ndarray
        Vector containing the (x) coordinates of the largest cluster.
    Y_g : numpy.ndarray
        Vector containing the (y) coordinates of the largest cluster.
    """

    # Stacks 1D arrays ([X], [Y]) into a 2D array ([X, Y])
    xy_stack = np.column_stack((X, Y))

    # sch.fclusterdata outputs a vector that contains cluster ID of each point
    # (which cluster does each point belong to)
    clust_id = sch.fclusterdata(xy_stack, max_dist, criterion="distance", metric="euclidean")

    # Set of all clusters
    clust_id_unique = np.unique(clust_id)

    # For loop that iterates over each cluster ID, sums its elements and finds
    # the largest
    n_max = 0
    for c in clust_id_unique:
        # How many elements are in each cluster
        n = np.sum(clust_id == c)

        # Update largest cluster and its cardinality
        if n > n_max:
            n_max = n
            largest_cluster = c

    # X, Y coordinates of points that belong to the largest cluster
    X_g = xy_stack[clust_id == largest_cluster, 0]
    Y_g = xy_stack[clust_id == largest_cluster, 1]

    # Output: those X, Y coordinates
    return X_g, Y_g


# -------------------------------------------------------------------------------------------------------------------------------------------------------
# fit_circle
# -------------------------------------------------------------------------------------------------------------------------------------------------------


def fit_circle(X, Y):
    """This function fits points within a tree section into a circle by
    least squares minimization. It is to be used inside fit_circle_check().

    Parameters
    ----------
    X : numpy.ndarray
        Vector containing (x) coordinates of points belonging to a tree section.
    Y : numpy.ndarray
        Vector containing (y) coordinates of points belonging to a tree section.

    Returns
    -------
    circle_c : numpy.ndarray
        Matrix containing the (x, y) coordinates of the circle center.
    mean_radius : numpy.ndarray
        Vector containing the radius of each fitted circle
        (units is meters).
    """

    # Function that computes distance from each 2D point to a single point defined by (X_c, Y_c)
    # It will be used to compute the distance from each point to the circle center.
    def _calc_R(X, Y, X_c, Y_c):
        return np.sqrt((X - X_c) ** 2 + (Y - Y_c) ** 2)

    # Function that computes algebraic distance from each 2D point to some middle circle c
    # It calls calc_R (just defined above) and it is used during the least squares optimization.
    def _f_2(c, X, Y):
        R_i = _calc_R(X, Y, *c)
        return R_i - R_i.mean()

    # Initial barycenter coordinates (middle circle c center)
    X_m = X.mean()
    Y_m = Y.mean()
    barycenter = X_m, Y_m

    # Least square minimization to find the circle that best fits all
    # points within the section. 'ier' is a flag indicating whether the solution
    # was found (ier = 1, 2, 3 or 4) or not (otherwise).
    circle_c, _ = opt.leastsq(_f_2, barycenter, args=(X, Y), maxfev=2000)

    # Its radius
    radius = _calc_R(X, Y, *circle_c)
    mean_radius = radius.mean()

    # Output: - X, Y coordinates of best-fit circle center - its radius
    return circle_c, mean_radius

def fit_circle_wrlts(X, Y, h_frac=0.75, n_starts=10, max_iter=50, tol=1e-6):
    """Fit a circle using Weighted Repeated Least Trimmed Squares (WRLTS).

    Uses bi-square weighted residuals when evaluating the trimmed objective.
    This reduces the influence of inlier variation (noise / off-surface points)
    and is preferred when the inlier spread is large (Nurunnabi et al. 2018,
    Pattern Recognition).

    The bi-square weight for each point in the h-subset is

        w(e*) = (1 - e*²)²   if |e*| < 1
                0             otherwise

    where  e* = e / (6 · MAD)  and  MAD = median(|e|)  over the h-subset.

    The weighted objective minimised over starts is

        SSe = Σ_{j=1}^{h}  w(e_j*) · e_j²

    Parameters
    ----------
    X, Y      : numpy.ndarray — 2D point coordinates
    h_frac    : float — fraction of points to keep (trimming fraction),
                default 0.75 (keep 75 %, trim 25 %)
    n_starts  : int   — number of random starting subsets
    max_iter  : int   — max iterations per start
    tol       : float — convergence tolerance on the weighted trimmed objective

    Returns
    -------
    center     : numpy.ndarray — (x, y) of the circle centre
    radius     : float         — circle radius
    sigma      : float         — std dev of point-to-centre distances (on h-subset)
    sigma_mean : float         — sigma / sqrt(h)
    """
    def _bisquare_weights(e):
        mad = float(np.median(np.abs(e)))
        if mad < 1e-12:
            return np.ones(len(e))
        e_star = e / (6.0 * mad)
        return np.where(np.abs(e_star) < 1.0, (1.0 - e_star**2)**2, 0.0)

    n = len(X)
    h = max(3, int(np.round(h_frac * n)))
    rng = np.random.default_rng()

    best_obj = np.inf
    best_center = None
    best_radius = None

    for _ in range(n_starts):
        idx = rng.choice(n, size=h, replace=False)

        prev_obj = np.inf
        for _ in range(max_iter):
            center, radius = fit_circle(X[idx], Y[idx])

            residuals = np.abs(np.sqrt((X - center[0])**2 + (Y - center[1])**2) - radius)

            idx = np.argsort(residuals)[:h]

            e_h = residuals[idx]
            w_h = _bisquare_weights(e_h)
            obj = float(np.sum(w_h * e_h**2))

            if abs(prev_obj - obj) < tol:
                break
            prev_obj = obj

        if obj < best_obj:
            best_obj = obj
            best_center = center
            best_radius = radius

    return best_center, float(best_radius)

# -----------------------------------------------------------------------------
# inner_circle
# -----------------------------------------------------------------------------


def inner_circle(X, Y, X_c, Y_c, R, times_R):
    """Function that computes an internal circle inside the one fitted by
    fit_circle. This new circle is used as a validation tool and it gives
    insight on the quality of the 'fit_circle-circle'.

        - If points are closest to the inner circle, then the first fit was not
          appropriate

        - On the contrary, if points are closer to the outer circle, the
          'fit_circle-circle' is appropriate and describes well the stem diameter.

    Instead of directly computing the inner circle, it just takes a proportion
    (less than one) of the original circle radius and its center. Then, it just
    checks how many points are closest to the inner circle than to the original
    circle.

    Parameters
    ----------
    X : numpy.ndarray
        Vector containing (x) coordinates of points belonging to a tree section.
    Y : numpy.ndarray
        Vector containing (y) coordinates of points belonging to a tree section.
    X_c : numpy.ndarray
    Vector containing (x) coordinates of fitted circles.
    Y_c : numpy.ndarray
        Vector containing (y) coordinates of fitted circles.
    R : numpy.ndarray
        Vector containing the radii of the fitted circles.

    Returns
    -------
    n_points_in : numpy.ndarray
        Vector containing the number of points inside the inner circle of each
        section.
    """

    # Distance from each 2D point to the center.
    distance = np.sqrt((X - X_c) ** 2 + (Y - Y_c) ** 2)

    # Number of points closest to the inner circle, whose radius is
    # proportionate to the outer circle radius by a factor defined by 'times_R'.
    n_points_in = np.sum(distance < R * times_R)

    # Output: Number of points closest to the inner circle.
    return n_points_in


# -------------------------------------------------------------------------------------------------------------------------------------------------------
# sector_occupancy
# -------------------------------------------------------------------------------------------------------------------------------------------------------


def sector_occupancy(X, Y, X_c, Y_c, R, n_sectors, min_n_sectors, width):
    """This function provides quality measurements for the fitting of the
    circle. It divides the section in a number of sectors to check if there are
    points within them (so they are occupied). If there are not enough occupied
    sectors, the section fails the test, as it is safe to assume it has an
    abnormal, non desirable structure.

    Parameters
    ----------
    X : numpy.ndarray
        Vector containing (x) coordinates of points belonging to a tree section.
    Y : numpy.ndarray
        Vector containing (y) coordinates of points belonging to a tree section.
    X_c : numpy.ndarray
        Vector containing (x) coordinates of fitted circles.
    Y_c : numpy.ndarray
        Vector containing (y) coordinates of fitted circles.
    R : numpy.ndarray
        Vector containing the radii of the fitted circles.
    n_sectors : int
        Number of sectors in which sections will be divided.
    min_n_sectors : int
        Minimum number of occupied sectors in a section for its fitted circle
        to be considered as valid.
    width : float
        Width around the fitted circle to look for points (units is
        meters).

    Returns
    -------
    perct_occupied_sectors : float
        Percentage of occupied sectors in each section.
    enough_occupied_sectors : int
        Binary indicators whether the fitted circle is valid
        or not. 1 - valid, 0 - not valid.
    """

    # Coordinates translation.
    X_red = X - X_c
    Y_red = Y - Y_c

    # Computation of radius and angle necessary to transform cartesian coordinates
    # to polar coordinates.
    radial_coord = np.sqrt(X_red**2 + Y_red**2)  # radial coordinate
    angular_coord = np.arctan2(X_red, Y_red)  # angular coordinate. This function from numpy directly computes it.

    # Points that are close enough to the circle that will be checked.
    points_within = (radial_coord > (R - width)) * (radial_coord < (R + width))

    # Codification of points in each sector. Basically the range of angular coordinates
    # is divided in n_sector pieces and granted an integer number. Then, every
    # point is assigned the integer corresponding to the sector it belongs to.
    norm_angles = np.floor(
        angular_coord[points_within] / (2 * np.pi / n_sectors)
    )  # np.floor only keep the integer part of the division

    # Number of points in each sector.
    n_occupied_sectors = np.size(np.unique(norm_angles))

    # Percentage of occupied sectors.
    perct_occupied_sectors = n_occupied_sectors * 100 / n_sectors

    # If there are enough occupied sectors, then it is a valid section.
    enough_occupied_sectors = 0 if n_occupied_sectors < min_n_sectors else 1  # TODO(RJ): Maybe convert this to boolean

    # Output: percentage of occupied sectors | boolean indicating if it has enough
    # occupied sectors to pass the test.
    return perct_occupied_sectors, enough_occupied_sectors


# -----------------------------------------------------------------------------
# find_nearest_valid_neighbor
# -----------------------------------------------------------------------------


def find_nearest_valid_neighbor(R_tree, section_idx):
    """Find the closest section with a valid circle (R > 0) for a given tree.

    Scans downward first (lower sections tend to be more reliable), then upward.

    Parameters
    ----------
    R_tree : numpy.ndarray
        1D array of radii for all sections of one tree. R == 0 means invalid.
    section_idx : int
        Index of the section to find a neighbor for.

    Returns
    -------
    int or None
        Index of the nearest valid neighbor, or None if no valid section exists.
    """
    n = len(R_tree)
    # Search downward first
    for i in range(section_idx - 1, -1, -1):
        if R_tree[i] > 0:
            return i
    # Then upward
    for i in range(section_idx + 1, n):
        if R_tree[i] > 0:
            return i
    return None


# -----------------------------------------------------------------------------
# filter_points_by_neighbor_circle
# -----------------------------------------------------------------------------


def filter_points_by_neighbor_circle(X, Y, X_c_nb, Y_c_nb, R_nb, inflation_factor):
    """Keep only points within an inflated version of a neighbor circle.

    Parameters
    ----------
    X, Y : numpy.ndarray
        Point coordinates of the current section.
    X_c_nb, Y_c_nb : float
        Center of the nearest valid neighbor circle.
    R_nb : float
        Radius of the nearest valid neighbor circle.
    inflation_factor : float
        Multiplier for the neighbor radius defining the filter boundary.

    Returns
    -------
    X_filt, Y_filt : numpy.ndarray
        Filtered point coordinates.
    """
    dist = np.sqrt((X - X_c_nb) ** 2 + (Y - Y_c_nb) ** 2)
    mask = dist < R_nb * inflation_factor
    return X[mask], Y[mask]


# -----------------------------------------------------------------------------
# check_relative_radius
# -----------------------------------------------------------------------------


def check_relative_radius(R_current, R_neighbor, max_relative_deviation):
    """Check if a fitted radius is within acceptable deviation of a neighbor.

    Parameters
    ----------
    R_current : float
        Radius of the current section's fitted circle.
    R_neighbor : float
        Radius of the nearest valid neighbor circle.
    max_relative_deviation : float
        Maximum allowed fractional deviation (e.g. 0.5 for 50%).

    Returns
    -------
    bool
        True if the radius passes the relative check.
    """
    if R_neighbor <= 0:
        return True
    return abs(R_current - R_neighbor) / R_neighbor < max_relative_deviation


# -----------------------------------------------------------------------------
# polar_sector_approximation
# -----------------------------------------------------------------------------


def polar_sector_approximation(X, Y, X_c_ref, Y_c_ref, n_sectors):
    """Approximate a circle using polar-coordinate sector averaging with
    center refinement.

    Converts points to polar coordinates around a reference center, bins them
    into angular sectors, and averages the radius per sector.  When at least
    3 sectors are occupied a linear cosine model is fitted to the per-sector
    radii to estimate an offset (dx, dy) from the reference center:

        r(θ) ≈ R_true + dx·cos(θ) + dy·sin(θ)

    This corrects systematic bias when the true stem center differs from the
    neighbor center.  With fewer than 3 occupied sectors the reference center
    is kept as-is (the system is underdetermined).

    Parameters
    ----------
    X, Y : numpy.ndarray
        Point coordinates.
    X_c_ref, Y_c_ref : float
        Reference center (from nearest valid neighbor circle).
    n_sectors : int
        Number of angular sectors.

    Returns
    -------
    X_c : float
        Refined circle center X.
    Y_c : float
        Refined circle center Y.
    R_approx : float
        Approximated radius.
    n_occupied : int
        Number of sectors that contained at least one point.
    """
    dX = X - X_c_ref
    dY = Y - Y_c_ref
    r = np.sqrt(dX ** 2 + dY ** 2)
    theta = np.arctan2(dY, dX)  # range [-pi, pi]

    # Bin into sectors
    sector_size = 2 * np.pi / n_sectors
    # Shift theta to [0, 2*pi) for clean binning
    theta_shifted = theta + np.pi
    sector_ids = np.floor(theta_shifted / sector_size).astype(int)
    sector_ids = np.clip(sector_ids, 0, n_sectors - 1)

    sector_angles = []
    sector_radii = []
    for s in range(n_sectors):
        mask = sector_ids == s
        if np.any(mask):
            sector_radii.append(np.mean(r[mask]))
            sector_angles.append(np.mean(theta[mask]))

    n_occupied = len(sector_radii)
    if n_occupied == 0:
        return X_c_ref, Y_c_ref, 0.0, 0

    sector_radii = np.asarray(sector_radii)
    sector_angles = np.asarray(sector_angles)

    if n_occupied >= 3:
        # Fit r(θ) = R + dx·cos(θ) + dy·sin(θ) via least squares
        A = np.column_stack([
            np.ones(n_occupied),
            np.cos(sector_angles),
            np.sin(sector_angles),
        ])
        # lstsq gives [R_true, dx, dy]
        params, _, _, _ = np.linalg.lstsq(A, sector_radii, rcond=None)
        R_approx = float(params[0])
        X_c = X_c_ref + float(params[1])
        Y_c = Y_c_ref + float(params[2])
    else:
        # Not enough sectors to determine center offset — keep reference
        R_approx = float(np.mean(sector_radii))
        X_c = X_c_ref
        Y_c = Y_c_ref

    return X_c, Y_c, R_approx, n_occupied


# -----------------------------------------------------------------------------
# run_quality_checks
# -----------------------------------------------------------------------------


def run_quality_checks(
    X, Y, X_c, Y_c, R,
    times_R, threshold, R_min, R_max,
    n_sectors, min_n_sectors, width,
    R_neighbor=None, max_relative_deviation=0.05,
):
    """Consolidated quality checks for a fitted circle.

    Combines inner circle test, sector occupancy, radius range, and
    optional relative radius check into a single pass/fail decision.

    Parameters
    ----------
    X, Y : numpy.ndarray
        Point coordinates.
    X_c, Y_c : float
        Fitted circle center.
    R : float
        Fitted circle radius.
    times_R : float
        Inner/outer circle ratio for inner_circle check.
    threshold : int
        Max points in inner circle.
    R_min, R_max : float
        Valid radius range.
    n_sectors : int
        Number of sectors for occupancy check.
    min_n_sectors : int
        Minimum occupied sectors required.
    width : float
        Width around circle to search for points.
    R_neighbor : float or None
        Radius of nearest valid neighbor. None skips relative check.
    max_relative_deviation : float
        Max fractional deviation from neighbor radius.

    Returns
    -------
    passed : bool
        True if all checks pass.
    sector_perct : float
        Percentage of occupied sectors.
    n_points_in : int
        Number of points in the inner circle.
    """
    n_points_in = inner_circle(X, Y, X_c, Y_c, R, times_R)
    sector_perct, enough_sectors = sector_occupancy(
        X, Y, X_c, Y_c, R, n_sectors, min_n_sectors, width
    )

    inner_ok = n_points_in <= threshold
    radius_ok = R_min <= R <= R_max
    sectors_ok = enough_sectors == 1

    if R_neighbor is not None:
        relative_ok = check_relative_radius(R, R_neighbor, max_relative_deviation)
    else:
        relative_ok = True

    passed = inner_ok and radius_ok and sectors_ok and relative_ok
    return passed, sector_perct, n_points_in


# -----------------------------------------------------------------------------
# fit_circle_check
# -----------------------------------------------------------------------------


def fit_circle_check(
    X,
    Y,
    times_R,
    threshold,
    R_min,
    R_max,
    n_points_section,
    n_sectors,
    min_n_sectors,
    width,
    use_wrlts=False,
    R_neighbor=None,
    max_relative_deviation=0.05,
):
    """Fit a circle to a tree section and run quality checks.

    This is a single-pass fit-and-check function. Multi-pass orchestration
    (neighbor filtering, clustering, polar approximation) is handled by
    compute_sections.

    Parameters
    ----------
    X : numpy.ndarray
        Vector containing (x) coordinates of points belonging to a tree section.
    Y : numpy.ndarray
        Vector containing (y) coordinates of points belonging to a tree section.
    times_R : float
        Ratio of radius between outer circle and inner circle.
    threshold : float
        Maximum number of points in inner circle for a fitted circle to be valid.
    R_min : float
        Minimum radius that a fitted circle must have to be valid.
    R_max : float
        Maximum radius that a fitted circle must have to be valid.
    n_points_section : int
        Minimum points within a section for its fitted circle to be valid.
    n_sectors : int
        Number of sectors in which sections will be divided.
    min_n_sectors : int
        Minimum number of occupied sectors in a section for its fitted circle
        to be considered as valid.
    width : float
        Width around the fitted circle to look for points (units is meters).
    use_wrlts : bool
        If True, use WRLTS fitting instead of standard LSM.
    R_neighbor : float or None
        Radius of nearest valid neighbor for relative radius check. None skips.
    max_relative_deviation : float
        Maximum allowed fractional deviation from neighbor radius.

    Returns
    -------
    X_c : float
        X coordinate of the center of the best-fit circle (0 if invalid).
    Y_c : float
        Y coordinate of the center of the best-fit circle (0 if invalid).
    R : float
        Best-fit circle radius (0 if invalid).
    passed : bool
        Whether the circle passed all quality checks.
    sector_perct : float
        Percentage of occupied sectors.
    n_points_in : int
        Number of points in the inner circle.
    """
    if X.size <= n_points_section:
        return 0, 0, 0, False, 0, 0

    if use_wrlts:
        (circle_center, R) = fit_circle_wrlts(X=X, Y=Y)
    else:
        (circle_center, R) = fit_circle(X=X, Y=Y)
    X_c = circle_center[0]
    Y_c = circle_center[1]

    passed, sector_perct, n_points_in = run_quality_checks(
        X, Y, X_c, Y_c, R,
        times_R, threshold, R_min, R_max,
        n_sectors, min_n_sectors, width,
        R_neighbor=R_neighbor,
        max_relative_deviation=max_relative_deviation,
    )

    return X_c, Y_c, R, passed, sector_perct, n_points_in


# -----------------------------------------------------------------------------
# compute_sections
# -----------------------------------------------------------------------------


def _process_single_tree(
    tree_points,
    sections,
    section_width,
    check_params,
    inflation_factor,
    max_relative_deviation,
    n_points_section,
    R_min,
    R_max,
    max_dist,
    n_sectors,
    min_n_sectors,
    width,
    X_field,
    Y_field,
    Z0_field,
):
    """Process all sections for a single tree.

    Runs the 3-pass circle fitting pipeline (LSM, WRLTS, Polar) on the
    points of one tree and returns per-section result arrays.

    Parameters
    ----------
    tree_points : numpy.ndarray
        Point cloud for a single tree (already filtered by tree ID).
    sections : numpy.ndarray
        Height values at which sections are computed.
    section_width : float
        Half-width of the height band for section extraction.
    check_params : dict
        Quality check parameters passed to ``fit_circle_check``.
    inflation_factor : float
        Radius multiplier for neighbor-based point prefiltering.
    max_relative_deviation : float
        Maximum fractional deviation from neighbor radius.
    n_points_section : int
        Minimum number of points required per section.
    R_min : float
        Minimum acceptable radius.
    R_max : float
        Maximum acceptable radius.
    max_dist : float
        Maximum distance for point clustering.
    n_sectors : int
        Number of angular sectors for occupancy check.
    min_n_sectors : int
        Minimum occupied sectors required.
    width : float
        Search radius around circle for sector occupancy.
    X_field : int
        Column index for X coordinate.
    Y_field : int
        Column index for Y coordinate.
    Z0_field : int
        Column index for normalized Z coordinate.

    Returns
    -------
    tuple of numpy.ndarray
        (X_c, Y_c, R, check_circle, pass_method, sector_perct, n_points_in),
        each of shape ``(n_sections,)``.
    """
    n_sections = sections.size

    # Per-tree output arrays
    X_c = np.zeros(n_sections, dtype=np.float32)
    Y_c = np.zeros(n_sections, dtype=np.float32)
    R = np.zeros(n_sections, dtype=np.float32)
    check_circle = np.zeros(n_sections, dtype=np.int8)
    pass_method = np.zeros(n_sections, dtype=np.int8)
    sector_perct = np.zeros(n_sections, dtype=np.float32)
    n_points_in = np.zeros(n_sections, dtype=np.int32)

    # --- Extract all section point sets once ---
    section_points = []
    for b in sections:
        mask = (tree_points[:, Z0_field] >= b - section_width / 2) & (
            tree_points[:, Z0_field] < b + section_width / 2
        )
        section_points.append((tree_points[mask, X_field], tree_points[mask, Y_field]))

    # Track which sections need further passes
    needs_pass2 = set()

    # ================================================================
    # PASS 1: Quick LSM
    # ================================================================
    for s_idx, (X_s, Y_s) in enumerate(section_points):
        xc, yc, r, passed, s_pct, n_in = fit_circle_check(
            X_s, Y_s, **check_params
        )
        if passed:
            X_c[s_idx] = xc
            Y_c[s_idx] = yc
            R[s_idx] = r
            check_circle[s_idx] = 1
            pass_method[s_idx] = 1
            sector_perct[s_idx] = s_pct
            n_points_in[s_idx] = n_in
        else:
            needs_pass2.add(s_idx)
            if X_s.size <= n_points_section:
                check_circle[s_idx] = 2
            else:
                check_circle[s_idx] = 1

    # Retroactive relative radius check on Pass 1 results.
    for s_idx in range(n_sections):
        if s_idx in needs_pass2:
            continue
        if R[s_idx] == 0:
            continue
        nb_idx = find_nearest_valid_neighbor(R, s_idx)
        if nb_idx is not None and not check_relative_radius(
            R[s_idx], R[nb_idx], max_relative_deviation
        ):
            needs_pass2.add(s_idx)
            X_c[s_idx] = 0
            Y_c[s_idx] = 0
            R[s_idx] = 0
            pass_method[s_idx] = 0
            sector_perct[s_idx] = 0
            n_points_in[s_idx] = 0

    # ================================================================
    # PASS 2: Filtered WRLTS
    # ================================================================
    needs_pass3 = set()
    for s_idx in needs_pass2:
        X_s, Y_s = section_points[s_idx]
        if X_s.size <= n_points_section:
            needs_pass3.add(s_idx)
            continue

        nb_idx = find_nearest_valid_neighbor(R, s_idx)
        if nb_idx is None:
            needs_pass3.add(s_idx)
            continue

        X_f, Y_f = filter_points_by_neighbor_circle(
            X_s, Y_s,
            X_c[nb_idx], Y_c[nb_idx], R[nb_idx],
            inflation_factor,
        )

        if X_f.size > n_points_section:
            X_f, Y_f = point_clustering(X_f, Y_f, max_dist)

        if X_f.size <= n_points_section:
            needs_pass3.add(s_idx)
            continue

        xc, yc, r, passed, s_pct, n_in = fit_circle_check(
            X_f, Y_f,
            **check_params,
            use_wrlts=True,
            R_neighbor=R[nb_idx],
            max_relative_deviation=max_relative_deviation,
        )

        if passed:
            X_c[s_idx] = xc
            Y_c[s_idx] = yc
            R[s_idx] = r
            check_circle[s_idx] = 1
            pass_method[s_idx] = 2
            sector_perct[s_idx] = s_pct
            n_points_in[s_idx] = n_in
        else:
            needs_pass3.add(s_idx)

    # ================================================================
    # PASS 3: Polar Sector Approximation
    # ================================================================
    for s_idx in needs_pass3:
        X_s, Y_s = section_points[s_idx]
        if X_s.size <= n_points_section:
            continue

        nb_idx = find_nearest_valid_neighbor(R, s_idx)
        if nb_idx is None:
            continue

        X_f, Y_f = filter_points_by_neighbor_circle(
            X_s, Y_s,
            X_c[nb_idx], Y_c[nb_idx], R[nb_idx],
            inflation_factor,
        )

        if X_f.size <= n_points_section:
            continue

        xc_approx, yc_approx, r_approx, n_occupied = polar_sector_approximation(
            X_f, Y_f,
            X_c[nb_idx], Y_c[nb_idx],
            n_sectors,
        )

        if r_approx <= 0 or r_approx < R_min or r_approx > R_max:
            continue

        baseline_perct, _ = sector_occupancy(
            X_f, Y_f,
            X_c[nb_idx], Y_c[nb_idx], R[nb_idx],
            n_sectors, min_n_sectors, width,
        )
        new_perct = n_occupied * 100.0 / n_sectors

        if new_perct > baseline_perct:
            X_c[s_idx] = xc_approx
            Y_c[s_idx] = yc_approx
            R[s_idx] = r_approx
            check_circle[s_idx] = 1
            pass_method[s_idx] = 3
            sector_perct[s_idx] = new_perct
            n_points_in[s_idx] = 0

    return (X_c, Y_c, R, check_circle, pass_method, sector_perct, n_points_in)


def compute_sections(
    stems,
    sections,
    section_width=0.02,
    times_R=0.5,
    threshold=5,
    R_min=0.03,
    R_max=0.5,
    max_dist=0.02,
    n_points_section=80,
    n_sectors=16,
    min_n_sectors=9,
    width=2,
    inflation_factor=1.5,
    max_relative_deviation=0.05,
    X_field=0,
    Y_field=1,
    Z0_field=3,
    tree_id_field=4,
    progress_hook=None,
    n_workers=None,
):
    """Compute stem diameters at given sections using a multi-pass circle
    fitting pipeline.

    For each tree, three passes are applied sequentially:

    Pass 1 - Quick LSM: Fit circles using least squares. Validate with
    quality checks including a retroactive relative radius check against
    neighboring sections.

    Pass 2 - Filtered WRLTS: For sections that failed Pass 1, prefilter
    points using the nearest valid neighbor circle, cluster, then fit
    using Weighted Repeated Least Trimmed Squares.

    Pass 3 - Polar sector approximation: For remaining failures, use the
    neighbor circle center to convert points to polar coordinates, average
    radius per angular sector, and derive a circle approximation.

    Parameters
    ----------
    stems : numpy.ndarray
        Point cloud containing the individualized trees. It is expected to
        have X, Y, Z0 and tree_ID fields.
    sections : numpy.ndarray
        Matrix containing a range of height values at which sections will be
        computed.
    section_width : float
        Points within this distance from any `sections` value will be considered
        as belonging to said section (units is meters). Defaults to 0.02.
    times_R : float
        Refer to fit_circle_check. Defaults to 0.5.
    threshold : float
        Refer to fit_circle_check. Defaults to 5.
    R_min : float
        Refer to fit_circle_check. Defaults to 0.03.
    R_max : float
        Refer to fit_circle_check. Defaults to 0.5.
    max_dist : float
        Refer to fit_circle_check. Defaults to 0.02.
    n_points_section : int
        Refer to fit_circle_check. Defaults to 80.
    n_sectors : int
        Refer to fit_circle_check. Defaults to 16.
    min_n_sectors : int
        Refer to fit_circle_check. Defaults to 9.
    width : float
        Refer to fit_circle_check. Defaults to 2.0.
    inflation_factor : float
        Multiplier for the neighbor circle radius when prefiltering points
        in Passes 2 and 3. Defaults to 1.5.
    max_relative_deviation : float
        Maximum allowed fractional deviation of a section radius from its
        nearest valid neighbor. Defaults to 0.5.
    X_field : int
        Index at which (x) coordinate is stored. Defaults to 0.
    Y_field : int
        Index at which (y) coordinate is stored. Defaults to 1.
    Z0_field : int
        Index at which (z0) coordinate is stored. Defaults to 3.
    tree_id_field : int
        Index at which cluster ID is stored. Defaults to 4.
    progress_hook : callable, optional
        A hook that take two int, the first is the current number of iteration
        and the second is the targeted number iteration. Defaults to None.
    n_workers : int, optional
        Number of parallel worker processes for per-tree processing.
        ``None`` auto-detects based on CPU count, ``1`` runs
        sequentially. Uses joblib with the loky backend.
        Defaults to None.

    Returns
    -------
    X_c : numpy.ndarray
        Matrix containing (x) coordinates of the center of the best-fit circles.
    Y_c : numpy.ndarray
        Matrix containing (y) coordinates of the center of the best-fit circles.
    R : numpy.ndarray
        Vector containing best-fit circle radii.
    check_circle : numpy.ndarray
        Matrix indicating review status (0=unchecked, 1=checked, 2=too few points).
    pass_method : numpy.ndarray
        Matrix indicating which pass produced the result
        (0=invalid, 1=Pass 1 LSM, 2=Pass 2 WRLTS, 3=Pass 3 polar).
    sector_perct : numpy.ndarray
        Matrix containing the percentage of occupied sectors.
    n_points_in : numpy.ndarray
        Matrix containing the number of points in the inner circles.
    """
    trees = np.unique(stems[:, tree_id_field])
    n_trees = trees.size
    n_sections = sections.size

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, n_trees)

    # Quality check parameters bundled for convenience
    check_params = dict(
        times_R=times_R,
        threshold=threshold,
        R_min=R_min,
        R_max=R_max,
        n_points_section=n_points_section,
        n_sectors=n_sectors,
        min_n_sectors=min_n_sectors,
        width=width,
    )

    # Pre-split stems by tree ID (O(N log N) once instead of O(N) per tree)
    sort_idx = np.argsort(stems[:, tree_id_field])
    stems_sorted = stems[sort_idx]
    split_points = np.searchsorted(stems_sorted[:, tree_id_field], trees, side="right")
    tree_point_arrays = np.split(stems_sorted, split_points[:-1])
    del stems_sorted, sort_idx

    if progress_hook is not None:
        progress_hook(0, n_trees)

    # joblib loky backend: spawns real processes (bypasses GIL),
    # memory-maps large numpy arrays, works from PyQt QThreads.
    all_results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_process_single_tree)(
            tree_points,
            sections,
            section_width,
            check_params,
            inflation_factor,
            max_relative_deviation,
            n_points_section,
            R_min,
            R_max,
            max_dist,
            n_sectors,
            min_n_sectors,
            width,
            X_field,
            Y_field,
            Z0_field,
        )
        for tree_points in tree_point_arrays
    )

    # Assemble per-tree results into output matrices
    X_c = np.zeros((n_trees, n_sections), dtype=np.float32)
    Y_c = np.zeros((n_trees, n_sections), dtype=np.float32)
    R = np.zeros((n_trees, n_sections), dtype=np.float32)
    check_circle = np.zeros((n_trees, n_sections), dtype=np.int8)
    pass_method = np.zeros((n_trees, n_sections), dtype=np.int8)
    sector_perct = np.zeros((n_trees, n_sections), dtype=np.float32)
    n_points_in = np.zeros((n_trees, n_sections), dtype=np.int32)

    for tree_idx, result in enumerate(all_results):
        X_c[tree_idx, :] = result[0]
        Y_c[tree_idx, :] = result[1]
        R[tree_idx, :] = result[2]
        check_circle[tree_idx, :] = result[3]
        pass_method[tree_idx, :] = result[4]
        sector_perct[tree_idx, :] = result[5]
        n_points_in[tree_idx, :] = result[6]
        if progress_hook is not None:
            progress_hook(tree_idx + 1, n_trees)

    return (X_c, Y_c, R, check_circle, pass_method, sector_perct, n_points_in)


# -----------------------------------------------------------------------------
# tilt_detection
# -----------------------------------------------------------------------------


def tilt_detection(X_tree, Y_tree, radius, sections, Z_field=2, w_1=3.0, w_2=1.0):
    """This function finds outlier tilting values among sections within a tree
    and assigns a score to the sections based on those outliers. Two kinds of
    outliers are considered.

        - Absolute outliers are obtained from the sum of the deviations from
          every section center to all axes within a tree (the most tilted sections
          relative to all axes)

        - Relative outliers are obtained from the deviations of other section
          centers from a certain axis, within a tree (the most tilted sections
          relative to a certain axis)

    The 'outlier score' consists on a weighted sum of the absolute tilting value
    and the relative tilting value.

    Parameters
    ----------
    X_tree : numpy.ndarray
        Matrix containing (x) coordinates of the center of the sections.
    Y_tree : numpy.ndarray
        Matrix containing (y) coordinates of the center of the sections.
    radius : numpy.ndarray
        Vector containing section radii.
    sections : numpy.ndarray
        Vector containing the height of the section associated to each section.
    Z_field : int
        Index at which (z) coordinate is stored. Defaults to 2.
    w_1 : float
        Weight of absolute deviation. Defaults to 3.0.
    w_2 : float
        Weight of relative deviation. Defaults to 1.0.

    Returns
    -------
    outlier_prob : numpy.ndarray
        Vector containing the 'outlier probability' of each section.
    """

    # This function simply defines 1st and 3rd quartile of a vector and separates
    # values that are outside the interquartile range defined by these. Those
    # are the candidates to be outliers. This filtering may be done either
    # directly from the interquartile range, or from a certain distance from it,
    # thanks to 'n_range' parameter. Its default value is 1.5.

    def _outlier_vector(vector, lower_q=0.25, upper_q=0.75, n_range=1.5):
        q1, q3 = np.quantile(vector, [lower_q, upper_q])  # First quartile and Third quartile
        iqr = q3 - q1  # Interquartile range

        lower_bound = q1 - iqr * n_range  # Lower bound of filter. If n_range = 0 -> lower_bound = q1
        upper_bound = q3 + iqr * n_range  # Upper bound of filter. If n_range = 0 -> upper_bound = q3

        # return the outlier vector.
        return ((vector < lower_bound) | (vector > upper_bound)).astype(int)

    # Empty matrix that will store the probabilities of a section to be invalid
    outlier_prob = np.zeros_like(X_tree)

    # First loop: iterates over each tree
    for i in range(X_tree.shape[0]):
        # If there is, at least, 1 circle with positive radius in a tree, then
        # proceed (invalid circles are stored with a radius value of 0)
        if np.sum(radius[i, :]) > 0:
            # Filtering sections within a tree that have valid circles (non-zero radius).
            valid_radius = radius[i, :] > 0
            num_valid_sections = np.size(sections[valid_radius])
            # Weights associated to each section. They are computed in a way
            # that the final value of outliers sums up to 1 as maximum.
            abs_outlier_w = w_1 / (num_valid_sections * w_2 + w_1)
            rel_outlier_w = w_2 / (num_valid_sections * w_2 + w_1)

            # Vertical distance matrix among all sections (among their centers)
            # Empty matrix to store heights of each section
            heights = np.zeros((num_valid_sections, Z_field))
            #  Height (Z value) of each section
            heights[:, 0] = np.transpose(sections[valid_radius])
            # Vertical distance matrix
            z_dist_matrix = distance_matrix(heights, heights)

            # Horizontal distance matrix among all sections (among their centers)
            # Store X, Y coordinates of each section
            c_coord = np.column_stack((X_tree[i][valid_radius], Y_tree[i][valid_radius]))
            # Horizontal distance matrix
            xy_dist_matrix = distance_matrix(c_coord, c_coord)

            # Tilting measured from every vertical within a tree: All verticals
            # obtained from the set of sections within a tree. For instance, if
            # there are 10 sections, there are 10 tilting values for each section.
            tilt_matrix = np.degrees(np.arctan(xy_dist_matrix / z_dist_matrix))

            # Summation of tilting values from each center.
            tilt_sum = np.nansum(tilt_matrix, axis=0)

            # Outliers within previous vector (too low / too high tilting values).
            # These are abnormals tilting values from ANY axis.
            outlier_prob[i][valid_radius] = _outlier_vector(tilt_sum) * abs_outlier_w

            # Second loop: iterates over each section (within a single tree).
            for j in range(np.size(sections[valid_radius])):
                # Search for abnormals tilting values from a CERTAIN axis.
                tilt_matrix[j, j] = np.quantile(tilt_matrix[j, ~j], 0.5)
                # Storing those values.
                rel_outlier = _outlier_vector(tilt_matrix[j]) * rel_outlier_w
                # Sum of absolute outlier value and relative outlier values
                outlier_prob[i][valid_radius] += rel_outlier

    return outlier_prob


# -----------------------------------------------------------------------------
# filter_radius_outliers
# -----------------------------------------------------------------------------


def filter_radius_outliers(
    R,
    sections,
    sector_perct,
    mad_multiplier=3.0,
    min_residual_threshold=0.005,
    min_valid_sections=3,
    max_slope_ci=0.2,
):
    """Filter section radii that deviate from a robust linear taper model.

    For each tree, a Theil-Sen estimator is used to check taper consistency
    via its slope confidence interval. If the CI width exceeds
    ``max_slope_ci``, the entire tree is invalidated. Otherwise, an
    occupancy-weighted linear fit determines the taper line, and individual
    sections whose residuals exceed a MAD-based threshold are set to 0.

    Parameters
    ----------
    R : numpy.ndarray
        Matrix of shape (n_trees, n_sections) with fitted circle radii.
        Sections with R == 0 are considered invalid.
    sections : numpy.ndarray
        Vector of section heights with shape (n_sections,).
    sector_perct : numpy.ndarray
        Matrix of shape (n_trees, n_sections) with sector occupancy
        percentages (0-100). Used as weights for the linear taper fit so
        that high-occupancy sections have more influence on the line.
    mad_multiplier : float
        Multiplier for the Median Absolute Deviation used as the outlier
        threshold. Higher values are more permissive. Defaults to 3.0.
    min_residual_threshold : float
        Floor for the residual threshold in meters. Prevents the threshold
        from collapsing to zero when the fit is nearly perfect. Defaults to
        0.005.
    min_valid_sections : int
        Minimum number of valid sections required to attempt taper fitting.
        Trees with fewer valid sections are left unchanged. Defaults to 3.
    max_slope_ci : float
        Maximum allowed width of the Theil-Sen confidence interval on the
        slope. Trees whose CI width exceeds this value have all sections
        invalidated. Lower values are stricter. Defaults to 0.2.

    Returns
    -------
    R_filtered : numpy.ndarray
        Copy of R with outlier radii set to 0.
    """
    R_filtered = R.copy()

    for i in range(R.shape[0]):
        valid_idx = np.where(R_filtered[i, :] > 0)[0]

        if len(valid_idx) < min_valid_sections:
            continue

        h = sections[valid_idx]
        r = R_filtered[i, valid_idx]

        # Theil-Sen for robust CI check on slope consistency
        theilslope, _, low_slope, high_slope = theilslopes(r, h)

        # Reject entire tree if the taper slope is too uncertain or positive (non-tapering)
        ci_width = high_slope - low_slope
        if ci_width > max_slope_ci or theilslope > 0:
            R_filtered[i, :] = 0
            continue

        # Occupancy-weighted linear fit for the taper line
        occ = sector_perct[i, valid_idx]
        coeffs = np.polyfit(h, r, 1, w=occ)
        slope, intercept = coeffs[0], coeffs[1]

        predicted = slope * h + intercept
        residuals = r - predicted

        mad = np.median(np.abs(residuals))
        threshold = max(mad_multiplier * mad, min_residual_threshold)

        outlier_mask = np.abs(residuals) > threshold
        R_filtered[i, valid_idx[outlier_mask]] = 0

    return R_filtered


# -----------------------------------------------------------------------------
# filter_occupancy_outliers
# -----------------------------------------------------------------------------


def filter_occupancy_outliers(
    R, sector_perct, mad_multiplier=3.0, min_threshold_perct=5.0, min_valid_sections=3
):
    """Filter sections with abnormally low sector occupancy for their tree.

    For each tree, the median sector occupancy of valid sections is computed.
    Sections whose occupancy falls below ``median - max(mad_multiplier * MAD,
    min_threshold_perct)`` are treated as outliers and their radii are set to 0.
    This is a one-sided filter: only unusually *low* occupancy is penalised.

    Parameters
    ----------
    R : numpy.ndarray
        Matrix of shape (n_trees, n_sections) with fitted circle radii.
        Sections with R == 0 are considered invalid.
    sector_perct : numpy.ndarray
        Matrix of shape (n_trees, n_sections) with sector occupancy
        percentages (0-100).
    mad_multiplier : float
        Multiplier for the Median Absolute Deviation used as the outlier
        threshold. Higher values are more permissive. Defaults to 3.0.
    min_threshold_perct : float
        Floor for the MAD-based deviation in percentage points. Prevents the
        threshold from being too tight when occupancy is very consistent.
        Defaults to 5.0.
    min_valid_sections : int
        Minimum number of valid sections required to attempt filtering. Trees
        with fewer valid sections are left unchanged. Defaults to 3.

    Returns
    -------
    R_filtered : numpy.ndarray
        Copy of R with outlier radii set to 0.
    """
    R_filtered = R.copy()

    for i in range(R.shape[0]):
        valid_idx = np.where(R_filtered[i, :] > 0)[0]

        if len(valid_idx) < min_valid_sections:
            continue

        occ = sector_perct[i, valid_idx]
        med = np.median(occ)
        mad = np.median(np.abs(occ - med))
        lower_bound = med - max(mad_multiplier * mad, min_threshold_perct)

        outlier_mask = occ < lower_bound
        R_filtered[i, valid_idx[outlier_mask]] = 0

    return R_filtered


# -----------------------------------------------------------------------------
# compute_tree_quality
# -----------------------------------------------------------------------------


def compute_tree_quality(
    R: np.ndarray,
    quality_mask: np.ndarray,
    outliers: np.ndarray,
    sector_perct: np.ndarray,
    sections: np.ndarray,
) -> np.ndarray:
    """Compute a per-tree quality score (0-1) from six weighted sub-scores.

    Parameters
    ----------
    R : np.ndarray
        (n_trees, n_sections) fitted circle radii after filtering. 0 = not fitted.
    quality_mask : np.ndarray
        Boolean (n_trees, n_sections). True = section passes all quality checks.
    outliers : np.ndarray
        (n_trees, n_sections) outlier probabilities in [0, 1] from tilt_detection.
    sector_perct : np.ndarray
        (n_trees, n_sections) sector occupancy percentages in [0, 100].
    sections : np.ndarray
        (n_sections,) normalised section heights (Z0) in metres.

    Returns
    -------
    np.ndarray
        float32 (n_trees,) with values in [0, 1].

    Notes
    -----
    Sub-scores and weights:

    1. Section pass rate        (0.25) – fraction of sections passing quality_mask
    2. Mean sector occupancy    (0.20) – mean sector_perct / 100
    3. Circle fit success rate  (0.15) – fraction of sections with R > 0
    4. Taper consistency        (0.15) – Theil-Sen linear fit of diameter vs height
    5. Low outlier probability  (0.15) – 1 - mean(outliers)
    6. Sector occ. consistency  (0.10) – 1 - coefficient of variation of sector_perct
    """
    from scipy.stats import theilslopes

    n_trees, n_sections_total = R.shape
    scores = np.zeros(n_trees, dtype=np.float32)

    for i in range(n_trees):
        dia_row = R[i] * 2.0
        qo_row = quality_mask[i].astype(float)
        q1_row = outliers[i].astype(float)
        q2_row = sector_perct[i].astype(float)

        # 1. Section pass rate
        s_pass_rate = float(np.nansum(qo_row == 1.0)) / n_sections_total

        # 2. Mean sector occupancy (0-100 -> 0-1)
        q2_valid = q2_row[~np.isnan(q2_row)]
        s_sector = float(np.mean(q2_valid) / 100.0) if len(q2_valid) > 0 else 0.0

        # 3. Sector occupancy consistency (1 - coefficient of variation)
        if len(q2_valid) >= 2 and np.mean(q2_valid) > 0:
            cv = float(np.std(q2_valid) / np.mean(q2_valid))
            s_sector_consistency = 1.0 - min(cv, 1.0)
        else:
            s_sector_consistency = 0.0

        # 4. Circle fit success rate
        valid_dia = ~np.isnan(dia_row) & (dia_row > 0)
        s_fit_rate = float(np.sum(valid_dia)) / n_sections_total

        # 5. Low outlier probability (invert: lower = higher quality)
        q1_valid = q1_row[~np.isnan(q1_row)]
        s_inlier = float(1.0 - np.mean(q1_valid)) if len(q1_valid) > 0 else 0.0

        # 6. Taper consistency (Theil-Sen fit of diameter vs height)
        n_heights = min(len(dia_row), len(sections))
        heights = sections[:n_heights].astype(float)
        valid_mask_taper = valid_dia[:n_heights] & ~np.isnan(heights)
        if np.sum(valid_mask_taper) >= 3:
            h = heights[valid_mask_taper]
            d = dia_row[:n_heights][valid_mask_taper]
            slope, intercept, _, _ = theilslopes(d, h)
            predicted = np.maximum(intercept + slope * h, 1e-6)
            rel_residuals = (d - predicted) / predicted
            rms = float(np.sqrt(np.mean(rel_residuals ** 2)))
            s_taper = 1.0 - min(rms, 1.0)
        else:
            s_taper = 0.5

        scores[i] = round(
            0.25 * s_pass_rate
            + 0.20 * s_sector
            + 0.15 * s_fit_rate
            + 0.15 * s_taper
            + 0.15 * s_inlier
            + 0.10 * s_sector_consistency,
            4,
        )

    return scores


# -----------------------------------------------------------------------------
# tree_locator
# --------------------------------------------------------------------------


def tree_locator(
    sections,
    X_c,
    Y_c,
    tree_vector,
    sector_perct,
    R,
    outliers,
    X_field=0,
    Y_field=1,
    Z_field=2,
):
    """This function generates points that locate the individualized trees and
    computes their DBH (diameter at breast height). It uses all the quality
    measurements defined in previous functions to check whether the DBH should
    be computed or not and to check which point should be used as the tree locator.

    The tree locators are then saved in a LAS file. Each tree locator corresponds
    on a one-to-one basis to the individualized trees.

    Parameters
    ----------
    sections : numpy.ndarray
        Vector containing section heights (normalized heights).
    X_c : numpy.ndarray
        Matrix containing (x) coordinates of the center of the sections.
    Y_c : numpy.ndarray
        Matrix containing (y) coordinates of the center of the sections.
    tree_vector : numpy.ndarray
        detected_trees output from individualize_trees.
    sector_perct : numpy.ndarray
        Matrix containing the percentage of occupied sectors.
    R : numpy.ndarray
        Vector containing section radii.
    outliers : numpy.ndarray
        Vector containing the 'outlier probability' of each section.
    X_field : int
        Index at which (x) coordinate is stored. Defaults to 0.
    Y_field : int
        Index at which (y) coordinate is stored. Defaults to 1.
    Z_field : int
        Index at which (z) coordinate is stored. Defaults to 2.

    Returns
    -------
    dbh_values : numpy.ndarray
        Vector containing DBH values.
    tree_locations : numpy.ndarray
        Matrix containing (x, y, z) coordinates of each tree locator.
    """
    DBH = 1.3  # Breast height constant

    # Number of trees
    n_trees = X_c.shape[0]
    # Empty vector to be filled with tree locators
    tree_locations = np.zeros((n_trees, 3))
    # Empty vector to be filled with DBH values.
    dbh_values = np.zeros((n_trees, 1))

    def _axis_location(index):
        """Given an index compute tree location from axis"""
        vector = -tree_vector[index, 1:4] if tree_vector[index, 3] < 0 else tree_vector[index, 1:4]
        dbh_values[index] = 0
        # Compute the height difference between centroid and BH
        diff_height = DBH - tree_vector[index, 6] + tree_vector[index, 7]
        # Compute the distance between centroid and axis point at BH.
        dist_centroid_dbh = diff_height / np.cos(np.radians(tree_vector[index, 8]))
        # Compute coordinates of axis point at BH.
        tree_locations[i, :] = vector * dist_centroid_dbh + tree_vector[i, 4:7]

    def _dbh_location(index, which_dbh):
        """Given an index, compute the tree location from the computed DBH"""
        dbh_values[index] = R[index, which_dbh] * 2
        # Their centers are averaged and we keep that value
        tree_locations[index, X_field] = X_c[index, which_dbh]
        tree_locations[index, Y_field] = Y_c[index, which_dbh]
        # Original height is obtained
        tree_locations[index, Z_field] = tree_vector[index, 7] + DBH

    # This if loop covers the cases where the stripe was defined in a way that
    # it did not include BH and DBH nor tree locator cannot be obtained from a
    # section at or close to BH. If that happens, tree axis is used to locate
    # the tree and DBH is not computed.
    if np.min(sections) > DBH:
        for i in range(n_trees):
            _axis_location(i)
    else:
        d = 1
        which_dbh = np.argmin(np.abs(sections - DBH))  # Which section is closer to BH.

        # get surrounding sections too
        lower_d_section = max(0, which_dbh - d)
        upper_d_section = min(sections.shape[0], which_dbh + d)
        # BH section and its neighbors. From now on, neighborhood
        close_to_dbh = np.arange(lower_d_section, upper_d_section + 1)  # upper bound is exclusive

        for i in range(n_trees):  # For each tree
            which_valid_R = R[i, close_to_dbh] > 0  # From neighborhood, select only those with non 0 radius
            # From neighborhood, select only those with outlier probability lower than 30 %
            which_valid_out = outliers[i, close_to_dbh] < 0.3
            # only those with sector occupancy higher than 30 %
            which_valid_sector_perct = sector_perct[i, close_to_dbh] > 30.0
            # valid points could be retrieved as well / i.e. only those with enough points in inner circle

            # If there are valid sections among the selected
            if np.any(which_valid_R) & np.any(which_valid_out):
                # If first section is BH section and if itself and its only neighbor are valid
                if (
                    (lower_d_section == 0)
                    & (np.all(which_valid_R))
                    & (np.all(which_valid_out))
                    & np.all(which_valid_sector_perct)
                ):  # Only happens when which_dbh == 0 in this case which_valid_points should be used here
                    # If they are coherent: difference among their radii is not larger than 10 % of the largest radius
                    if np.abs(R[i, close_to_dbh[0]] - R[i, close_to_dbh[1]]) < np.max(R[i, close_to_dbh]) * 0.1:
                        _dbh_location(i, which_dbh)
                    # If not all of them are valid, then there is no coherence and the axis location is used
                    else:
                        _axis_location(i)

                # If last section is BH section and if itself and its only neighbor are valid
                elif (upper_d_section == sections.shape[0]) & (np.all(which_valid_R)) & (np.all(which_valid_out)):
                    # if they are coherent; difference among their radii is not larger than 15 % of the largest radius
                    if np.abs(R[i, close_to_dbh[0]] - R[i, close_to_dbh[1]]) < np.max(R[i, close_to_dbh]) * 0.15:
                        # use BH section diameter as DBH
                        _dbh_location(i, which_dbh)

                    # If not all of them are valid, then there is no coherence in
                    # any case, and the axis location is used and DBH is not computed
                    else:
                        _axis_location(i)

                # In any other case, BH section is not first or last section, so it has 2 neighbors
                # 3 possibilities left:
                # A: Not all of three sections are valid: there is no possible coherence
                # B: All of three sections are valid, and there is coherence among the three
                # C: All of three sections are valid, but there is only coherence among neighbors
                # and not BH section or All of three sections are valid, but there is no coherence
                else:
                    # Case A:
                    if not ((np.all(which_valid_R)) & (np.all(which_valid_out)) & np.all(which_valid_sector_perct)):
                        _axis_location(i)
                    # case B&C:
                    else:
                        valid_sections = close_to_dbh  # Valid sections indexes
                        valid_radii = R[i, valid_sections]  # Valid sections radii
                        median_radius = np.median(valid_radii)  # Valid sections median radius
                        # Valid sections absolute deviation from median radius
                        abs_dev = np.abs(valid_radii - median_radius)
                        mad = np.median(abs_dev)  # Median absolute deviation
                        # Only keep sections close to median radius (3 MAD criterion)
                        filtered_sections = valid_sections[abs_dev <= 3 * mad]
                        # 3 things can happen here:
                        # There are no deviated sections --> there is coherence among 3 --> case B
                        # There are 2 deviated sections --> only median radius survives filter --> case C
                        # Case B
                        if filtered_sections.shape[0] == close_to_dbh.shape[0]:
                            _dbh_location(i, which_dbh)
                        # Case C
                        else:
                            _axis_location(i)
            # If there is not a single section that either has non 0 radius nor low
            # outlier probability, there is nothing else to do -> axis location is used
            else:
                _axis_location(i)

    return dbh_values, tree_locations
