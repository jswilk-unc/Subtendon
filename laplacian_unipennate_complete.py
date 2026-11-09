"""
Laplacian Fascicle Reconstruction
==================================
Implementation of the method described in:
    Choi HF, Blemker SS (2013) Skeletal Muscle Fascicle Arrangements Can Be
    Reconstructed Using a Laplacian Vector Field Simulation. PLoS ONE 8(10): e77576.

Physics summary
---------------
The Laplace equation  ∇²φ = 0  is solved inside the muscle volume.
- Muscle belly surface  →  zero Neumann flux  (∂φ/∂n = 0)
- Origin attachment     →  inflow  flux  g_in  (uniform, per unit area)
- Insertion attachment  →  outflow flux  g_out (scaled so total flux balances)
Fascicle tracts are then traced as streamlines of ∇φ.

Dependencies (install via pip)
-------------------------------
    pip install pyvista tetgen scipy numpy

Usage
-----
Set the six paths in the  ── INPUT / OUTPUT PATHS ──  section below,
then run the script in PyCharm (or any Python environment).
All outputs land in OUTPUT_DIR.
"""

import os
import numpy as np
import pyvista as pv
import tetgen

from scipy.spatial  import cKDTree
from scipy.sparse   import coo_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# ── INPUT / OUTPUT PATHS  (edit these before running) ──
# ============================================================

# Base folder containing all STL files and where model output folders will be created
BASE_OUTPUT_DIR = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers"

# ============================================================
# SOLEUS
# ============================================================
SOLEUS_MUSCLE_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\soleus.stl"
SOLEUS_ORIGIN_PATCH_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\soleus-origin.stl"
SOLEUS_INSERTION_PATCH_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\soleus-insertion.stl"

# ============================================================
# LATERAL GASTROCNEMIUS
# ============================================================
LATGAS_MUSCLE_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\latgas.stl"
LATGAS_ORIGIN_PATCH_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\latgas-origin.stl"
LATGAS_INSERTION_PATCH_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\latgas-insertion.stl"

# ============================================================
# MEDIAL GASTROCNEMIUS
# ============================================================
MEDGAS_MUSCLE_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\medgas.stl"
MEDGAS_ORIGIN_PATCH_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\medgas-origin.stl"
MEDGAS_INSERTION_PATCH_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\medgas-insertion.stl"

# ============================================================
# ACHILLES
# ============================================================
ACHILLES_MUSCLE_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\achilles.stl"
ACHILLES_ORIGIN_PATCH_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\achilles-origin.stl"
ACHILLES_INSERTION_PATCH_STL = r"C:\Users\jwilk\OneDrive - University of North Carolina at Chapel Hill\MuscleFibers\achilles-insertion.stl"

# Variables used by the original calculation code.
# They are automatically updated for each model before main() runs.
MUSCLE_STL = MEDGAS_MUSCLE_STL
ORIGIN_PATCH_STL = MEDGAS_ORIGIN_PATCH_STL
INSERTION_PATCH_STL = MEDGAS_INSERTION_PATCH_STL
OUTPUT_DIR = BASE_OUTPUT_DIR

# ============================================================
# ── TUNABLE PARAMETERS ──
# ============================================================

# --- Tetrahedralization ---
TET_QUALITY = True  # enforce quality constraints (recommended)
TET_MINRATIO = 1.1  # min radius-edge ratio  (lower = better quality, slower)
TET_MINDIHEDRAL = 10.0  # min dihedral angle in degrees

# --- Patch matching ---
# How many times to subdivide the attachment patch before centroid matching.
# More subdivisions = finer matching. 2 is usually sufficient.
PATCH_SUBDIVISIONS = 2
# Tolerance multiplier relative to the median edge length of the subdivided patch.
# Increase if the script reports "0 boundary triangles matched".
PATCH_TOL_SCALE = 1.3

# --- Seeding ---
N_SEEDS = 300  # number of fascicle seed points on the origin patch
SEED_RANDOM_SEED = 42  # random seed for reproducible seeding
# Inward offset of seed points (fraction of bounding-box diagonal).
# A small offset keeps seeds just inside the volume.
SEED_OFFSET_FRAC = 0.002

# --- Streamline tracing ---
# Maximum number of RK steps per streamline.
STREAMLINE_MAX_STEPS = 6000
# A streamline must reach within this distance of the insertion to be kept.
# Expressed as a multiple of the patch-matching tolerance.
TARGET_HIT_TOL_SCALE = 2.0
# Discard fascicles shorter than this fraction of the bounding-box diagonal.
MIN_LINE_LENGTH_FRAC = 0.06

# --- Post-processing ---
LINE_SMOOTH_ITERS = 4  # Laplacian smoothing iterations on each polyline
RESAMPLE_POINTS = 100  # uniform resampling to this many points (paper uses 100)

# --- Vector field smoothing (nodal gradient averaging) ---
VECTOR_SMOOTH_ITERS = 3
VECTOR_SMOOTH_K = 14  # number of nearest neighbours used

# --- Surface (skin) fascicles ---
# These fibers hug the outer muscle surface and make the model look anatomically real.
#
# How it works:
#   Near the surface, the Laplacian gradient is blended with its projection
#   onto the local surface tangent plane (normal component removed).
#   This steers fibers to follow the muscle's curvature instead of cutting inward.
#
N_SURFACE_SEEDS = 200  # seeds on origin patch for surface-layer fibers
# Shell thickness as fraction of bounding-box diagonal.
# Fibers within this distance of the surface feel the tangent blending.
# 0.04–0.08 works well for most muscles.
SURFACE_SHELL_FRAC = 0.05
# How strongly the tangent field dominates at the surface (0=no effect, 1=fully tangent).
# 0.85 gives fibers that hug the surface without collapsing onto it.
SURFACE_BLEND = 0.85
# Extra distance filter: surface fascicles must stay within this fraction of shell
# thickness from the surface on average.  Increase to allow more "dipping" inward.
SURFACE_MAX_MEAN_DIST_FRAC = 0.60
SURFACE_MAX_POINT_DIST_FRAC = 1.40

# --- Visualisation ---
SHOW_INTERACTIVE_PLOT = False  # set True to open an interactive 3D window at the end

# ============================================================
# ── DERIVED OUTPUT PATHS (no need to edit) ──
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_TET_FIELD = os.path.join(OUTPUT_DIR, "laplace_field.vtu")
OUT_SEEDS = os.path.join(OUTPUT_DIR, "seed_points_deep.vtk")
OUT_SEEDS_SURFACE = os.path.join(OUTPUT_DIR, "seed_points_surface.vtk")
OUT_FASCICLES = os.path.join(OUTPUT_DIR, "fascicles_deep.vtk")
OUT_FASCICLES_SURF = os.path.join(OUTPUT_DIR, "fascicles_surface.vtk")
OUT_FASCICLES_ALL = os.path.join(OUTPUT_DIR, "fascicles_combined.vtk")
OUT_PREVIEW_PNG = os.path.join(OUTPUT_DIR, "preview.png")


# ============================================================
# SECTION 1 – MESH HELPERS
# ============================================================

def load_surface(path: str) -> pv.PolyData:
    """Read any PyVista-supported surface and return a clean triangulation."""
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh.triangulate().clean()


def get_tri_faces(poly: pv.PolyData) -> np.ndarray:
    """Return (N, 3) integer array of triangle vertex indices."""
    faces = poly.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Mesh is not purely triangular.")
    return faces[:, 1:4].copy()


def median_edge_length(poly: pv.PolyData) -> float:
    """Median edge length of all unique edges in a triangle mesh."""
    faces = get_tri_faces(poly)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(
        poly.points[edges[:, 0]] - poly.points[edges[:, 1]], axis=1
    )
    return float(np.median(lengths))


def safe_subdivide(poly: pv.PolyData, n: int) -> pv.PolyData:
    """Linear subdivision with graceful fallback if subdivision fails."""
    out = poly.triangulate().clean()
    for _ in range(n):
        try:
            out = out.subdivide(1, subfilter="linear")
        except Exception:
            break
    return out.clean()


def triangle_areas(pts: np.ndarray) -> np.ndarray:
    """Return area of each triangle given (N, 3, 3) vertex array."""
    cross = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def normalize_vectors(v: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(mag, 1e-12)


# ============================================================
# SECTION 2 – TETRAHEDRALIZATION
# ============================================================

def tetrahedralize(surface: pv.PolyData):
    """
    Fill the surface with linear tetrahedra using TetGen.

    Returns
    -------
    nodes : (N, 3) float64
    tets  : (M, 4) int64   – zero-based node indices
    grid  : pv.UnstructuredGrid
    """
    print("Running TetGen tetrahedralization …")
    tgen = tetgen.TetGen(surface)
    result = tgen.tetrahedralize(
        nobisect=True,
        quality=TET_QUALITY,
        minratio=TET_MINRATIO,
        mindihedral=TET_MINDIHEDRAL,
        order=1,
    )
    nodes = np.asarray(result[0], dtype=np.float64)
    tets = np.asarray(result[1], dtype=np.int64)

    # TetGen sometimes uses 1-based indexing
    if tets.min() == 1:
        tets -= 1

    grid = tgen.grid
    print(f"  {len(nodes):,} nodes, {len(tets):,} tetrahedra")
    return nodes, tets, grid


def boundary_surface_and_global_faces(
        grid: pv.UnstructuredGrid, nodes: np.ndarray
):
    """
    Extract the boundary triangulation of the tet mesh and map its local
    vertex indices back to the global node array.
    """
    surf = grid.extract_surface().triangulate().clean()
    kd = cKDTree(nodes)
    _, gi = kd.query(surf.points, k=1)

    local_faces = get_tri_faces(surf)
    global_faces = gi[local_faces]
    return surf, global_faces


# ============================================================
# SECTION 3 – PATCH → BOUNDARY TRIANGLE MATCHING
# ============================================================

def match_patch_to_boundary(
        patch: pv.PolyData,
        boundary_surf: pv.PolyData,
        boundary_faces: np.ndarray,
        tol_scale: float = 1.0,
        subdivisions: int = 2,
        label: str = "patch",
):
    """
    Find boundary triangles whose centroids lie within a tolerance of the
    (subdivided) attachment patch.

    Returns
    -------
    tri_ids   : 1-D int array  – indices into boundary_faces / boundary_surf
    patch_pts : (K, 3) float   – dense point set on the patch (for seeding)
    tol       : float          – tolerance that was used
    """
    dense = safe_subdivide(patch, subdivisions)
    patch_pts = dense.points

    local_faces = get_tri_faces(boundary_surf)
    tri_pts = boundary_surf.points[local_faces]
    centroids = tri_pts.mean(axis=1)

    kd_patch = cKDTree(patch_pts)
    dist, _ = kd_patch.query(centroids, k=1)

    base_tol = median_edge_length(dense) * tol_scale
    tri_ids = None
    used_tol = None

    for scale in [1.0, 1.5, 2.0, 3.0]:
        tol = base_tol * scale
        ids = np.where(dist <= tol)[0]
        if len(ids) > 0:
            tri_ids = ids
            used_tol = tol
            break

    if tri_ids is None or len(tri_ids) == 0:
        raise RuntimeError(
            f"Could not match any boundary triangles to '{label}'. "
            "Try increasing PATCH_TOL_SCALE or PATCH_SUBDIVISIONS."
        )

    print(f"  {label}: {len(tri_ids)} boundary triangles matched (tol = {used_tol:.4f} mm)")
    return tri_ids, patch_pts, used_tol


# ============================================================
# SECTION 4 – FEM ASSEMBLY  (P1 linear tetrahedra)
# ============================================================

def assemble_stiffness(nodes: np.ndarray, tets: np.ndarray):
    """
    Assemble the global FEM stiffness (Laplacian) matrix K.

    For a P1 tetrahedral element the element stiffness matrix is:
        K_e = V_e * G^T G
    where G = invX[1:, :] are the Cartesian gradients of the basis functions
    and V_e is the element volume.

    Uses vectorised NumPy; no Python loops over elements.
    """
    print("Assembling global stiffness matrix (vectorised) …")

    pts = nodes[tets]  # (M, 4, 3)
    X = np.concatenate(
        [np.ones((len(tets), 4, 1)), pts], axis=2
    )  # (M, 4, 4)

    # Invert all element matrices at once
    try:
        invX = np.linalg.inv(X)  # (M, 4, 4)
    except np.linalg.LinAlgError:
        # Fallback: element-wise (handles degenerate tets gracefully)
        invX = np.zeros_like(X)
        for i, xi in enumerate(X):
            try:
                invX[i] = np.linalg.inv(xi)
            except np.linalg.LinAlgError:
                pass

    # grads of the 4 basis functions w.r.t. x,y,z  → shape (M, 4, 3)
    # invX is (M,4,4); rows 1-3 of each column give the x,y,z gradient
    # of the corresponding basis function, so we take rows 1: and transpose.
    grads = invX[:, 1:, :].transpose(0, 2, 1)  # (M, 4, 3)

    # Element volumes  V = |det([p1-p0, p2-p0, p3-p0])| / 6
    d = pts[:, 1:] - pts[:, :1]  # (M, 3, 3)
    vols = np.abs(np.linalg.det(d)) / 6.0  # (M,)

    # Element stiffness:  Ke[a,b] = V_e * (∇N_a · ∇N_b)
    # grads: (M, 4, 3)  →  einsum over spatial dim → (M, 4, 4)
    Ke = vols[:, None, None] * np.einsum("mak,mbk->mab", grads, grads)

    # Scatter into global COO arrays – 16 (a,b) pairs per element
    n_nodes = nodes.shape[0]
    local = np.arange(4)
    ii, jj = np.meshgrid(local, local, indexing="ij")
    ii = ii.ravel()  # shape (16,)
    jj = jj.ravel()

    # rows/cols: for each of M elements, each of 16 pairs → (M*16,)
    rows = tets[:, ii].ravel()
    cols = tets[:, jj].ravel()
    data = Ke[:, ii, jj].ravel()

    K = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    print(f"  Stiffness matrix: {n_nodes:,} × {n_nodes:,}, {K.nnz:,} non-zeros")
    return K


def assemble_neumann_rhs(
        n_nodes: int,
        boundary_surf: pv.PolyData,
        global_faces: np.ndarray,
        origin_ids: np.ndarray,
        insert_ids: np.ndarray,
) -> np.ndarray:
    """
    Assemble the right-hand side vector for the Neumann boundary conditions.

    Following equation (2) of the paper:
        g_in  * A_in  =  g_out * A_out   (flux balance)

    We set g_in = -1 and scale g_out to balance, consistent with
    an inflow at the origin and outflow at the insertion.
    """
    local_faces = get_tri_faces(boundary_surf)
    tri_pts = boundary_surf.points[local_faces]
    areas = triangle_areas(tri_pts)

    A_origin = float(areas[origin_ids].sum())
    A_insert = float(areas[insert_ids].sum())

    if A_origin <= 0 or A_insert <= 0:
        raise RuntimeError("Attachment patch area is zero – check STL alignment.")

    g_in = -1.0
    g_out = A_origin / A_insert  # balances total flux

    print(f"  Origin area: {A_origin:.2f} mm²  |  Insertion area: {A_insert:.2f} mm²")
    print(f"  Neumann flux:  g_in = {g_in:.4f}  |  g_out = {g_out:.4f}")

    b = np.zeros(n_nodes, dtype=np.float64)

    for ids, g in [(origin_ids, g_in), (insert_ids, g_out)]:
        for tri_id in ids:
            gnodes = global_faces[tri_id]
            load = g * areas[tri_id] / 3.0
            b[gnodes[0]] += load
            b[gnodes[1]] += load
            b[gnodes[2]] += load

    return b


def solve_neumann(K, b, anchor: int = 0) -> np.ndarray:
    """
    Solve the pure-Neumann system K u = b by pinning one node to zero
    (removes the singular nullspace).  The absolute value of the solution
    is arbitrary; only the gradient matters.
    """
    print("Solving linear system (pinning node 0 to fix Neumann nullspace) …")
    Kl = K.tolil()
    Kl[anchor, :] = 0.0
    Kl[:, anchor] = 0.0
    Kl[anchor, anchor] = 1.0

    b2 = b.copy()
    b2[anchor] = 0.0

    u = spsolve(Kl.tocsr(), b2)
    print(f"  Solution range: [{u.min():.4f}, {u.max():.4f}]")
    return u


# ============================================================
# SECTION 5 – GRADIENT FIELD  (fascicle orientation vectors)
# ============================================================

def nodal_gradients(nodes: np.ndarray, tets: np.ndarray, u: np.ndarray):
    """
    Compute volume-weighted nodal gradient field of scalar potential u.

    The constant element gradient is:
        ∇u_e = G · u_e
    where G = invX[1:, :] as in the stiffness assembly.

    Returns
    -------
    grad_raw  : (N, 3) raw (non-unit) gradients
    vec_unit  : (N, 3) unit vectors  ← fascicle orientation
    """
    print("Computing nodal gradient field …")

    pts = nodes[tets]
    X = np.concatenate([np.ones((len(tets), 4, 1)), pts], axis=2)

    try:
        invX = np.linalg.inv(X)
    except np.linalg.LinAlgError:
        invX = np.zeros_like(X)
        for i, xi in enumerate(X):
            try:
                invX[i] = np.linalg.inv(xi)
            except:
                pass

    # grads of basis functions: (M, 4, 3)
    grads = invX[:, 1:, :].transpose(0, 2, 1)
    d = pts[:, 1:] - pts[:, :1]
    vols = np.abs(np.linalg.det(d)) / 6.0  # (M,)

    # Element gradient ∇u_e = sum_a (u_a * ∇N_a) → (M, 3)
    u_e = u[tets]  # (M, 4)
    grad_e = np.einsum("mak,ma->mk", grads, u_e)  # (M, 3)

    # Accumulate volume-weighted gradients at nodes
    grad_acc = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    weight_acc = np.zeros(nodes.shape[0], dtype=np.float64)

    for local_i in range(4):
        global_i = tets[:, local_i]  # (M,)
        np.add.at(grad_acc, global_i, (grad_e * vols[:, None]))
        np.add.at(weight_acc, global_i, vols)

    weight_acc = np.maximum(weight_acc, 1e-16)
    grad_raw = grad_acc / weight_acc[:, None]

    mag = np.linalg.norm(grad_raw, axis=1, keepdims=True)
    vec_unit = grad_raw / np.maximum(mag, 1e-12)

    return grad_raw, vec_unit


def smooth_vector_field(
        nodes: np.ndarray,
        vectors: np.ndarray,
        k: int = 14,
        n_iters: int = 3,
) -> np.ndarray:
    """
    k-NN weighted smoothing of a unit vector field.
    Neighbours closer to a node get higher weight.
    """
    print(f"Smoothing vector field ({n_iters} iterations, k={k}) …")
    k = min(k, len(nodes))
    kd = cKDTree(nodes)
    dist, idx = kd.query(nodes, k=k)

    out = vectors.copy()
    for _ in range(n_iters):
        nbr = out[idx]  # (N, k, 3)
        w = 1.0 / np.maximum(dist, 1e-8)  # (N, k)
        w[:, 0] *= 3.0  # self-weight boost
        out = (nbr * w[..., None]).sum(axis=1) / w.sum(axis=1)[:, None]
        mag = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(mag, 1e-12)

    return out


# ============================================================
# SECTION 5b – SURFACE-TANGENT VECTOR FIELD
# ============================================================

def build_surface_normal_field(muscle_surface: pv.PolyData, nodes: np.ndarray):
    """
    For every tet node, find the nearest muscle surface point and return:
      - nearest_pts    : (N,3)  nearest surface point
      - nearest_normals: (N,3)  outward unit normal at that point
      - d_surf         : (N,)   distance from node to nearest surface point
    """
    surf = muscle_surface.compute_normals(
        point_normals=True, cell_normals=False,
        auto_orient_normals=True, consistent_normals=True
    )
    surf_pts = surf.points
    surf_nrm = normalize_vectors(np.asarray(surf.point_data["Normals"]))

    kd = cKDTree(surf_pts)
    d_surf, idx = kd.query(nodes, k=1)

    return surf_pts[idx], surf_nrm[idx], d_surf


def build_surface_tangent_field(
        base_vec: np.ndarray,  # (N,3) smoothed Laplacian unit vectors
        nearest_normals: np.ndarray,  # (N,3) outward surface normals per node
        d_surf: np.ndarray,  # (N,)  distance to surface
        shell_thickness: float,  # nodes closer than this feel the blend
        blend_strength: float = 0.85,  # 1 = fully tangent at surface, 0 = no effect
) -> np.ndarray:
    """
    Near the muscle surface, replace the normal-to-surface component of each
    fascicle vector with zero, making fibers hug the surface curvature.

    The blend weight w goes from blend_strength (at the surface, d=0) to 0
    (at shell_thickness depth).  Deep nodes are unaffected.

    Tangent projection:
        tang = base_vec - (base_vec · n̂) n̂
    Blended field:
        vec = (1-w)*base_vec + w*tang   then re-normalised
    """
    base_vec = normalize_vectors(base_vec)
    nearest_normals = normalize_vectors(nearest_normals)

    # Project base_vec onto the tangent plane of the surface
    dotn = np.sum(base_vec * nearest_normals, axis=1, keepdims=True)
    tang = base_vec - dotn * nearest_normals

    # Fallback: if tang is near-zero (fiber already perpendicular to surface)
    tang_mag = np.linalg.norm(tang, axis=1, keepdims=True)
    degenerate = (tang_mag[:, 0] < 1e-8)
    tang = tang / np.maximum(tang_mag, 1e-12)
    tang[degenerate] = base_vec[degenerate]

    # Blend weight: linear decay from surface to shell_thickness depth
    w = blend_strength * np.clip(1.0 - d_surf / shell_thickness, 0.0, 1.0)

    blended = (1.0 - w)[:, None] * base_vec + w[:, None] * tang
    return normalize_vectors(blended)


def filter_surface_streamlines(
        streamlines: pv.PolyData,
        target_pts: np.ndarray,
        muscle_surface_pts: np.ndarray,
        shell_thickness: float,
        target_tol: float,
        bbox_diag: float,
) -> pv.PolyData:
    """
    Like filter_streamlines, but also enforces that the fascicle stays
    close to the muscle surface throughout its length.
    """
    if streamlines.n_cells == 0:
        return pv.PolyData()

    kd_target = cKDTree(target_pts)
    kd_surface = cKDTree(muscle_surface_pts)
    min_len = MIN_LINE_LENGTH_FRAC * bbox_diag
    kept = []

    max_mean = SURFACE_MAX_MEAN_DIST_FRAC * shell_thickness
    max_point = SURFACE_MAX_POINT_DIST_FRAC * shell_thickness

    for ids in parse_cells(streamlines):
        if len(ids) < 4:
            continue
        pts = streamlines.points[ids]

        # Must reach insertion
        d_target, _ = kd_target.query(pts, k=1)
        if float(d_target.min()) > target_tol:
            continue

        # Must stay near the surface
        d_surf, _ = kd_surface.query(pts, k=1)
        if float(d_surf.mean()) > max_mean:
            continue
        if float(d_surf.max()) > max_point:
            continue

        if polyline_length(pts) < min_len:
            continue

        pts = smooth_polyline(pts, LINE_SMOOTH_ITERS)
        pts = resample_polyline(pts, RESAMPLE_POINTS)
        kept.append(pts)

    return build_polydata(kept)


def make_surface_seed_points(
        origin_patch: pv.PolyData,
        muscle_surface: pv.PolyData,
        nodes: np.ndarray,
        node_normals: np.ndarray,
        bbox_diag: float,
) -> pv.PolyData:
    """
    Sample N_SURFACE_SEEDS on the origin patch and offset them only very
    slightly inward — much less than the deep seeds — so they sit right
    inside the surface shell.
    """
    # Use a very small offset (¼ of the deep seed offset) to stay in the shell
    surface_offset_frac = SEED_OFFSET_FRAC * 0.25

    pts = uniform_surface_samples(origin_patch, N_SURFACE_SEEDS,
                                  seed=SEED_RANDOM_SEED + 99)
    kd = cKDTree(nodes)
    _, nn = kd.query(pts, k=1)

    offset = -surface_offset_frac * bbox_diag * node_normals[nn]
    seed_pts = pts + offset
    return pv.PolyData(seed_pts)


def uniform_surface_samples(poly: pv.PolyData, n: int, seed: int = 42):
    """
    Area-weighted random sampling of points on a triangle mesh surface.
    Returns (n, 3) float array.
    """
    poly = poly.triangulate().clean()
    faces = poly.faces.reshape(-1, 4)[:, 1:]
    p0, p1, p2 = [poly.points[faces[:, i]] for i in range(3)]

    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    areas = np.maximum(areas, 1e-16)
    prob = areas / areas.sum()

    rng = np.random.default_rng(seed)
    tri_id = rng.choice(len(faces), size=n, replace=True, p=prob)

    a, b_, c = p0[tri_id], p1[tri_id], p2[tri_id]
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    pts = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b_ + (r1 * r2)[:, None] * c
    return pts


def make_seed_points(
        origin_patch: pv.PolyData,
        nodes: np.ndarray,
        surface_normals: np.ndarray,
        bbox_diag: float,
) -> pv.PolyData:
    """
    Sample N_SEEDS points on the origin patch and nudge them inward
    along the surface normal so they sit just inside the volume.
    """
    pts = uniform_surface_samples(origin_patch, N_SEEDS, seed=SEED_RANDOM_SEED)
    kd = cKDTree(nodes)
    _, nn = kd.query(pts, k=1)

    # Inward = opposite to outward normal
    offset = -SEED_OFFSET_FRAC * bbox_diag * surface_normals[nn]
    seed_pts = pts + offset
    return pv.PolyData(seed_pts)


# ============================================================
# SECTION 7 – STREAMLINE TRACING
# ============================================================

def trace_streamlines(
        grid: pv.UnstructuredGrid,
        seeds: pv.PolyData,
        bbox_diag: float,
        vectors_key: str = "fascicle_vec",
) -> pv.PolyData:
    """Trace streamlines forward through the vector field."""
    common = dict(
        vectors=vectors_key,
        integration_direction="forward",
        max_steps=STREAMLINE_MAX_STEPS,
        terminal_speed=1e-12,
        compute_vorticity=False,
    )
    try:
        sl = grid.streamlines_from_source(
            seeds, max_length=2.5 * bbox_diag, **common
        )
    except TypeError:
        try:
            sl = grid.streamlines_from_source(
                seeds, max_time=2.5 * bbox_diag, **common
            )
        except TypeError:
            sl = grid.streamlines_from_source(seeds, **common)

    return sl


# ============================================================
# SECTION 8 – POLYLINE POST-PROCESSING
# ============================================================

def polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def smooth_polyline(pts: np.ndarray, n_iters: int = 4) -> np.ndarray:
    """Simple Laplacian smoothing; endpoints are fixed."""
    p = pts.copy()
    for _ in range(n_iters):
        new = p.copy()
        new[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
        p = new
    return p


def resample_polyline(pts: np.ndarray, n: int = 100) -> np.ndarray:
    """Uniformly resample a polyline to n equidistant points."""
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, pts[:, d]) for d in range(3)])


def parse_cells(poly: pv.PolyData):
    """Parse PyVista polyline cells → list of index arrays."""
    lines = poly.lines
    cells = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        cells.append(np.asarray(lines[i + 1: i + 1 + n], dtype=np.int64))
        i += n + 1
    return cells


def build_polydata(line_arrays):
    """Assemble a list of (N_i, 3) arrays into a single PolyData with polylines."""
    if not line_arrays:
        return pv.PolyData()
    all_pts, lines, offset = [], [], 0
    for pts in line_arrays:
        if len(pts) < 2:
            continue
        all_pts.append(pts)
        lines.extend([len(pts)] + list(range(offset, offset + len(pts))))
        offset += len(pts)
    if not all_pts:
        return pv.PolyData()
    poly = pv.PolyData(np.vstack(all_pts))
    poly.lines = np.asarray(lines, dtype=np.int64)
    return poly


def filter_streamlines(
        streamlines: pv.PolyData,
        target_pts: np.ndarray,  # dense point set on insertion patch
        bbox_diag: float,
        target_tol: float,
) -> pv.PolyData:
    """
    Keep only streamlines that reach within target_tol of the insertion patch.
    Each kept streamline is smoothed and resampled to RESAMPLE_POINTS points.
    """
    if streamlines.n_cells == 0:
        return pv.PolyData()

    kd = cKDTree(target_pts)
    cells = parse_cells(streamlines)
    min_len = MIN_LINE_LENGTH_FRAC * bbox_diag
    kept = []

    for ids in cells:
        if len(ids) < 4:
            continue
        pts = streamlines.points[ids]
        d, _ = kd.query(pts, k=1)
        if float(d.min()) > target_tol:
            continue  # did not reach insertion
        if polyline_length(pts) < min_len:
            continue  # too short
        pts = smooth_polyline(pts, LINE_SMOOTH_ITERS)
        pts = resample_polyline(pts, RESAMPLE_POINTS)
        kept.append(pts)

    return build_polydata(kept)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Laplacian Fascicle Reconstruction")
    print("Choi & Blemker, PLoS ONE 2013")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load surfaces
    # ----------------------------------------------------------
    print("\n[1/8] Loading surfaces …")
    muscle = load_surface(MUSCLE_STL)
    origin = load_surface(ORIGIN_PATCH_STL)
    insert = load_surface(INSERTION_PATCH_STL)

    print(f"  Muscle  : {muscle.n_points:,} pts, {muscle.n_cells:,} faces")
    print(f"  Origin  : {origin.n_points:,} pts, {origin.n_cells:,} faces")
    print(f"  Insertion: {insert.n_points:,} pts, {insert.n_cells:,} faces")

    bbox_diag = float(np.linalg.norm(
        muscle.bounds[1::2] - np.array(muscle.bounds[::2])
    ))
    print(f"  Bounding-box diagonal: {bbox_diag:.2f} mm")

    # ----------------------------------------------------------
    # 2. Tetrahedralize
    # ----------------------------------------------------------
    print("\n[2/8] Tetrahedralizing muscle volume …")
    nodes, tets, grid = tetrahedralize(muscle)

    # ----------------------------------------------------------
    # 3. Map patches to boundary triangles
    # ----------------------------------------------------------
    print("\n[3/8] Matching attachment patches to boundary triangles …")
    bsurf, bfaces = boundary_surface_and_global_faces(grid, nodes)

    origin_ids, origin_dense_pts, origin_tol = match_patch_to_boundary(
        origin, bsurf, bfaces,
        tol_scale=PATCH_TOL_SCALE,
        subdivisions=PATCH_SUBDIVISIONS,
        label="origin",
    )
    insert_ids, insert_dense_pts, insert_tol = match_patch_to_boundary(
        insert, bsurf, bfaces,
        tol_scale=PATCH_TOL_SCALE,
        subdivisions=PATCH_SUBDIVISIONS,
        label="insertion",
    )

    # ----------------------------------------------------------
    # 4. Assemble & solve the Laplace equation
    # ----------------------------------------------------------
    print("\n[4/8] FEM assembly …")
    K = assemble_stiffness(nodes, tets)

    print("\n[5/8] Assembling Neumann RHS …")
    b = assemble_neumann_rhs(
        nodes.shape[0], bsurf, bfaces, origin_ids, insert_ids
    )

    print("\n[6/8] Solving Laplace equation …")
    u = solve_neumann(K, b)

    # ----------------------------------------------------------
    # 5. Gradient field, smoothing & surface-tangent blend
    # ----------------------------------------------------------
    grad_raw, vec_unit = nodal_gradients(nodes, tets, u)
    vec_smooth = smooth_vector_field(
        nodes, vec_unit, k=VECTOR_SMOOTH_K, n_iters=VECTOR_SMOOTH_ITERS
    )

    # Build surface normal field for every node
    print("  Building surface normal field for tangent blending …")
    nearest_surf_pts, node_normals, d_surf = build_surface_normal_field(muscle, nodes)
    shell_thickness = SURFACE_SHELL_FRAC * bbox_diag
    print(f"  Surface shell thickness: {shell_thickness:.2f} mm")

    # Surface-tangent blended field (used for surface-layer fascicles)
    vec_surface = build_surface_tangent_field(
        base_vec=vec_smooth,
        nearest_normals=node_normals,
        d_surf=d_surf,
        shell_thickness=shell_thickness,
        blend_strength=SURFACE_BLEND,
    )

    # Attach all fields to the tet grid and save
    grid.point_data["potential"] = u
    grid.point_data["gradient_raw"] = grad_raw
    grid.point_data["fascicle_vec"] = vec_smooth
    grid.point_data["surface_fasc_vec"] = vec_surface
    grid.save(OUT_TET_FIELD)
    print(f"\n  Laplace field saved → {OUT_TET_FIELD}")

    # ----------------------------------------------------------
    # 6. Surface normals for seed offset (reuse node_normals above)
    # ----------------------------------------------------------
    # node_normals already computed in step 5 above

    # ----------------------------------------------------------
    # 7a. Deep (interior) fascicles
    # ----------------------------------------------------------
    print("\n[7/8] Seeding and tracing deep (interior) fascicles …")
    seeds_deep = make_seed_points(origin, nodes, node_normals, bbox_diag)
    seeds_deep.save(OUT_SEEDS)
    print(f"  {N_SEEDS} deep seed points saved → {OUT_SEEDS}")

    sl_deep = trace_streamlines(grid, seeds_deep, bbox_diag, "fascicle_vec")
    print(f"  Raw deep streamlines: {sl_deep.n_cells}")

    target_tol = max(origin_tol, insert_tol) * TARGET_HIT_TOL_SCALE
    fascicles_deep = filter_streamlines(
        sl_deep, insert_dense_pts, bbox_diag, target_tol
    )
    fascicles_deep.save(OUT_FASCICLES)
    print(f"  Deep fascicles after filtering: {fascicles_deep.n_cells}")
    print(f"  Saved → {OUT_FASCICLES}")

    # ----------------------------------------------------------
    # 7b. Surface-layer fascicles (hug the muscle surface)
    # ----------------------------------------------------------
    print("\n  Seeding and tracing surface-layer fascicles …")
    seeds_surf = make_surface_seed_points(
        origin, muscle, nodes, node_normals, bbox_diag
    )
    seeds_surf.save(OUT_SEEDS_SURFACE)
    print(f"  {N_SURFACE_SEEDS} surface seed points saved → {OUT_SEEDS_SURFACE}")

    sl_surf = trace_streamlines(grid, seeds_surf, bbox_diag, "surface_fasc_vec")
    print(f"  Raw surface streamlines: {sl_surf.n_cells}")

    fascicles_surf = filter_surface_streamlines(
        sl_surf,
        insert_dense_pts,
        muscle.points,
        shell_thickness,
        target_tol,
        bbox_diag,
    )
    fascicles_surf.save(OUT_FASCICLES_SURF)
    print(f"  Surface fascicles after filtering: {fascicles_surf.n_cells}")
    print(f"  Saved → {OUT_FASCICLES_SURF}")

    # ----------------------------------------------------------
    # 7c. Combine deep + surface into one file
    # ----------------------------------------------------------
    all_lines = (
            [fascicles_deep.points[ids].copy() for ids in parse_cells(fascicles_deep)]
            + [fascicles_surf.points[ids].copy() for ids in parse_cells(fascicles_surf)]
    )
    fascicles_all = build_polydata(all_lines)
    fascicles_all.save(OUT_FASCICLES_ALL)
    print(f"\n  Combined fascicles: {fascicles_all.n_cells}  →  {OUT_FASCICLES_ALL}")

    # ----------------------------------------------------------
    # 8. Quick statistics
    # ----------------------------------------------------------
    for label, fasc in [("Deep", fascicles_deep), ("Surface", fascicles_surf),
                        ("Combined", fascicles_all)]:
        if fasc.n_cells > 0:
            cells = parse_cells(fasc)
            lengths = [polyline_length(fasc.points[ids]) for ids in cells]
            print(f"  {label:10s}  n={fasc.n_cells:4d}  "
                  f"len = {np.mean(lengths):.1f} ± {np.std(lengths):.1f} mm  "
                  f"[{np.min(lengths):.1f} – {np.max(lengths):.1f}]")

    # ----------------------------------------------------------
    # 9. Render preview (deep = orange, surface = red)
    # ----------------------------------------------------------
    print("\n[8/8] Rendering preview …")
    pl = pv.Plotter(off_screen=not SHOW_INTERACTIVE_PLOT)
    pl.set_background("white")
    pl.add_mesh(muscle, color="lightgray", opacity=0.15, label="Muscle")
    pl.add_mesh(origin, color="green", opacity=0.85, label="Origin")
    pl.add_mesh(insert, color="blue", opacity=0.85, label="Insertion")

    radius = 0.001 * bbox_diag
    if fascicles_deep.n_cells > 0:
        try:
            pl.add_mesh(fascicles_deep.tube(radius=radius),
                        color="orange", label="Deep fascicles")
        except Exception:
            pl.add_mesh(fascicles_deep, color="orange", line_width=2)

    if fascicles_surf.n_cells > 0:
        try:
            pl.add_mesh(fascicles_surf.tube(radius=radius * 0.8),
                        color="red", label="Surface fascicles")
        except Exception:
            pl.add_mesh(fascicles_surf, color="red", line_width=2)

    pl.add_legend()
    pl.show(screenshot=OUT_PREVIEW_PNG)
    print(f"  Preview saved → {OUT_PREVIEW_PNG}")

    if SHOW_INTERACTIVE_PLOT:
        pl2 = pv.Plotter()
        pl2.set_background("white")
        pl2.add_mesh(muscle, color="lightgray", opacity=0.15)
        pl2.add_mesh(origin, color="green", opacity=0.85)
        pl2.add_mesh(insert, color="blue", opacity=0.85)
        if fascicles_deep.n_cells > 0:
            try:
                pl2.add_mesh(fascicles_deep.tube(radius=radius), color="orange")
            except Exception:
                pl2.add_mesh(fascicles_deep, color="orange", line_width=2)
        if fascicles_surf.n_cells > 0:
            try:
                pl2.add_mesh(fascicles_surf.tube(radius=radius * 0.8), color="red")
            except Exception:
                pl2.add_mesh(fascicles_surf, color="red", line_width=2)
        pl2.show()

    print("\n" + "=" * 60)
    print("Done.  All outputs written to:", OUTPUT_DIR)
    print("=" * 60)


# ============================================================
# MULTI-MODEL RUNNER
# ============================================================
#
# The calculation code above is unchanged.
# This wrapper changes the input/output paths and calls the
# original main() once for each model.
# ============================================================

def run_model(name, muscle_stl, origin_patch_stl, insertion_patch_stl):
    global MUSCLE_STL, ORIGIN_PATCH_STL, INSERTION_PATCH_STL, OUTPUT_DIR
    global OUT_TET_FIELD, OUT_SEEDS, OUT_SEEDS_SURFACE
    global OUT_FASCICLES, OUT_FASCICLES_SURF, OUT_FASCICLES_ALL
    global OUT_PREVIEW_PNG

    # Set this model's input files
    MUSCLE_STL = muscle_stl
    ORIGIN_PATCH_STL = origin_patch_stl
    INSERTION_PATCH_STL = insertion_patch_stl

    # Give this model its own output folder
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Rebuild the original output paths for this model
    OUT_TET_FIELD = os.path.join(OUTPUT_DIR, "laplace_field.vtu")
    OUT_SEEDS = os.path.join(OUTPUT_DIR, "seed_points_deep.vtk")
    OUT_SEEDS_SURFACE = os.path.join(OUTPUT_DIR, "seed_points_surface.vtk")
    OUT_FASCICLES = os.path.join(OUTPUT_DIR, "fascicles_deep.vtk")
    OUT_FASCICLES_SURF = os.path.join(OUTPUT_DIR, "fascicles_surface.vtk")
    OUT_FASCICLES_ALL = os.path.join(OUTPUT_DIR, "fascicles_combined.vtk")
    OUT_PREVIEW_PNG = os.path.join(OUTPUT_DIR, "preview.png")

    print("\n" + "=" * 80)
    print(f"RUNNING MODEL: {name.upper()}")
    print("=" * 80)
    print(f"Muscle STL:    {MUSCLE_STL}")
    print(f"Origin STL:    {ORIGIN_PATCH_STL}")
    print(f"Insertion STL: {INSERTION_PATCH_STL}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 80)

    # Run the original simulation
    main()

    print("\n" + "=" * 80)
    print(f"FINISHED MODEL: {name.upper()}")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    models = [
        ("soleus", SOLEUS_MUSCLE_STL, SOLEUS_ORIGIN_PATCH_STL, SOLEUS_INSERTION_PATCH_STL),
        ("latgas", LATGAS_MUSCLE_STL, LATGAS_ORIGIN_PATCH_STL, LATGAS_INSERTION_PATCH_STL),
        ("medgas", MEDGAS_MUSCLE_STL, MEDGAS_ORIGIN_PATCH_STL, MEDGAS_INSERTION_PATCH_STL),
        ("achilles", ACHILLES_MUSCLE_STL, ACHILLES_ORIGIN_PATCH_STL, ACHILLES_INSERTION_PATCH_STL),
    ]

    for name, muscle_stl, origin_stl, insertion_stl in models:
        run_model(name, muscle_stl, origin_stl, insertion_stl)

    print("\n" + "=" * 80)
    print("ALL FOUR MODELS ARE FINISHED")
    print("=" * 80)
    print(f"Results are located in: {BASE_OUTPUT_DIR}")
    print("=" * 80)
