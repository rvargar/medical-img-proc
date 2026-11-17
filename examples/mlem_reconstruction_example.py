"""
MLEM / OSEM / OS-LALM tomosynthesis reconstruction example.

Pipeline (same acquisition set-up as x_ray_simulator_example.py):
  1. load + normalise the CT volume, choose the projection plane
  2. simulate the projections with the existing CUDA ``Projector`` (mode 2)
  3. optionally corrupt the projection borders with synthetic clutter
     (collimator shadow / scatter bands, dead & saturated rows, hot pixels,
     noise) - the typical source of edge artefacts
  4. reconstruct (GPU resident iterations)
  5. evaluate against the ground truth resampled on the reconstruction grid

Two modes (``"mode"`` in the config or ``--mode``):
  recommended  run only the configured pipeline (edge handling, detector
               extrapolation, reconstruction variant) and save the volume,
               a slab overview and the metrics
  compare      run all variants side by side, write the comparison metrics,
               convergence plots and the README figures in ``misc/``

Run from the repository root:
    python examples/mlem_reconstruction_example.py --config examples/configs/recommended.json
    python examples/mlem_reconstruction_example.py --config examples/configs/compare_all.json
    python examples/mlem_reconstruction_example.py --variant mlem --n-iter 30 --no-clutter
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pycuda.gpuarray as gpuarray

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from utils.Logger import Logger  # noqa: E402
from utils.DicomCTParser import DicomCTParser  # noqa: E402
from utils.CudaWrapper import Projector  # noqa: E402
from utils.MLEM import (TomoGeometry, GpuTomoOperator, MLEMReconstructor,  # noqa: E402
                        EdgeHandling, bin_projections, build_ray_weights, extend_detector)
from utils.VolumeVisualizer import VolumeVisualizer as visu  # noqa: E402

logger = Logger().get_logger()

# ── default configuration (overridden by --config JSON and the CLI flags) ──────
DEFAULT_CONFIG = {
    'mode': 'recommended',
    'input_path': './data/Lung-Pet-CT-Dx/Lung_Dx-A0018/07-24-2008-lungc-69037/301.000000-1.25mm-74521/1',
    'out_dir': './results/mlem',
    'misc_dir': './misc',
    'plane': 'coronal',               # 'axial' | 'coronal' | 'sagittal'
    'num_projections': 20,
    'detector_px': 1512,              # simulated detector (as in the simulator example)
    'bin': 2,                         # detector binning used for the reconstruction
    'recon_vz': 40,                   # number of reconstructed slabs
    'n_iter': 20,
    'add_clutter': True,
    'seed': 7,
    'show': False,                    # interactive slice viewer at the end
    'projector_study': False,         # compare mode: ray / voxel / footprint on a finer grid
    'edge_handling': {                # EdgeHandling fields, see utils/MLEM.py
        'border_px': 2, 'auto_detect': True, 'outlier_k': 8.0,
        'sens_threshold': 0.02, 'coverage_min': 0.02,
    },
    'extrapolation': {'enabled': True, 'pad_px': 160, 'shape': 'sqrt', 'w_ext': 0.3},
    'reconstruction': {
        'variant': 'os-lalm',         # mlem | osem | osem-tv | os-lalm | vanilla
        'backprojector': 'voxel',     # voxel | ray | footprint
        'n_subsets': 10,
        'lalm': {'prior': 'hyperbola', 'beta': 1.0, 'delta': 0.2},
        'tv': None,                   # osem-tv options, e.g. {"a": 3.0, "wz": 1.0}
    },
}
COMPARISON = [                        # (panel title, run name) for the README figures
    ('MLEM (clean data)', 'clean | MLEM voxel'),
    ('Cluttered, no handling', 'clutter | MLEM voxel'),
    ('Cluttered, vanilla MLEM', 'clutter | vanilla MLEM'),
    ('Edge handling', 'clutter | MLEM voxel + edge'),
    ('+ extrapolation', 'clutter | MLEM voxel + edge + extrap'),
    ('+ OS-LALM (hyperbola)', 'clutter | OS-LALM-10 hyperbola + edge + extrap'),
    ('+ OS-LALM (TV)', 'clutter | OS-LALM-2 TV + edge + extrap'),
]
# ──────────────────────────────────────────────────────────────────────────────


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(argv=None) -> dict:
    """DEFAULT_CONFIG <- JSON file (--config) <- command-line flags."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', help='JSON config file (see examples/configs/)')
    ap.add_argument('--mode', choices=['recommended', 'compare'])
    ap.add_argument('--input', dest='input_path', help='DICOM directory')
    ap.add_argument('--out-dir', dest='out_dir')
    ap.add_argument('--plane', choices=['axial', 'coronal', 'sagittal'])
    ap.add_argument('--n-iter', dest='n_iter', type=int)
    ap.add_argument('--variant', choices=['mlem', 'osem', 'osem-tv', 'os-lalm', 'vanilla'])
    ap.add_argument('--backprojector', choices=['voxel', 'ray', 'footprint'])
    ap.add_argument('--subsets', dest='n_subsets', type=int)
    ap.add_argument('--prior', choices=['hyperbola', 'tv'], help='OS-LALM prior')
    ap.add_argument('--no-clutter', dest='add_clutter', action='store_false', default=None)
    ap.add_argument('--no-extrapolation', dest='extrap_off', action='store_true')
    ap.add_argument('--projector-study', dest='projector_study', action='store_true', default=None)
    ap.add_argument('--show', action='store_true', default=None)
    ap.add_argument('--print-config', action='store_true', help='print the effective config and exit')
    a = ap.parse_args(argv)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if a.config:
        with open(a.config) as f:
            _deep_update(cfg, json.load(f))
    for k in ('mode', 'input_path', 'out_dir', 'plane', 'n_iter', 'add_clutter', 'projector_study', 'show'):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    rc = cfg['reconstruction']
    if a.variant:
        rc['variant'] = a.variant
    if a.backprojector:
        rc['backprojector'] = a.backprojector
    if a.n_subsets:
        rc['n_subsets'] = a.n_subsets
    if a.prior:
        rc.setdefault('lalm', {})['prior'] = a.prior
    if a.extrap_off:
        cfg['extrapolation']['enabled'] = False
    if a.print_config:
        print(json.dumps(cfg, indent=2))
        sys.exit(0)
    return cfg


def setup_plane(ct_parser: DicomCTParser, plane: str):
    """Identical volume orientation / extents / source geometry as x_ray_simulator_example.py."""
    if plane == 'axial':
        ct_parser.transpose((2, 1, 0))
        ext = (500.0, 500.0, 200.0)
        z_off = 185.0
        TSD = np.array([-1800 * np.tan(np.radians(20)), 0, 1800], dtype=np.float32)
    elif plane == 'coronal':
        ct_parser.flip_y()
        ct_parser.transpose((0, 2, 1))
        ext = (530.0, 300.0, 900.0)
        z_off = 0.0
        TSD = np.array([-1500 * np.tan(np.radians(15)), 0, 1500], dtype=np.float32)
    elif plane == 'sagittal':
        ct_parser.transpose((1, 0, 2))
        ext = (200.0, 500.0, 500.0)
        z_off = 0.0
        TSD = np.array([0, 0, 1500], dtype=np.float32)
    else:
        raise ValueError(f'Unknown projection plane: {plane}')
    oxb, oyb, ozb = ext
    xsb = np.array([-oxb / 2, oxb / 2], dtype=np.float32)
    ysb = np.array([-oyb / 2, oyb / 2], dtype=np.float32)
    zsb = np.array([-ozb / 2, ozb / 2], dtype=np.float32) + z_off
    TDD = np.array([0, 0, 0], dtype=np.float32)
    return np.ascontiguousarray(ct_parser.get_volume(), dtype=np.float32), xsb, ysb, zsb, TSD, TDD


def add_clutter(sino: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthetic edge clutter on (nproj, nv, nu) projections.

    * bright collimator-shadow / scatter band on the left & right border whose
      width follows the source position, with an exponential tail
    * dead (zero) rows at the top, saturated rows at the bottom
    * sparse hot pixels and signal dependent noise everywhere
    """
    out = sino.copy()
    nproj, nv, nu = sino.shape
    vmax = float(sino.max())
    u = np.arange(nu, dtype=np.float32)
    for p in range(nproj):
        phase = p / max(nproj - 1, 1)
        wl = int(round(nu * (0.015 + 0.025 * phase)))        # band width moves with the source
        wr = int(round(nu * (0.040 - 0.025 * phase)))
        band = np.zeros(nu, np.float32)
        band[:wl] = 1.0
        band[nu - wr:] = 1.0
        decay = 0.006 * nu                                    # exponential scatter tail
        tail = np.exp(-(u - wl) / decay) * (u >= wl) + np.exp(-((nu - wr) - u) / decay) * (u < nu - wr)
        band = np.maximum(band, 0.6 * tail)
        level = vmax * (1.2 + 0.3 * rng.random())
        out[p] += level * band[None, :] * (1 + 0.05 * rng.standard_normal((nv, 1))).astype(np.float32)
        out[p, :int(0.008 * nv)] = 0.0                       # dead rows
        out[p, nv - int(0.005 * nv):] = 1.5 * vmax            # saturated rows
        hot = rng.random((nv, nu)) < 2e-4
        out[p][hot] = 2.0 * vmax
    noise = rng.standard_normal(out.shape).astype(np.float32) * (0.01 * vmax) * np.sqrt(np.clip(out, 0, None) / vmax)
    return np.clip(out + noise, 0, None).astype(np.float32)


def _interp_matrix(src_c: np.ndarray, dst_c: np.ndarray) -> np.ndarray:
    """Linear interpolation matrix M (len(dst) x len(src)), zero outside."""
    m = np.zeros((len(dst_c), len(src_c)), np.float32)
    idx = np.interp(dst_c, src_c, np.arange(len(src_c)), left=-1, right=-1)
    for i, f in enumerate(idx):
        if f < 0:
            continue
        j = int(np.floor(f))
        a = f - j
        m[i, j] += 1 - a
        if a > 0 and j + 1 < len(src_c):
            m[i, j + 1] += a
    return m


def reference_on_grid(vol, xs, ys, coordz_src, zv_src, geom: TomoGeometry):
    """Ground truth on the reconstruction grid: slab average in z, linear in x/y."""
    vz, vy, vx = vol.shape
    xv_s, yv_s = (xs[1] - xs[0]) / vx, (ys[1] - ys[0]) / vy
    xv, yv, zv = geom.voxel_size
    mx = _interp_matrix(xs[0] + np.arange(vx) * xv_s, geom.xs[0] + np.arange(geom.vx) * xv)
    my = _interp_matrix(ys[0] + np.arange(vy) * yv_s, geom.ys[0] + np.arange(geom.vy) * yv)
    ref = np.zeros(geom.vol_shape, np.float32)
    for iz, c in enumerate(geom.coordz):
        sel = np.abs(coordz_src - c) <= 0.5 * zv
        if not sel.any():
            continue
        slab = vol[sel].mean(axis=0)
        ref[iz] = my @ slab @ mx.T
    return ref


def region_masks(geom: TomoGeometry):
    """'all', 'center' (inner 60 % in x/y) and 'edge' (outer 12 % band in x/y)."""
    vz, vy, vx = geom.vol_shape
    fx = np.abs((np.arange(vx) + 0.5) / vx * 2 - 1)
    fy = np.abs((np.arange(vy) + 0.5) / vy * 2 - 1)
    r = np.maximum(fx[None, :], fy[:, None])
    center = np.broadcast_to(r < 0.6, (vz, vy, vx))
    edge = np.broadcast_to(r > 0.88, (vz, vy, vx))
    return {'all': np.ones((vz, vy, vx), bool), 'center': center, 'edge': edge}


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def save_readme_figures(ref, results, iz, misc_dir):
    """Side-by-side slice comparison (PNG) and depth sweep (GIF) for the README."""
    import imageio
    os.makedirs(misc_dir, exist_ok=True)
    panels = [('Ground truth', ref)] + [(title, results[key]['volume']) for title, key in COMPARISON]
    vmax = float(np.percentile(ref[iz], 99.5))

    def draw(z, dpi):
        fig, axs = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.5), dpi=dpi)
        for a, (title, vol) in zip(axs, panels):
            a.imshow(vol[z], cmap='gray', vmin=0, vmax=vmax)
            a.set_title(title, fontsize=9)
            a.axis('off')
        fig.suptitle(f'coronal slab {z + 1}/{ref.shape[0]}', fontsize=10)
        fig.tight_layout()
        return fig

    # side-by-side comparison of one reconstructed slab
    fig = draw(iz, dpi=120)
    fig.savefig(os.path.join(misc_dir, 'mlem_comparison.png'))
    plt.close(fig)

    # depth sweep through the slabs that contain the object
    zs = [z for z in range(ref.shape[0]) if ref[z].mean() > 0.25 * ref.mean(axis=(1, 2)).max()]
    frames = []
    for z in zs:
        fig = draw(z, dpi=70)
        frames.append(_fig_to_rgb(fig))
        plt.close(fig)
    imageio.mimsave(os.path.join(misc_dir, 'mlem_reconstruction.gif'), frames, duration=0.25, loop=0)
    logger.info(f'Saved {misc_dir}/mlem_comparison.png and mlem_reconstruction.gif ({len(frames)} frames)')


def projector_study(volume_np, xsb, ysb, coordz, zv_src, geom, sino_clean, misc_dir, n_iter):
    """ray / voxel / footprint back-projectors on a grid finer than the detector.

    With voxel size = detector pitch the footprint (box-overlap) weights reduce
    to the bilinear ones, so the projectors only differ when the sampling is
    mismatched; here the in-plane voxel is half the (binned) pixel pitch.
    """
    fine = TomoGeometry(**{k: getattr(geom, k) for k in geom.__dataclass_fields__})
    fine.vx, fine.vy = 2 * geom.vx, 2 * geom.vy
    ref = reference_on_grid(volume_np, xsb, ysb, coordz, zv_src, fine)
    rec = MLEMReconstructor(fine)
    rec.set_data(sino_clean)
    rec.set_reference(ref, region_masks(fine))
    rec.set_edge_handling(EdgeHandling.none())
    iz = int(np.argmax(ref.reshape(ref.shape[0], -1).mean(1)))
    cy, cx = fine.vy // 2, fine.vx // 2
    crop = (slice(cy - 140, cy + 140), slice(cx + 100, cx + 460))
    out, crops = {}, {'ground truth': ref[iz][crop]}
    for bp in ('ray', 'voxel', 'footprint'):
        r = rec.run(n_iter=n_iter, backprojector=bp, verbose=False)
        sl = r.volume[iz][crop]
        rough = float(np.abs(np.diff(sl, axis=1)).mean() / sl.mean())
        f = r.history[-1]
        out[bp] = dict(adjoint=rec.op.adjoint_test(bp), seconds=r.seconds, nrmse_all=f['nrmse_all'],
                       nrmse_center=f['nrmse_center'], nrmse_edge=f['nrmse_edge'], roughness=rough)
        crops[f'MLEM {bp}'] = sl
        logger.info(f'projector study {bp}: {out[bp]}')
    vmax = float(np.percentile(crops['ground truth'], 99.5))
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, (title, im) in zip(axs, crops.items()):
        a.imshow(im, cmap='gray', vmin=0, vmax=vmax)
        extra = '' if title == 'ground truth' else f"\nroughness {out[title.split()[-1]]['roughness']:.4f}"
        a.set_title(title + extra, fontsize=9)
        a.axis('off')
    fig.suptitle('voxel = detector pitch / 2 (zoom, slab %d)' % (iz + 1), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(misc_dir, 'projector_comparison.png'), dpi=100)
    plt.close(fig)
    del rec
    return out


def simulate(cfg):
    """CT volume, acquisition geometry and simulated (clean / cluttered) projections."""
    rng = np.random.default_rng(cfg['seed'])
    ct_parser = DicomCTParser(path=cfg['input_path'], order_by_slice_loc=True)
    ct_parser.normalize()
    volume_np, xsb, ysb, zsb, TSD, TDD = setup_plane(ct_parser, cfg['plane'])
    vzb, vyb, vxb = volume_np.shape
    oxb, oyb, ozb = xsb[1] - xsb[0], ysb[1] - ysb[0], zsb[1] - zsb[0]
    coordz = np.linspace(zsb[0], zsb[1], vzb, dtype=np.float32)
    su = sv = 440.0
    npx, nproj, b = cfg['detector_px'], cfg['num_projections'], cfg['bin']
    src_max = TSD[2] * np.tan(np.radians(15))
    source = np.linspace(-src_max, src_max, nproj, dtype=np.float32)
    distance = 150.0
    detector = (source * (-distance / (TSD[2] - distance))).astype(np.float32)

    # projections with the existing simulator (mode 2, texture trilinear)
    projector = Projector(xsb, ysb, zsb, vxb, vyb, vzb, oxb / vxb, oyb / vyb, ozb / vzb,
                          TSD, TDD, su, sv, npx, npx, nproj, source, detector, coordz, mode=2)
    sino_full = projector.project(volume_np)
    logger.info(f'Simulated projections: {sino_full.shape}, max {sino_full.max():.2f}')
    sino_clean = bin_projections(sino_full, b)
    sino_meas = bin_projections(add_clutter(sino_full, rng), b) if cfg['add_clutter'] else sino_clean

    # reconstruction geometry: in-plane voxel ~ detector pitch
    nu = npx // b
    du = su / nu
    geom = TomoGeometry(xs=xsb, ys=ysb, zs=zsb, vx=int(round(oxb / du)), vy=int(round(oyb / du)),
                        vz=cfg['recon_vz'], TSD=TSD, TDD=TDD, su=su, sv=sv, nu=nu, nv=nu,
                        source=source, detector=detector)
    logger.info(f'Recon grid (vz, vy, vx) = {geom.vol_shape}, voxel {np.round(geom.voxel_size, 3)} mm')
    ref = reference_on_grid(volume_np, xsb, ysb, coordz, ozb / vzb, geom)
    return dict(volume=volume_np, xs=xsb, ys=ysb, coordz=coordz, zv_src=ozb / vzb, geom=geom,
                clean=sino_clean, meas=sino_meas, ref=ref, masks=region_masks(geom))


def build_reconstructor(sim, sino, edge: EdgeHandling, extrap: dict, op=None):
    """Reconstructor with edge handling and (optionally) the virtually extended detector."""
    geom = sim['geom']
    rec = MLEMReconstructor(geom, op)
    rec.set_data(sino)
    w = rec.set_edge_handling(edge)
    if extrap and extrap.get('enabled'):
        pad = int(extrap['pad_px'])
        geom_ext, y_ext, w_ext, _ = extend_detector(sino, w, geom, pad_px=pad,
                                                    shape=extrap['shape'], w_ext=extrap['w_ext'])
        rec = MLEMReconstructor(geom_ext)
        rec.set_data(y_ext)
        rec.set_weights(w_ext, edge)
    rec.set_reference(sim['ref'], sim['masks'])
    return rec


def run_recommended(cfg, sim):
    """Only the configured pipeline: save the volume, an overview figure and the metrics."""
    rc = cfg['reconstruction']
    edge = EdgeHandling(**cfg['edge_handling'])
    rec = build_reconstructor(sim, sim['meas'], edge, cfg['extrapolation'])
    r = rec.run(n_iter=cfg['n_iter'], variant=rc['variant'], backprojector=rc['backprojector'],
                n_subsets=rc.get('n_subsets', 1), lalm=rc.get('lalm'), tv=rc.get('tv'), verbose=True)
    f = r.history[-1]
    logger.success(f"{rc['variant']}/{rc['backprojector']}: NRMSE all/center/edge = "
                   f"{f['nrmse_all']:.4f}/{f['nrmse_center']:.4f}/{f['nrmse_edge']:.4f} in {r.seconds:.1f}s")
    out = cfg['out_dir']
    np.save(os.path.join(out, 'reconstruction.npy'), r.volume)
    with open(os.path.join(out, 'metrics_recommended.json'), 'w') as fh:
        json.dump(dict(config=cfg, seconds=r.seconds, history=r.history), fh, indent=1, default=float)
    ref = sim['ref']
    zs = np.linspace(0.2, 0.8, 4) * (ref.shape[0] - 1)
    fig, axs = plt.subplots(2, 4, figsize=(16, 5.5))
    vmax = float(np.percentile(ref, 99.5))
    for i, z in enumerate(zs.astype(int)):
        axs[0, i].imshow(ref[z], cmap='gray', vmin=0, vmax=vmax)
        axs[0, i].set_title(f'ground truth slab {z + 1}', fontsize=9)
        axs[1, i].imshow(r.volume[z], cmap='gray', vmin=0, vmax=vmax)
        axs[1, i].set_title(f"{rc['variant']} slab {z + 1}", fontsize=9)
    for a in axs.ravel():
        a.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'reconstruction_slabs.png'), dpi=100)
    plt.close(fig)
    if cfg['show']:
        visu.show_slices(r.volume.transpose(1, 2, 0), pause_t=0.22)
    logger.success(f'Done, results in {out}')


def save_os_lalm_figure(ref, results, iz, misc_dir):
    """Zoomed comparison of the regularised solvers + their NRMSE convergence."""
    keys = [('MLEM + edge + extrap', 'clutter | MLEM voxel + edge + extrap'),
            ('OSEM-TV-5 (Sidky)', 'clutter | OSEM-TV-5 + edge + extrap'),
            ('OS-LALM-10 hyperbola', 'clutter | OS-LALM-10 hyperbola + edge + extrap'),
            ('OS-LALM-2 TV', 'clutter | OS-LALM-2 TV + edge + extrap')]
    cy, cx = ref.shape[1] // 2, ref.shape[2] // 2
    crop = (slice(cy - 180, cy + 180), slice(cx - 330, cx + 330))
    vmax = float(np.percentile(ref[iz], 99.5))
    fig = plt.figure(figsize=(20, 7.5), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1])
    panels = [('ground truth', ref)] + [(t, results[k]['volume']) for t, k in keys]
    for i, (a, (t, vol)) in enumerate(zip(top.subplots(1, len(panels)), panels)):
        a.imshow(vol[iz][crop], cmap='gray', vmin=0, vmax=vmax)
        extra = '' if t == 'ground truth' else f"\nNRMSE {results[keys[i - 1][1]]['final']['nrmse_all']:.3f}"
        a.set_title(t + extra, fontsize=10)
        a.axis('off')
    for j, (a, (metric, title)) in enumerate(zip(bottom.subplots(1, 3), [
            ('nrmse_all', 'NRMSE whole volume'), ('nrmse_center', 'NRMSE central ROI'),
            ('nrmse_edge', 'NRMSE edge band')])):
        for t, k in keys:
            h = results[k]['history']
            a.plot([m['iter'] for m in h], [m[metric] for m in h], label=t)
        a.set_title(title, fontsize=10)
        a.set_xlabel('iteration')
        a.grid(alpha=0.3)
        if j == 0:
            a.legend(fontsize=8)
    fig.savefig(os.path.join(misc_dir, 'os_lalm_comparison.png'), dpi=90)
    plt.close(fig)


def run_compare(cfg, sim):
    """All variants side by side + README figures."""
    geom, ref, masks = sim['geom'], sim['ref'], sim['masks']
    sino_clean, sino_meas = sim['clean'], sim['meas']
    n_iter, out, misc = cfg['n_iter'], cfg['out_dir'], cfg['misc_dir']
    os.makedirs(misc, exist_ok=True)
    op = GpuTomoOperator(geom)
    adj = {m: op.adjoint_test(m) for m in ('ray', 'voxel', 'footprint')}
    logger.info(f'Adjoint test |<Ax,y>-<x,By>|/|<Ax,y>|: {adj}')

    proposed = EdgeHandling(**cfg['edge_handling'])
    none = EdgeHandling.none()
    extrap = cfg['extrapolation']
    hyp = dict(prior='hyperbola', beta=1.0, delta=0.2)
    tv = dict(prior='tv', tv_lambda=0.01, tv_wz=1.0, tv_inner=50)
    runs = [
        # name,                                  data,   variant,   bp,     subsets, edge, extrapolate, options
        ('clean | MLEM ray',                     'clean', 'mlem',    'ray',       1, none, False, {}),
        ('clean | MLEM voxel',                   'clean', 'mlem',    'voxel',     1, none, False, {}),
        ('clean | MLEM footprint',               'clean', 'mlem',    'footprint', 1, none, False, {}),
        ('clean | OSEM-5 voxel',                 'clean', 'osem',    'voxel',     5, none, False, {}),
        ('clean | vanilla MLEM',                 'clean', 'vanilla', 'voxel',     1, none, False, {}),
        ('clutter | MLEM voxel',                 'meas',  'mlem',    'voxel',     1, none, False, {}),
        ('clutter | vanilla MLEM',               'meas',  'vanilla', 'voxel',     1, none, False, {}),
        ('clutter | MLEM voxel + crop40',        'meas',  'mlem',    'voxel',     1, EdgeHandling(border_px=40), False, {}),
        ('clutter | MLEM voxel + edge',          'meas',  'mlem',    'voxel',     1, proposed, False, {}),
        ('clutter | OSEM-5 voxel + edge',        'meas',  'osem',    'voxel',     5, proposed, False, {}),
        ('clutter | MLEM voxel + edge + extrap', 'meas',  'mlem',    'voxel',     1, proposed, True, {}),
        ('clutter | OSEM-TV-5 + edge + extrap',  'meas',  'osem-tv', 'voxel',     5, proposed, True,
         dict(tv=dict(a=3.0, wz=1.0))),
        ('clutter | OS-LALM-10 hyperbola + edge + extrap', 'meas', 'os-lalm', 'voxel', 10, proposed, True, dict(lalm=hyp)),
        ('clutter | OS-LALM-2 TV + edge + extrap', 'meas', 'os-lalm', 'voxel', 2, proposed, True, dict(lalm=tv)),
    ]
    rec = MLEMReconstructor(geom, op)
    rec.set_reference(ref, masks)
    rec_ext = None
    clean_mask = np.zeros(geom.sino_shape, bool)
    b = int(0.06 * geom.nu)
    clean_mask[:, b:-b, b:-b] = True
    results = {}
    for name, data, variant, bp, nsub, edge, ext, opts in runs:
        logger.info(f'===== {name} =====')
        if ext:
            if rec_ext is None:
                rec_ext = build_reconstructor(sim, sino_meas, proposed, dict(extrap, enabled=True))
            run_rec, w = rec_ext, rec_ext.w.get()[:, :, extrap['pad_px']:-extrap['pad_px']]
        else:
            rec.set_data(sino_clean if data == 'clean' else sino_meas)
            w = rec.set_edge_handling(edge)
            run_rec = rec
        r = run_rec.run(n_iter=n_iter, variant=variant, backprojector=bp, n_subsets=nsub, verbose=False, **opts)
        # projection error against the *clean* projections (inner detector region)
        q = op.new_sino()
        op.forward(gpuarray.to_gpu(r.volume), q, method=op.projector_for(bp))
        qh = q.get()
        perr = float(np.linalg.norm((qh - sino_clean)[clean_mask]) / np.linalg.norm(sino_clean[clean_mask]))
        last = r.history[-1]
        best = min(r.history, key=lambda m: m['nrmse_all'])
        results[name] = dict(variant=variant, backprojector=bp, subsets=nsub, data=data,
                             edge=edge.__dict__, seconds=r.seconds, proj_err_clean=perr,
                             final=last, best_iter=best['iter'], best_nrmse_all=best['nrmse_all'],
                             history=r.history, volume=r.volume, weights=w[0])
        logger.info(f"{name}: NRMSE all/center/edge = {last['nrmse_all']:.4f}/{last['nrmse_center']:.4f}/"
                    f"{last['nrmse_edge']:.4f}  proj-err(clean) = {perr:.4f}  time {r.seconds:.1f}s")

    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ('volume', 'weights')} for k, v in results.items()}
    summary['_adjoint_test'] = adj
    summary['_config'] = cfg

    print('\n{:<48s} {:>7s} {:>8s} {:>8s} {:>8s} {:>9s} {:>6s} {:>7s}'.format(
        'variant', 'time[s]', 'NRMSE', 'center', 'edge', 'projErr', 'best@', 'bestNR'))
    for k, v in results.items():
        f_ = v['final']
        print('{:<48s} {:7.1f} {:8.4f} {:8.4f} {:8.4f} {:9.4f} {:6d} {:7.4f}'.format(
            k, v['seconds'], f_['nrmse_all'], f_['nrmse_center'], f_['nrmse_edge'],
            v['proj_err_clean'], v['best_iter'], v['best_nrmse_all']))

    # convergence plots
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for k, v in results.items():
        it = [m['iter'] for m in v['history']]
        ax[0].plot(it, [m['nrmse_all'] for m in v['history']], label=k)
        ax[1].plot(it, [m['nrmse_center'] for m in v['history']], label=k)
        ax[2].plot(it, [m['nrmse_edge'] for m in v['history']], label=k)
        ax[3].semilogy(it, [m['kl'] for m in v['history']], label=k)
    for a, t in zip(ax, ['NRMSE whole volume', 'NRMSE central ROI', 'NRMSE edge band', 'weighted KL (data term)']):
        a.set_title(t)
        a.set_xlabel('iteration')
        a.grid(alpha=0.3)
    ax[3].legend(fontsize=6)
    ax[2].set_ylim(0, 2.5)
    ax[0].set_ylim(0.25, 1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'convergence.png'), dpi=110)
    plt.close(fig)

    # weight map of the proposed edge handling
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(sino_meas[0], cmap='gray')
    ax[0].set_title('measured projection 0')
    ax[1].imshow(results['clutter | MLEM voxel + edge']['weights'], cmap='viridis', vmin=0, vmax=1)
    ax[1].set_title('ray weights w (proposed)')
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'edge_weights.png'), dpi=110)
    plt.close(fig)

    iz = int(np.argmax(ref.reshape(ref.shape[0], -1).mean(1)))   # slab with the most tissue
    save_readme_figures(ref, results, iz, misc)
    save_os_lalm_figure(ref, results, iz, misc)

    if cfg['projector_study']:
        study = projector_study(sim['volume'], sim['xs'], sim['ys'], sim['coordz'], sim['zv_src'],
                                geom, sino_clean, misc, n_iter)
        summary['_projector_study'] = study
        print('\nProjector study (voxel = pixel/2, clean data, MLEM)')
        print('{:<10s} {:>9s} {:>8s} {:>8s} {:>8s} {:>10s} {:>8s}'.format(
            'bp', 'adjoint', 'NRMSE', 'center', 'edge', 'roughness', 'time[s]'))
        for k, v in study.items():
            print('{:<10s} {:9.1e} {:8.4f} {:8.4f} {:8.4f} {:10.4f} {:8.1f}'.format(
                k, v['adjoint'], v['nrmse_all'], v['nrmse_center'], v['nrmse_edge'], v['roughness'], v['seconds']))
    with open(os.path.join(out, 'metrics.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, default=float)
    logger.success(f'Done, results in {out}')


def main(argv=None):
    cfg = load_config(argv)
    os.makedirs(cfg['out_dir'], exist_ok=True)
    sim = simulate(cfg)
    if cfg['mode'] == 'compare':
        run_compare(cfg, sim)
    else:
        run_recommended(cfg, sim)


if __name__ == '__main__':
    main()
