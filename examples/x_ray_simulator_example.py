import numpy as np

from utils.Logger import Logger
from utils.DicomCTParser import DicomCTParser
from utils.VolumeVisualizer import VolumeVisualizer as visu
from utils.CudaWrapper import Projector,Backprojector
import utils.TomoFilter

logger = Logger().get_logger()

if __name__ == '__main__':

    # ── Choose projection plane ──────────────────────────────────────────────
    # 'axial'   : source/detector move in X, integrate along Z (sup-inf)
    # 'coronal' : source/detector move in X, integrate along Y (ant-post)
    # 'sagittal' : source/detector move in Z, integrate along X (left-right)
    PROJECTION_PLANE = 'coronal'#'coronal'#'sagittal'
    # ────────────────────────────────────────────────────────────────────────

    input_path = './data/Lung-Pet-CT-Dx/Lung_Dx-A0018/07-24-2008-lungc-69037/301.000000-1.25mm-74521/1' # Update this path to your DICOM directory
    logger.info(f"Reading dicom files from: {input_path}")
    ct_parser = DicomCTParser(path=input_path, order_by_slice_loc=True)
    ct_parser.normalize()

    # Raw volume after loading: (rows, cols, slices) = (512, 512, 237)
    #   axis 0 = rows   → patient Y  (anterior-posterior)
    #   axis 1 = cols   → patient X  (left-right)
    #   axis 2 = slices → patient Z  (superior-inferior)

    if PROJECTION_PLANE == 'axial':
        # Kernel needs memory layout (vz, vy, vx) where vz is the integration axis.
        # Integration axis = patient Z  → transpose (slices, cols, rows) = (Z, X, Y)
        ct_parser.transpose((2, 1, 0))          # (237, 512, 512) = (vz, vy, vx)
        volume_np = ct_parser.get_volume()
        vzb, vyb, vxb = volume_np.shape         # (237, 512, 512)

        # Physical extents  [mm]
        oxb = 500.0   # X physical size
        oyb = 500.0   # Y physical size
        ozb = 200.0   # Z physical size  (237 slices × ~0.84 mm)

        xsb = np.array([-oxb/2,  oxb/2], dtype=np.float32)
        ysb = np.array([-oyb/2,  oyb/2], dtype=np.float32)
        zsb = np.array([-ozb/2,  ozb/2], dtype=np.float32) + 185   # patient offset

        # Source at +Z, detector at origin; source/detector sweep in X
        TSD = np.array([-1800*np.tan(np.radians(20)), 0, 1800], dtype=np.float32)
        TDD = np.array([0, 0, 0], dtype=np.float32)

        # coordz spans the Z extent (integration axis)
        coordz = np.linspace(zsb[0], zsb[1], vzb, dtype=np.float32)

    elif PROJECTION_PLANE == 'coronal':
        # Integration axis = patient Y  (rows, axis 0 of raw volume).
        # Remap so that the kernel's vz-loop runs over Y slices:
        #   raw (rows=512, cols=512, slices=237)
        #   transpose (rows, slices, cols) → (vz=512, vy=237, vx=512)
        #                                      Y        Z       X
        ct_parser.flip_y()
        ct_parser.transpose((0, 2, 1))          # (512, 237, 512) = (vz=Y, vy=Z, vx=X)
        volume_np = ct_parser.get_volume()
        vzb, vyb, vxb = volume_np.shape         # (512, 237, 512)

        # Physical extents  [mm]
        oxb = 530.0   # X physical size  (vx axis)
        oyb = 300.0   # Z physical size  (vy axis, was patient Z)
        ozb = 900.0   # Y physical size  (vz / integration axis)

        # xsb / ysb bound the detector-plane axes (X and Z in patient space)
        xsb = np.array([-oxb/2,  oxb/2], dtype=np.float32)
        ysb = np.array([-oyb/2,  oyb/2], dtype=np.float32)
        # zsb bounds the integration axis (patient Y, anterior-posterior)
        zsb = np.array([-ozb/2,  ozb/2], dtype=np.float32)

        # Source at +Y, detector at origin; source/detector sweep in X.
        # TSD.z is the source distance along the integration axis (patient Y here).
        TSD = np.array([-1500*np.tan(np.radians(15)), 0, 1500], dtype=np.float32)
        TDD = np.array([0, 0, 0], dtype=np.float32)

        # coordz spans the Y extent (integration axis)
        coordz = np.linspace(zsb[0], zsb[1], vzb, dtype=np.float32)

    elif PROJECTION_PLANE == 'sagittal':
        # Integration axis = patient X (cols, axis 1 of raw volume).
        # Remap so that the kernel's vz-loop runs over X slices:
        #   raw (rows=512, cols=512, slices=237) → (cols, rows, slices) = (vz=X, vy=Y, vx=Z)
        ct_parser.transpose((1, 0, 2))  # (512, 512, 237) = (vz=X, vy=Y, vx=Z)
        volume_np = ct_parser.get_volume()
        vzb, vyb, vxb = volume_np.shape  # (512, 512, 237)

        # Physical extents [mm]
        oxb = 200.0  # Z physical size (vx axis)
        oyb = 500.0  # Y physical size (vy axis)
        ozb = 500.0  # X physical size (vz / integration axis)

        # xsb / ysb bound the detector-plane axes (Y and Z in patient space)
        xsb = np.array([-oxb / 2, oxb / 2], dtype=np.float32)
        ysb = np.array([-oyb / 2, oyb / 2], dtype=np.float32)
        # zsb bounds the integration axis (patient X, left-right)
        zsb = np.array([-ozb / 2, ozb / 2], dtype=np.float32)

        # Source at +X, detector at origin; source/detector sweep in Z.
        TSD = np.array([0, 0, 1500], dtype=np.float32)  # Source at +X (integration axis)
        TDD = np.array([0, 0, 0], dtype=np.float32)

        # coordz spans the X extent (integration axis)
        coordz = np.linspace(zsb[0], zsb[1], vzb, dtype=np.float32)


    else:
        raise ValueError(f"Unknown PROJECTION_PLANE: {PROJECTION_PLANE}")

    volume_dims = volume_np.shape
    logger.info(f"Volume dimensions (vz, vy, vx): {volume_dims}")

    # Voxel sizes for CT sinograms
    xvb = oxb / vxb
    yvb = oyb / vyb
    zvb = ozb / vzb

    # Detector size and resolution
    su = 440.0
    sv = 440.0
    nu = 1512
    nv = 1512

    num_projections = 20
    source = np.linspace(-1500*np.tan(np.radians(15)), 1500*np.tan(np.radians(15)),
                         num_projections, dtype=np.float32)
    logger.info(f"Size of source array: {source.shape}")

    distance = 150.0
    detector = source * (-distance / (1500 - distance))
    logger.info(f"Size of detector array: {detector.shape}")
    logger.info(f"Size of coordz: {coordz.shape}")

    projector = Projector(xsb, ysb, zsb,
                          vxb, vyb, vzb,
                          xvb, yvb, zvb,
                          TSD, TDD,
                          su, sv,
                          nu, nv,
                          num_projections,
                          source,
                          detector, coordz, mode=2)

    res = projector.project(volume_np)

    # res shape: (nproj, nv, nu) — each res[i] is one detector image
    logger.info(f"Result shape: {res.shape}")

    filt_res = np.zeros_like(res)

    for i in range(res.shape[0]):
        filt_res[i,:,:], _, _ = utils.TomoFilter.tomo_filter(res[i,:,:], nu, nv, num_projections, su, sv, d=1, filter_type='hamming')


    # Backprojection resolution
    vzb, vyb, vxb = (40,1512,1512)
    # coordz spans the Y extent (integration axis)
    coordz = np.linspace(zsb[0], zsb[1], vzb, dtype=np.float32)



    # Voxel sizes for reconstruction
    xvb = oxb / vxb
    yvb = oyb / vyb
    zvb = ozb / vzb

    backprojector = Backprojector(xsb, ysb, zsb,
                                xvb, yvb, zvb,
                                vxb, vyb,  vzb,
                                TSD, TDD,
                                su, sv,
                                nu, nv,
                                num_projections,
                                source,
                                detector, coordz, mode=2)
    recon = backprojector.backproject(filt_res)

    visu.show_slices(recon.transpose(1, 2, 0), pause_t=0.22)
    logger.success("Done..")
