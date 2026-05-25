import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import os
import ctypes
from utils.Logger import Logger
logger = Logger().get_logger()

class Projector:
    def __init__(self, xs : np.ndarray, ys : np.ndarray, zs : np.ndarray,
                 xv : int, yv : int, zv : int,
                 vx : float, vy : float, vz : float,
                 TSD : np.ndarray, TDD : np.ndarray,
                 su : float, sv : float,
                 nu : int, nv : int,
                 nproj: int,
                 source: np.ndarray,
                 detector: np.ndarray,
                 coordz: np.ndarray,
                 mode: int):
        # Construct the path to the CUDA kernel file dynamically
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cuda_file_path = os.path.join(current_dir, '..', 'cuda-files', 'projection.cu')
        
        try:
            with open(cuda_file_path, 'r') as f:
                f_content = f.read()
        except FileNotFoundError:
            logger.error(f"CUDA kernel file not found at: {os.path.abspath(cuda_file_path)}")
            raise FileNotFoundError(f"CUDA kernel file not found at: {os.path.abspath(cuda_file_path)}")

        self.kernel_template_code = f_content
        self.block_size = (32, 32, 1)
        self.X = int(np.ceil(nu / self.block_size[0]))
        self.Y = int(np.ceil(nproj * nv / self.block_size[1]))
        self.grid_size = (self.X, self.Y)
        self.module = SourceModule(self.kernel_template_code, arch='sm_80')

        # Result layout in kernel: result[x + yy*nu + z*nu*nv]  →  C-order shape is (nproj, nv, nu)
        logger.info(f"Allocation of GPU memory for projection results with size: {nu * nv * nproj * np.float32().nbytes}")
        self.result_gpu = cuda.mem_alloc(nu * nv * nproj * np.float32().nbytes)

        self.func = self.module.get_function("projection")

        # Keep as int for result shape; pass as float32 to kernel
        self.nu_int = int(nu)
        self.nv_int = int(nv)
        self.nproj_int = int(nproj)

        self.nu = np.float32(nu)
        self.nv = np.float32(nv)
        self.nproj = np.float32(nproj)

        self.xs_gpu = cuda.mem_alloc(xs.nbytes)
        cuda.memcpy_htod(self.xs_gpu, xs)
        self.ys_gpu = cuda.mem_alloc(ys.nbytes)
        cuda.memcpy_htod(self.ys_gpu, ys)
        self.zs_gpu = cuda.mem_alloc(zs.nbytes)
        cuda.memcpy_htod(self.zs_gpu, zs)

        self.xv = np.int32(xv)
        self.yv = np.int32(yv)
        self.zv = np.int32(zv)

        self.vx = np.float32(vx)
        self.vy = np.float32(vy)
        self.vz = np.float32(vz)

        self.TSD_gpu = cuda.mem_alloc(TSD.nbytes)
        cuda.memcpy_htod(self.TSD_gpu, TSD)
        self.TDD_gpu = cuda.mem_alloc(TDD.nbytes)
        cuda.memcpy_htod(self.TDD_gpu, TDD)

        self.su = np.float32(su)
        self.sv = np.float32(sv)

        self.source_gpu = cuda.mem_alloc(source.nbytes)
        cuda.memcpy_htod(self.source_gpu, source)
        self.detector_gpu = cuda.mem_alloc(detector.nbytes)
        cuda.memcpy_htod(self.detector_gpu, detector)
        self.coordz_gpu = cuda.mem_alloc(coordz.nbytes)
        cuda.memcpy_htod(self.coordz_gpu, coordz)
        self.mode = np.int32(mode)

        # NOTE: vx/vy/vz are voxel *sizes* (floats), NOT volume dimension counts.
        # Volume shape is read directly from input_volume.shape in project().

    # ------------------------------------------------------------------
    # Texture-object helpers (mode 2)
    # ------------------------------------------------------------------

    def _create_volume_texture(self, input_volume: np.ndarray):
        """
        Upload *input_volume* (shape vz×vy×vx, float32, C-contiguous) into a
        CUDA 3-D array and wrap it in a texture object with:
          - linear (trilinear) filtering
          - clamp-to-border addressing (out-of-bounds → 0)
          - non-normalised coordinates

        Returns (cuda_array, tex_obj_handle_as_uint64).
        The caller is responsible for destroying both after the kernel call.
        """
        vz, vy, vx = input_volume.shape  # depth, height, width

        # Allocate a pitched 3-D CUDA array (format: 32-bit float, 1 channel)
        descr = cuda.ArrayDescriptor3D()
        descr.width  = vx
        descr.height = vy
        descr.depth  = vz
        descr.format = cuda.array_format.FLOAT
        descr.num_channels = 1
        descr.flags = 0
        cuda_array = cuda.Array(descr)

        # Copy host volume → CUDA array
        copy_params = cuda.Memcpy3D()
        copy_params.set_src_host(input_volume)
        copy_params.set_dst_array(cuda_array)
        copy_params.width_in_bytes = vx * np.dtype(np.float32).itemsize
        copy_params.height         = vy
        copy_params.depth          = vz
        copy_params()

        # Build texture resource descriptor (cudaResourceDesc)
        # Layout: kind(4B) + pad(4B) + array_ptr(8B)  →  16 bytes minimum
        res_desc = np.zeros(16, dtype=np.uint8)
        # cudaResourceTypeArray = 0
        res_desc_view = res_desc.view(np.uint32)
        res_desc_view[0] = 0  # kind = cudaResourceTypeArray
        # store the array handle (pointer) at offset 8
        arr_ptr = int(cuda_array.handle)
        res_desc.view(np.uint64)[1] = arr_ptr

        # Build texture descriptor (cudaTextureDesc) — 64 bytes
        tex_desc = np.zeros(64, dtype=np.uint8)
        tex_desc_u32 = tex_desc.view(np.uint32)
        # addressMode[0..2] = cudaAddressModeBorder (2) → out-of-bounds returns 0
        tex_desc_u32[0] = 2  # addressMode[0] (x)
        tex_desc_u32[1] = 2  # addressMode[1] (y)
        tex_desc_u32[2] = 2  # addressMode[2] (z)
        # filterMode = cudaFilterModeLinear (1) → hardware trilinear interpolation
        tex_desc_u32[3] = 1
        # readMode = cudaReadModeElementType (0)
        tex_desc_u32[4] = 0
        # normalizedCoords = 0  (use absolute voxel coordinates)
        tex_desc_u32[5] = 0

        # Call cudaCreateTextureObject via ctypes
        libcudart = ctypes.CDLL('libcudart.so', use_errno=True)
        tex_obj = ctypes.c_uint64(0)
        ret = libcudart.cudaCreateTextureObject(
            ctypes.byref(tex_obj),
            res_desc.ctypes.data_as(ctypes.c_void_p),
            tex_desc.ctypes.data_as(ctypes.c_void_p),
            None  # no resource view descriptor
        )
        if ret != 0:
            raise RuntimeError(f"cudaCreateTextureObject failed with error code {ret}")

        logger.info(f"Created 3D texture object (handle={tex_obj.value}) for volume {vz}×{vy}×{vx}")
        return cuda_array, np.uint64(tex_obj.value)

    def _destroy_volume_texture(self, cuda_array, tex_handle: np.uint64):
        """Destroy the texture object and free the CUDA array."""
        libcudart = ctypes.CDLL('libcudart.so', use_errno=True)
        libcudart.cudaDestroyTextureObject(ctypes.c_uint64(int(tex_handle)))
        # cuda_array is freed when it goes out of scope (PyCUDA reference counting)

    # ------------------------------------------------------------------

    def project(self, input_volume):
        # Ensure the volume is C-contiguous float32 before copying to GPU.
        # np.transpose() returns a non-contiguous view which causes either
        # a ValueError (htod) or an illegal memory access (dtoh).
        input_volume = np.ascontiguousarray(input_volume, dtype=np.float32)

        # ---- texture object for mode 2 --------------------------------
        cuda_array = None
        tex_handle = None
        if int(self.mode) == 2:
            # input_volume is already shaped (vz, vy, vx) — use its shape directly.
            # Do NOT try to reshape using constructor vx/vy/vz floats (those are
            # voxel *sizes*, not dimension counts).
            if input_volume.ndim != 3:
                raise ValueError(
                    f"mode=2 requires a 3-D volume (vz, vy, vx), got shape {input_volume.shape}"
                )
            cuda_array, tex_handle = self._create_volume_texture(input_volume)
            tex_arg = tex_handle          # np.uint64 — passed as cudaTextureObject_t
        else:
            tex_arg = np.uint64(0)        # unused dummy for modes 0 / 1
        # ---------------------------------------------------------------

        input_gpu = cuda.mem_alloc(input_volume.nbytes)
        cuda.memcpy_htod(input_gpu, input_volume)

        self.func(self.result_gpu, input_gpu,
                  self.xs_gpu, self.ys_gpu, self.zs_gpu,
                  self.vx, self.vy, self.vz,
                  self.xv, self.yv, self.zv,
                  self.TSD_gpu, self.TDD_gpu,
                  self.su, self.sv,
                  self.nu, self.nv,
                  self.source_gpu, self.detector_gpu, self.coordz_gpu,
                  self.mode,
                  tex_arg,
                  block=self.block_size, grid=self.grid_size)

        # Kernel writes: result[x + yy*nu + z*nu*nv]
        # In C-order this is shape (nproj, nv, nu) — each projection is a (nv, nu) image.
        result = np.zeros((self.nproj_int, self.nv_int, self.nu_int), dtype=np.float32)
        logger.info(f"Size of result array: {result.nbytes} bytes")
        cuda.memcpy_dtoh(result, self.result_gpu)

        input_gpu.free()

        if cuda_array is not None:
            self._destroy_volume_texture(cuda_array, tex_handle)

        return result

class Backprojector:
    def __init__(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                 xv: int, yv: int, zv: int,
                 vx: float, vy: float, vz: float,
                 TSD: np.ndarray, TDD: np.ndarray,
                 su: float, sv: float,
                 nu: int, nv: int,
                 nproj: int,
                 source: np.ndarray,
                 detector: np.ndarray,
                 coordz: np.ndarray,
                 mode: int):
        # Construct the path to the CUDA kernel file dynamically
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cuda_file_path = os.path.join(current_dir, '..', 'cuda-files', 'backprojection.cu')

        try:
            with open(cuda_file_path, 'r') as f:
                f_content = f.read()
        except FileNotFoundError:
            logger.error(f"CUDA kernel file not found at: {os.path.abspath(cuda_file_path)}")
            raise FileNotFoundError(f"CUDA kernel file not found at: {os.path.abspath(cuda_file_path)}")

        self.kernel_code = f_content
        self.block_size = (32, 32, 1)
        # Grid covers vx columns × (vy * vz) rows (each row encodes a y+slice pair)
        self.X = int(np.ceil(vx / self.block_size[0]))
        self.Y = int(np.ceil(vy * vz / self.block_size[1]))
        self.grid_size = (self.X, self.Y)
        self.module = SourceModule(self.kernel_code, arch='sm_80')

        # Result layout: result[x + yy*vx + z*vx*vy]  →  C-order shape (vz, vy, vx)
        result_bytes = int(vx) * int(vy) * int(vz) * np.float32().nbytes
        logger.info(f"Allocation of GPU memory for backprojection results with size: {result_bytes}")
        self.result_gpu = cuda.mem_alloc(result_bytes)

        self.func = self.module.get_function("backprojection")

        # Volume dimension counts (int) and sizes (float32)
        self.vx_int = int(vx)
        self.vy_int = int(vy)
        self.vz_int = int(vz)

        self.xs_gpu = cuda.mem_alloc(xs.nbytes)
        cuda.memcpy_htod(self.xs_gpu, xs)
        self.ys_gpu = cuda.mem_alloc(ys.nbytes)
        cuda.memcpy_htod(self.ys_gpu, ys)
        self.zs_gpu = cuda.mem_alloc(zs.nbytes)
        cuda.memcpy_htod(self.zs_gpu, zs)

        self.xv = np.float32(xv)
        self.yv = np.float32(yv)
        self.zv = np.float32(zv)

        self.vx = np.int32(vx)
        self.vy = np.int32(vy)
        self.vz = np.int32(vz)

        self.TSD_gpu = cuda.mem_alloc(TSD.nbytes)
        cuda.memcpy_htod(self.TSD_gpu, TSD)
        self.TDD_gpu = cuda.mem_alloc(TDD.nbytes)
        cuda.memcpy_htod(self.TDD_gpu, TDD)

        self.su = np.float32(su)
        self.sv = np.float32(sv)
        self.nu = np.float32(nu)
        self.nv = np.float32(nv)
        self.nu_int = int(nu)
        self.nv_int = int(nv)
        self.nproj = np.int32(nproj)
        self.nproj_int = int(nproj)

        self.source_gpu = cuda.mem_alloc(source.nbytes)
        cuda.memcpy_htod(self.source_gpu, source)
        self.detector_gpu = cuda.mem_alloc(detector.nbytes)
        cuda.memcpy_htod(self.detector_gpu, detector)
        self.coordz_gpu = cuda.mem_alloc(coordz.nbytes)
        cuda.memcpy_htod(self.coordz_gpu, coordz)

        self.mode = np.int32(mode)

    # ------------------------------------------------------------------
    # Texture-object helpers (mode 2)
    # ------------------------------------------------------------------

    def _create_sinogram_texture(self, sinogram: np.ndarray):
        """
        Upload *sinogram* (shape nproj×nv×nu, float32, C-contiguous) into a
        CUDA 3-D array and wrap it in a texture object with:
          - linear (trilinear) filtering
          - clamp-to-border addressing (out-of-bounds → 0)
          - non-normalised coordinates

        Returns (cuda_array, tex_obj_handle_as_uint64).
        The caller is responsible for destroying both after the kernel call.
        """
        nproj, nv, nu = sinogram.shape

        descr = cuda.ArrayDescriptor3D()
        descr.width  = nu
        descr.height = nv
        descr.depth  = nproj
        descr.format = cuda.array_format.FLOAT
        descr.num_channels = 1
        descr.flags = 0
        cuda_array = cuda.Array(descr)

        copy_params = cuda.Memcpy3D()
        copy_params.set_src_host(sinogram)
        copy_params.set_dst_array(cuda_array)
        copy_params.width_in_bytes = nu * np.dtype(np.float32).itemsize
        copy_params.height         = nv
        copy_params.depth          = nproj
        copy_params()

        # Build texture resource descriptor (cudaResourceDesc)
        res_desc = np.zeros(16, dtype=np.uint8)
        res_desc.view(np.uint32)[0] = 0  # kind = cudaResourceTypeArray
        res_desc.view(np.uint64)[1] = int(cuda_array.handle)

        # Build texture descriptor (cudaTextureDesc) — 64 bytes
        tex_desc = np.zeros(64, dtype=np.uint8)
        tex_desc_u32 = tex_desc.view(np.uint32)
        tex_desc_u32[0] = 2  # addressMode[0] (x) = cudaAddressModeBorder
        tex_desc_u32[1] = 2  # addressMode[1] (y) = cudaAddressModeBorder
        tex_desc_u32[2] = 2  # addressMode[2] (z) = cudaAddressModeBorder
        tex_desc_u32[3] = 1  # filterMode = cudaFilterModeLinear
        tex_desc_u32[4] = 0  # readMode = cudaReadModeElementType
        tex_desc_u32[5] = 0  # normalizedCoords = 0

        libcudart = ctypes.CDLL('libcudart.so', use_errno=True)
        tex_obj = ctypes.c_uint64(0)
        ret = libcudart.cudaCreateTextureObject(
            ctypes.byref(tex_obj),
            res_desc.ctypes.data_as(ctypes.c_void_p),
            tex_desc.ctypes.data_as(ctypes.c_void_p),
            None
        )
        if ret != 0:
            raise RuntimeError(f"cudaCreateTextureObject failed with error code {ret}")

        logger.info(f"Created 3D sinogram texture object (handle={tex_obj.value}) for sinogram {nproj}×{nv}×{nu}")
        return cuda_array, np.uint64(tex_obj.value)

    def _destroy_sinogram_texture(self, cuda_array, tex_handle: np.uint64):
        """Destroy the texture object and free the CUDA array."""
        libcudart = ctypes.CDLL('libcudart.so', use_errno=True)
        libcudart.cudaDestroyTextureObject(ctypes.c_uint64(int(tex_handle)))

    # ------------------------------------------------------------------

    def backproject(self, sinogram: np.ndarray) -> np.ndarray:
        """
        Backproject *sinogram* (shape nproj×nv×nu, float32) into a volume.

        Returns a float32 ndarray of shape (vz, vy, vx).
        """
        sinogram = np.ascontiguousarray(sinogram, dtype=np.float32)

        # ---- texture object for mode 2 --------------------------------
        cuda_array = None
        tex_handle = None
        if int(self.mode) == 2:
            if sinogram.ndim != 3:
                raise ValueError(
                    f"mode=2 requires a 3-D sinogram (nproj, nv, nu), got shape {sinogram.shape}"
                )
            cuda_array, tex_handle = self._create_sinogram_texture(sinogram)
            tex_arg = tex_handle          # np.uint64 — passed as cudaTextureObject_t
        else:
            tex_arg = np.uint64(0)        # unused dummy for modes 0 / 1
        # ---------------------------------------------------------------

        # Zero the result buffer before accumulation
        cuda.memset_d32(self.result_gpu, 0, self.vx_int * self.vy_int * self.vz_int)

        input_gpu = cuda.mem_alloc(sinogram.nbytes)
        cuda.memcpy_htod(input_gpu, sinogram)

        self.func(self.result_gpu, input_gpu,
                  self.xs_gpu, self.ys_gpu, self.zs_gpu,
                  self.xv, self.yv, self.zv,
                  self.vx, self.vy, self.vz,
                  self.TSD_gpu, self.TDD_gpu,
                  self.su, self.sv,
                  self.nu, self.nv,
                  self.nproj,
                  self.source_gpu, self.detector_gpu, self.coordz_gpu,
                  self.mode,
                  tex_arg,
                  block=self.block_size, grid=self.grid_size)

        # Kernel writes: result[x + yy*vx + z*vx*vy]  →  C-order shape (vz, vy, vx)
        result = np.zeros((self.vz_int, self.vy_int, self.vx_int), dtype=np.float32)
        logger.info(f"Size of backprojection result array: {result.nbytes} bytes")
        cuda.memcpy_dtoh(result, self.result_gpu)

        input_gpu.free()

        if cuda_array is not None:
            self._destroy_sinogram_texture(cuda_array, tex_handle)

        return result
