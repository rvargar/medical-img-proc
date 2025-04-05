from pathlib import Path
import os
import numpy as np
import pydicom

from utils.Logger import Logger
logger = Logger().get_logger()


class DicomCTParser:
    def __init__(self, path: str, order_by_slice_loc: bool = True):
        self.path = Path(path)
        self.order_by_slice_loc = order_by_slice_loc
        self.volume = self._load_volume()

    def _load_volume(self) -> np.ndarray:
        """
        Loads dcm files onto numpy array in a sorted order on the third axes Z.
        :return: Parsed image array.
        """
        dcm_files = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.lower().endswith('.dcm')]
        if not dcm_files:
            logger.error(f"No DICOM files found in directory: {self.path}")
            raise FileNotFoundError(f"No DICOM files found in directory: {self.path}")

        slices = []
        # Sorting dcm files based on SliceLocation and InstanceNumber attributes
        for file in dcm_files:
            ds = pydicom.dcmread(file)
            if self.order_by_slice_loc:
                key = getattr(ds, 'SliceLocation', getattr(ds, 'InstanceNumber', 0))
            else:
                key = getattr(ds, 'InstanceNumber', getattr(ds, 'SliceLocation', 0))
            slices.append((key, ds))

        slices.sort(key=lambda x: x[0])

        sorted_datasets = [s[1] for s in slices]
        pixel_arrays = [ds.pixel_array.astype(np.float32) for ds in sorted_datasets]
        volume = np.stack(pixel_arrays, axis=-1)
        return volume

    def get_volume(self) -> np.ndarray:
        """
        :return: Simply returns the image volume.
        """
        return self.volume

    def transpose(self, axes: tuple) -> np.ndarray:
        """
        Transpose image volume based upon the axes permutations
        :param axes: Described permutation of the axes.
        :return: Transposed image volume
        """
        if sorted(axes) != [0, 1, 2]:
            logger.error("Axes must be a permutation of (0, 1, 2)")
            raise ValueError("Axes must be a permutation of (0, 1, 2)")
        self.volume = np.transpose(self.volume, axes)
        return self.volume

    def normalize(self, min_val: float = None, max_val: float = None) -> np.ndarray:
        """
        Normalize the volume to [0, 1] or a custom range.
        :param min_val: normalizing with defined min_val range.
        :param max_val: normalizing with defined max_val range.
        :return: Normalized image volume
        """
        min_v = min_val if min_val is not None else self.volume.min()
        max_v = max_val if max_val is not None else self.volume.max()

        self.volume = (self.volume - min_v) / (max_v - min_v)
        return self.volume

    def get_global_min(self) -> float:
        """
        :return: returns global min value of image array.
        """
        return float(self.volume.min())

    def get_global_max(self) -> float:
        """
        :return: returns global maxalue of image array.
        """
        return float(self.volume.max())

    def get_slice_minmax(self) -> list:
        """
        :return: list of (min, max) tuples for each slice along Z-axis.
        """
        return [(float(self.volume[:, :, slice_i].min()), float(self.volume[:, :, slice_i].max()))
                for slice_i in range(self.volume.shape[2])]
