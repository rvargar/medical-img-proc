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
        dcm_files = sorted([os.path.join(self.path, f) for f in os.listdir(self.path) if f.lower().endswith('.dcm')])
        if not dcm_files:
            logger.error(f"No DICOM files found in directory: {self.path}")
            raise FileNotFoundError(f"No DICOM files found in directory: {self.path}")


        dicom_datasets = []
        for file in dcm_files:
            try:
                ds = pydicom.dcmread(file)
                dicom_datasets.append(ds)
            except Exception as e:
                logger.error(f"Error reading DICOM file {file}: {e}")
                continue

        if not dicom_datasets:
            logger.error(f"No valid DICOM files could be read from directory: {self.path}")
            raise ValueError(f"No valid DICOM files could be read from directory: {self.path}")

        # Sort using the improved method
        sorted_datasets = self._sort_dicom_slices(dicom_datasets)
        pixel_arrays = [ds.pixel_array.astype(np.float32) for ds in sorted_datasets]
        volume = np.stack(pixel_arrays, axis=-1)
        return volume

    def _sort_dicom_slices(self, dicom_datasets: list) -> list:
        """
        Sorts a list of DICOM datasets based on spatial metadata with improved reliability.
        
        Parameters:
            dicom_datasets (list): List of pydicom Dataset objects.
            
        Returns:
            list: Sorted list of pydicom Dataset objects.
        """
        if not dicom_datasets:
            return []

        try:
            # First try the most reliable method using spatial information
            if all(hasattr(ds, 'ImageOrientationPatient') and hasattr(ds, 'ImagePositionPatient') 
                   for ds in dicom_datasets):
                
                # Use ImageOrientationPatient to get slice normal
                iop = dicom_datasets[0].ImageOrientationPatient  # 6 values: 2 direction cosines
                dir_cos_x = np.array(iop[:3])
                dir_cos_y = np.array(iop[3:])
                slice_normal = np.cross(dir_cos_x, dir_cos_y)

                slice_normal = slice_normal / np.linalg.norm(slice_normal)

                def slice_position(ds):
                    ipp = np.array(ds.ImagePositionPatient)
                    return np.dot(ipp, slice_normal)

                sorted_datasets = sorted(dicom_datasets, key=slice_position)
                
                # Validate the sorted order
                positions = [slice_position(ds) for ds in sorted_datasets]
                for i in range(1, len(positions)):
                    if abs(positions[i] - positions[i-1]) > 1e-3:  # Threshold for detecting jumps
                        logger.warning(f"Potential jump detected between slices {i-1} and {i}: "
                                      f"{positions[i-1]} -> {positions[i]}")
                
                return sorted_datasets

            # If spatial info is missing, try SliceLocation
            elif all(hasattr(ds, 'SliceLocation') for ds in dicom_datasets):
                sorted_datasets = sorted(dicom_datasets, key=lambda ds: float(ds.SliceLocation))
                
                # Validate the sorted order
                positions = [float(ds.SliceLocation) for ds in sorted_datasets]
                for i in range(1, len(positions)):
                    if abs(positions[i] - positions[i-1]) > 1e-3:  # Threshold for detecting jumps
                        logger.warning(f"Potential jump detected between slices {i-1} and {i}: "
                                      f"{positions[i-1]} -> {positions[i]}")
                
                return sorted_datasets


            elif all(hasattr(ds, 'InstanceNumber') for ds in dicom_datasets):
                sorted_datasets = sorted(dicom_datasets, key=lambda ds: int(ds.InstanceNumber))
                
                # Validate the sorted order
                positions = [int(ds.InstanceNumber) for ds in sorted_datasets]
                for i in range(1, len(positions)):
                    if positions[i] - positions[i-1] != 1:  # Expecting consecutive numbers
                        logger.warning(f"Non-consecutive InstanceNumber detected between slices {i-1} and {i}: "
                                      f"{positions[i-1]} -> {positions[i]}")
                
                return sorted_datasets

            else:
                logger.warning("Falling back to filename sorting - slice order may not be correct")
                return sorted(dicom_datasets, key=lambda ds: ds.filename)
                
        except Exception as e:
            logger.error(f"Sorting failed: {e}")

            return dicom_datasets

    def get_volume(self) -> np.ndarray:
        """
        :return: Simply returns the image volume.
        """
        return self.volume

    def transpose(self, axes: tuple):
        """
        Transpose image volume based upon the axes permutations
        :param axes: Described permutation of the axes.
        :return: Transposed image volume
        """
        if sorted(axes) != [0, 1, 2]:
            logger.error("Axes must be a permutation of (0, 1, 2)")
            raise ValueError("Axes must be a permutation of (0, 1, 2)")
        self.volume = np.transpose(self.volume, axes)
        #return self.volume

    def normalize(self, min_val: float = None, max_val: float = None):
        """
        Normalize the volume to [0, 1] or a custom range.
        :param min_val: normalizing with defined min_val range.
        :param max_val: normalizing with defined max_val range.
        :return: Normalized image volume
        """
        min_v = min_val if min_val is not None else self.volume.min()
        max_v = max_val if max_val is not None else self.volume.max()

        self.volume = (self.volume - min_v) / (max_v - min_v)
        #return self.volume

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

    def flip_y(self):
        """
        Flip the image volume along the Y axis (axis=0).
        :return: Flipped image volume
        """
        self.volume = np.flip(self.volume, axis=2)
        #return self.volume
