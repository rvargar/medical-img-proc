import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2

from utils.Logger import Logger
from utils.DicomCTParser import DicomCTParser
from utils.VolumeVisualizer import VolumeVisualizer as visu
logger = Logger().get_logger()

if __name__ == '__main__':
    input_path = './data/Lung-Pet-CT-Dx/Lung_Dx-A0018/07-24-2008-lungc-69037/301.000000-1.25mm-74521'
    logger.info(f"Reading dicom files from: {input_path}")
    ct_parser = DicomCTParser(path=input_path, order_by_slice_loc=True).transpose((1, 2, 0))
    volume_np = ct_parser
    visu.show_slices(volume_np)
    logger.success("Done..")

