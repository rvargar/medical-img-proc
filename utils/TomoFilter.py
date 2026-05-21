import numpy as np


def tomo_filter(proj, nu, nv, nProj, su, sv, d, filter_type):
    """
    Applies a frequency domain filter to projection data for tomographic reconstruction.
    """
    # Get the filter and frequency axis
    filt, w = generate_filter(filter_type, nu, d)

    # Create the zero-padded projection grid
    fproj = np.zeros((len(filt), nv))

    # Calculate start and end indices for centering (equivalent to end/2-nu/2 : end/2+nu/2-1)
    start_idx = len(filt) // 2 - nu // 2
    end_idx = start_idx + nu

    # Insert the projection data into the center of the zero-padded array
    fproj[start_idx:end_idx, :] = proj

    # FFT doesn't like NaNs, replace them with 0
    fproj[np.isnan(fproj)] = 0.0

    # Compute 1D FFT along the columns (axis=0)
    fproj = np.fft.fft(fproj, axis=0)

    # Multiply the frequency domain projections by the filter
    # Using np.newaxis to broadcast the 1D filter across all columns,
    fproj = fproj * filt[:, np.newaxis]

    # Compute the inverse 1D FFT along the columns and take the real part
    fproj = np.real(np.fft.ifft(fproj, axis=0))

    # Extract the filtered, original-sized projections and scale
    proj_filtered = fproj[start_idx:end_idx, :]
    proj_filtered = proj_filtered / 2.0 * (2 * np.pi / nProj) / (su / nu)

    return proj_filtered, filt, w


def generate_filter(filter_type, length, d):
    """
    Generates the frequency filter.
    (Named 'generate_filter' to avoid clustering with Python's built-in 'filter')
    """
    next_pow_2 = int(np.ceil(np.log2(2 * length)))
    order = max(64, 2 ** next_pow_2)

    # Create double sized filter for the image
    # Note: size is order/2 + 1 to include Nyquist
    filt = 2 * np.arange(order // 2 + 1) / order

    # Frequency axis up to Nyquist
    w = 2 * np.pi * np.arange(len(filt)) / order

    filter_lower = filter_type.lower()

    if filter_lower == 'ramp':
        pass  # Do nothing, filt is already a ramp

    elif filter_lower == 'shepp-logan':
        filt[1:] = filt[1:] * (np.sin(w[1:] / (2 * d)) / (w[1:] / (2 * d)))

    elif filter_lower == 'cosine':
        filt[1:] = filt[1:] * np.cos(w[1:] / (2 * d))

    elif filter_lower == 'hamming':
        filt[1:] = filt[1:] * (0.54 + 0.46 * np.cos(w[1:] / d))

    elif filter_lower == 'hanning':
        filt[1:] = filt[1:] * (1 + np.cos(w[1:] / d)) / 2

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Crop the frequency response
    filt[w > np.pi * d] = 0

    # Symmetry of the filter for real-valued signals: mirror the positive frequencies to negative frequencies
    filt = np.concatenate((filt, filt[-2:0:-1]))

    return filt, w