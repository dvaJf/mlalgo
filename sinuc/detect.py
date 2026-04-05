import numpy as np

def detect(df, n_sigma=3.0,frequency=1):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    offset = y.mean()

    A = np.column_stack([np.sin(frequency * x), np.cos(frequency * x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y - offset, rcond=None)
    phase = np.arctan2(coeffs[1], coeffs[0])
    amplitude = np.sqrt(coeffs[0]**2 + coeffs[1]**2)

    fitted = amplitude * np.sin(frequency * x + phase) + offset
    residuals = y - fitted
    threshold = n_sigma * residuals.std()

    return np.abs(residuals) > threshold