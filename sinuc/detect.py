import numpy as np

def detect(df, window=None, n_sigma=2.5):
    y = df["y"].to_numpy()
    n = len(y)
    
    if window is None:
        window = max(20, n // 100)
    
    anomalies = np.zeros(n, dtype=bool)
    
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)

        local = y[start:end]
        local_avg = np.mean(local)
        local_std = np.std(local)
        
        if local_std > 0:
            z_score = abs(y[i] - local_avg) / local_std
            if z_score > n_sigma:
                anomalies[i] = True
    
    return anomalies