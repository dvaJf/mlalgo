import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

def _compute_features(x, y):
    n = len(y)
    features = {}

    features["y"] = y
    dy = np.gradient(y)
    features["dy"] = dy
    features["d2y"] = np.gradient(dy)
    window = max(10, n // 100)
    y_smooth = np.convolve(y, np.ones(window) / window, mode="same")
    residual = y - y_smooth
    features["window"] = np.abs(residual)
    features["mean"] = np.abs(y - np.mean(y))
    features["median"] = np.abs(y - np.median(y))

    return features


def detect_ml(df):
    y = df["y"].to_numpy()
    n = len(y)
    
    feat_dict = _compute_features(df["x"].to_numpy(), y)
    features = np.column_stack(list(feat_dict.values()))
    features_scaled = RobustScaler().fit_transform(features)

    model = IsolationForest(
        n_estimators=400,
        max_samples=256,
        max_features=0.8,
        contamination="auto",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(features_scaled)
    scores = model.score_samples(features_scaled)
    
    sortscore = np.sort(scores)
    diffs = np.diff(sortscore[int((n-1)*0.01):int((n-1)*0.3) + 1])
    if len(diffs) > 1:
        smoothed_diffs = np.convolve(diffs, np.ones(3)/3, mode='valid')
        knee_idx = np.argmax(smoothed_diffs) + 1
    else:
        knee_idx = (n-1)*0.3
    
    threshold = sortscore[knee_idx]
    anomalies = scores <= threshold
    
    return anomalies