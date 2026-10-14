"""Validated inference using the notebook's all-history artifacts. No training."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CATEGORIES = ('M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06')
FILES = ('data/iqvia_pharma_sales.csv', 'pharma_lstm_model.keras', 'scaler.pkl', 'forecast_metadata.json')

def fingerprint(root=ROOT):
    missing = [name for name in FILES if not (root / name).is_file()]
    if missing:
        raise ValueError('Missing required files: ' + ', '.join(missing) + '. Copy the notebook export bundle into the project folder.')
    return tuple(hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES)

def validate_data(df):
    required = ['datum', *CATEGORIES]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))
    df = df.copy()
    try:
        df['datum'] = pd.to_datetime(df['datum'], errors='raise')
        df[list(CATEGORIES)] = df[list(CATEGORIES)].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Dates must be valid and sales must be numeric.') from exc
    if df['datum'].isna().any() or df['datum'].duplicated().any():
        raise ValueError('Dates must be present and unique. Resolve duplicate or missing dates.')
    df = df.sort_values('datum').reset_index(drop=True)
    if len(df) < 26:
        raise ValueError('At least 26 weekly observations are required.')
    if not df['datum'].diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError('Weekly continuity failed. Observations must be exactly seven days apart.')
    values = df[list(CATEGORIES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Sales contain missing or non-finite values. Resolve them before forecasting.')
    if (values < 0).any():
        raise ValueError('Negative sales require investigation before forecasting.')
    return df

def load_bundle(root=ROOT):
    fingerprint(root)
    df = validate_data(pd.read_csv(root / FILES[0]))
    metadata = json.loads((root / FILES[3]).read_text())
    if (metadata.get('target') != 'N02BE' or metadata.get('sequence_length') != 26
            or metadata.get('input_features') != ['N02BE']):
        raise ValueError('Metadata must specify N02BE, one input feature and a 26-week sequence.')
    if metadata.get('history_end') != str(df['datum'].max().date()):
        raise ValueError('Metadata history_end does not match the CSV. Export matching model, scaler and metadata together.')
    # Only load this repository's trusted notebook export; no user pickle uploads.
    with (root / FILES[2]).open('rb') as handle:
        scaler = pickle.load(handle)
    from sklearn.preprocessing import MinMaxScaler
    if not isinstance(scaler, MinMaxScaler) or scaler.n_features_in_ != 1:
        raise ValueError('Expected the notebook single-feature MinMaxScaler.')
    series = df['N02BE'].to_numpy(dtype=float)
    if (scaler.n_samples_seen_ != len(df) or
            not np.allclose(scaler.data_min_, [series.min()]) or
            not np.allclose(scaler.data_max_, [series.max()])):
        raise ValueError('Scaler does not match the all-history dataset. Export the matching bundle.')
    import tensorflow as tf
    model = tf.keras.models.load_model(root / FILES[1], compile=False, safe_mode=True)
    if tuple(model.input_shape[1:]) != (26, 1) or tuple(model.output_shape[1:]) != (1,):
        raise ValueError('Model must accept 26 observations of one feature and return one value.')
    return df, model, scaler, metadata

def recursive_forecast(model, scaler, history_values, steps):
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 12:
        raise ValueError('Forecast horizon must be an integer from 1 to 12 weeks.')
    history = np.asarray(history_values, dtype=float).reshape(-1, 1)
    if len(history) < 26 or not np.isfinite(history).all():
        raise ValueError('Forecast history requires at least 26 finite values.')
    sequence = scaler.transform(history)[-26:].reshape(1, 26, 1)
    predictions = []
    for _ in range(steps):
        value = float(model(sequence, training=False).numpy()[0, 0])
        if not np.isfinite(value):
            raise ValueError('Model produced a non-finite projection.')
        predictions.append(value)
        sequence = np.concatenate([sequence[:, 1:, :], np.array([[[value]]])], axis=1)
    result = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    if not np.isfinite(result).all():
        raise ValueError('Inverse scaling produced a non-finite projection.')
    return result

def forecast_frame(df, model, scaler, steps):
    values = recursive_forecast(model, scaler, df['N02BE'], steps)
    return pd.DataFrame({
        'Date': pd.date_range(df['datum'].max() + pd.Timedelta(days=7), periods=steps, freq='7D'),
        'Projected sales (Extended Units)': values,
    })
