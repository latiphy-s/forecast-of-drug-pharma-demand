# Notebook evidence audit

Sebastian Latiphy confirmed that this is his implementation, using an adapted notebook structure. The corrected notebook replaces an earlier copy containing leftover author and deployment links. Recorded model results are preserved.

## Recorded results

Training was not rerun. Results below come from saved notebook cells 24, 27 and 28 (zero-based).

| Model | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| LSTM | 3122.4608 | 3945.1343 | 0.9212 | 4.07% |
| Prophet | 4392.9871 | 5719.1311 | 0.8343 | 5.63% |
| ARIMA | 12713.7405 | 14164.3723 | -0.0162 | 18.26% |

## What the code actually does

- The supplied CSV has 313 weekly rows, eight sales categories and 22 columns, from 2018-01-01 to 2023-12-25. The introduction's monthly description is incorrect.
- Feature engineering removes the first 52 rows, leaving 261 observations. Sequences use 26 prior observations and one target. There are 188 training sequences and 47 test sequences.
- The LSTM input has one feature: past N02BE sales. Engineered calendar, rolling and domain features are not supplied to this model. The claim that they improved R² is not supported by the shown experiment.
- MinMaxScaler is fitted on the complete series before the train/test split. Fit it on training observations only in a revised experiment.
- LSTM test inputs contain observed prior test-period sales, giving rolling one-step predictions. ARIMA and Prophet generate the complete test horizon from one origin. These are different forecasting tasks.
- Source observations are Mondays. Prophet uses freq='W', which generates Sundays, so predicted dates should be aligned explicitly before comparing values.
- The future scenario recursively predicts 12 steps and applies an additional seasonal multiplier. It is not the same procedure as the test evaluation.
- The displayed '90% Confidence Interval' is a fixed plus/minus 8% band. It is not calibrated statistical uncertainty.
- Correlation alone cannot establish that patients switch between drugs. Sales need not equal unconstrained demand.
- The CSV says 'IQVIA MIDAS', but that label alone does not verify origin, authenticity or redistribution terms.
- The supplied files contain no saved model, scaler or runnable Streamlit app. The notebook's existing external app link is not Sebastian's deployment.

## Next experiment

Use identical rolling origins and forecast horizons for every model, train-only preprocessing, explicit date alignment and a seasonal-naive baseline. Compare one-step and 12-week recursive forecasts separately. Estimate interval coverage from rolling backtests. Record original and revised results separately.
