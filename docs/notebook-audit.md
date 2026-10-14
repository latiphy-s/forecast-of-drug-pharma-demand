# Forecasting methodology and evidence

## Data and model inputs

The supplied CSV contains 313 weekly observations from 1 January 2018 to 25 December 2023 and eight sales categories. The LSTM uses the previous 26 N02BE observations as its only input feature. Calendar and related-category features are candidates for further experiments.

## Evaluation protocol

Notebook section 6 compares forecasts from a common origin of 14 November 2022. All models predict the same Monday dates without receiving holdout sales. The first 12 weeks end on 6 February 2023; the full 58-week holdout ends on 25 December 2023.

The scaler is fitted on the initial fitting observations. A chronological validation segment selects seven training epochs, then the evaluation LSTM is refitted on pre-test sequences. Persistence, seasonal naive (52 weeks), ARIMA(2,1,2) and Prophet provide comparison forecasts. The one-step diagnostic is distinct from this matched-origin comparison.

`evaluation.json` records the displayed scores from notebook cell 29. The dashboard and README report only the matched-origin results. A single origin does not establish statistical superiority or a universally best model.

## Dashboard inference

Notebook cells 32 and 34 train and export a separate all-history model with its matching scaler and metadata. The app loads `pharma_lstm_model.keras` with compilation disabled and caches it. Each scaled prediction is appended to the 26-week input window, and projections are inverse-transformed into Extended Units. Dates start seven days after the final observation.

The 12-week projection covers 1 January to 18 March 2024. There are no additional seasonal multipliers or uncertainty bands. Tests compare each horizon with the notebook's saved projection output, allowing 0.1 Extended Units for cross-platform floating-point differences. Local verification found a maximum difference below 0.003 Extended Units.

## Interpretation boundaries

N02BE monthly average weekly sales peak in March–May in this dataset; no clinical cause is established. Aggregate category correlation does not demonstrate patient substitution. Observed sales may be constrained by inventory availability. The CSV's source label does not independently establish provenance. Model performance across multiple origins, uncertainty coverage and operational inventory benefits remain to be assessed.
