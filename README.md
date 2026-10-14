## Chronological data splits

`shared_data.get_xyts()` returns four CPU PyTorch tensors:
`X_train`, `y_train`, `X_test`, and `y_test`.

Samples are ordered by market date and split across all tickers together
using `train_test_split(..., shuffle=False)`. There is currently no separate
validation set.

- **Training:** The earlier prediction dates. Model weights are trained on
  these samples. When scaling is enabled, `StandardScaler` is fitted only
  on the unique market days included in training input windows.
- **Test:** The later prediction dates. With `test_size=0.2`, the latest
  20% of eligible samples are assigned to testing, before training samples
  are purged at the boundary. This set currently supports development
  evaluation.

### Input windows and targets

Each input sample contains `lookback` consecutive trading observations,
ending on its prediction date, for all 20 tickers.

Each ticker's target is determined by its open-to-open return:

`return = open[prediction_date + horizon] / open[prediction_date] - 1`

Here, `horizon` counts trading observations, not calendar days.

- **0 — Negative:** Return below `-deadzone`.
- **1 — Neutral:** Return between `-deadzone` and `+deadzone`, inclusive.
- **2 — Positive:** Return above `+deadzone`.

Training samples are removed if their target date falls on or after the
first test prediction date. With `horizon=5`, this removes the final five
otherwise eligible training samples.

Test input windows may include earlier training-period observations:
these are historical context available at prediction time. Target labels
are never included in the inputs.

### Feature conditions

- **market:** Six market features per ticker.
- **congress:** The six market features plus seven daily congressional
  aggregates per ticker.
- **noise:** The six market features plus resampled congressional daily
  vectors, drawn only from days included in training input windows.

Disclosures are assigned to the first market day strictly after their
filing date, rather than their transaction date.

When scaling is enabled, the same fitted congressional scaler is used
for actual and resampled congressional features. Target labels always
use original, unscaled open prices.

For comparisons, the lookback, horizon, deadzone, test size, start date,
and source data must remain identical across feature conditions.
The noise seed controls reproducibility of the noise condition.