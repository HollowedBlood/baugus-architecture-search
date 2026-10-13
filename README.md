## Chronological data splits

Data is ordered by market date and split across all tickers together. Samples are never randomly shuffled between training, validation, and test sets.

- **Training:** The earliest observations. Model weights, preprocessing statistics, and any data-derived noise-control parameters are fitted using this portion only.
- **Search validation:** One or more later chronological periods, following their corresponding training periods. These determine early stopping, checkpoint selection, hyperparameter optimization, and finalist selection. If multiple periods are used, training expands forward through time, with each validation period remaining later than its training data.
- **Final test:** The most recent reserved period. Its labels and performance results are excluded from all search decisions. Finalists and evaluation rules are frozen before this period is evaluated.

At every boundary, training samples whose future target reaches into the next partition are removed. For a five-trading-step prediction horizon, this means purging the final five training prediction dates before validation begins; the same rule applies before the final test period.

Validation and test input windows may include earlier historical observations, because those would be available at prediction time. Their target outcomes must remain entirely within their respective partitions.

Every architecture, feature condition, and seed uses the same prediction dates, target definition, and split boundaries. Congressional features become available on their public disclosure date, rather than their transaction date.

**Exact date boundaries, validation-period count, and partition sizes are pending. They will be recorded in the experiment configuration and this README before the search begins.**
