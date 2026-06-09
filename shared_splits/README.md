# shared_splits

Unified temporal split policy for all thesis model pipelines:

- **train** — all APKs from `train_years` (default 2020–2021)
- **val + test** — stratified partition of `holdout_years` (default 2022–2023), disjoint
- 2022 and 2023 APKs may appear in either val or test; no APK is in both

Config keys (under `splits` or `preprocessing`):

```yaml
train_years: [2020, 2021]
holdout_years: [2022, 2023]
val_fraction_of_holdout: 0.5
random_seed: 42
split_mode: temporal_holdout
```

Legacy aliases (`test_years`, `temporal_holdout_years`, `val_fraction`, `temporal_year`) are
resolved for backward compatibility but configs should use the keys above.
