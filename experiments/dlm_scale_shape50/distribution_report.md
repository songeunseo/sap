# Distribution survey

All224 projections, original dense weights, frozen80 corrupted inputs and same8 clean inputs. Exact moments of weight/channel-RMS/score; random scalar activation samples1024 per state. Quantile/histogram summaries use at most131072 values and are not full distributions. The positive-log variance excludes zero scores; zero mass is separately retained. Plots show equal-projection conditional nonzero densities; use weights column in CSV for parameter-weighted aggregation.

- clean_corrupted_mean_rho: 0.998269
- clean_corrupted_logvariance_rho: 0.993610
- score_mean_depth_rho: 0.936233
- score_logvariance_depth_rho: -0.578064

These are descriptive associations; clean inputs are a distribution control, not a clean-input likelihood target or an AR model control.
