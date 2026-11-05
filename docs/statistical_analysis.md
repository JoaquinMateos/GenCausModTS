# Statistical analysis (StaTDS)

Friedman + Nemenyi + Holm, computed with `statds` 1.1.7.

### CRL representation quality (MCC, higher is better)

Friedman over 19 cases: chi2 = 36.04, p = 2.84e-07 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.40 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| NCTRL | 1.89 | 0.749 | 0.808 | 0.187 |
| LEAP | 2.42 | 0.728 | 0.723 | 0.186 |
| TDRL | 2.68 | 0.712 | 0.754 | 0.192 |
| MOSAIC | 3.26 | 0.704 | 0.700 | 0.179 |
| iVAE | 4.74 | 0.561 | 0.539 | 0.204 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| NCTRL vs iVAE | 5.54 | 3.02e-08 | 1.21e-07 | Reject H0 |
| NCTRL vs LEAP | 1.03 | 0.305 | 0.305 | Fail to Reject H0 |
| NCTRL vs TDRL | 1.54 | 0.124 | 0.248 | Fail to Reject H0 |
| NCTRL vs MOSAIC | 2.67 | 0.00764 | 0.0229 | Reject H0 |

### CRL one-step forecasting (MSE, lower is better)

Friedman over 19 cases: chi2 = 37.68, p = 1.3e-07 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.40 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| LEAP | 2.00 | 0.180 | 0.007 | 0.562 |
| NCTRL | 2.05 | 0.182 | 0.008 | 0.572 |
| TDRL | 2.68 | 0.215 | 0.007 | 0.691 |
| MOSAIC | 3.68 | 0.260 | 0.017 | 0.809 |
| iVAE | 4.58 | 0.260 | 0.018 | 0.799 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| LEAP vs iVAE | 5.03 | 4.97e-07 | 1.99e-06 | Reject H0 |
| LEAP vs TDRL | 1.33 | 0.182 | 0.365 | Fail to Reject H0 |
| LEAP vs NCTRL | 0.10 | 0.918 | 0.918 | Fail to Reject H0 |
| LEAP vs MOSAIC | 3.28 | 0.00103 | 0.00308 | Reject H0 |

### 18-problem grid, MCC (higher is better)

Friedman over 18 cases: chi2 = 46.22, p = 2.21e-09 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.44 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| TDRL | 1.72 | 0.650 | 0.647 | 0.103 |
| NCTRL | 1.94 | 0.649 | 0.617 | 0.106 |
| LEAP | 2.94 | 0.632 | 0.622 | 0.098 |
| MOSAIC | 3.56 | 0.611 | 0.588 | 0.083 |
| iVAE | 4.83 | 0.503 | 0.487 | 0.094 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| TDRL vs iVAE | 5.90 | 3.57e-09 | 1.43e-08 | Reject H0 |
| TDRL vs LEAP | 2.32 | 0.0204 | 0.0408 | Reject H0 |
| TDRL vs NCTRL | 0.42 | 0.673 | 0.673 | Fail to Reject H0 |
| TDRL vs MOSAIC | 3.48 | 0.000504 | 0.00151 | Reject H0 |

### 18-problem grid, one-step forecasting (MSE, lower is better)

Friedman over 18 cases: chi2 = 68.58, p = 4.53e-14 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.44 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| LEAP | 1.00 | 0.064 | 0.058 | 0.018 |
| NCTRL | 2.00 | 0.064 | 0.058 | 0.017 |
| TDRL | 3.39 | 0.068 | 0.063 | 0.019 |
| iVAE | 3.61 | 0.070 | 0.063 | 0.017 |
| MOSAIC | 5.00 | 0.082 | 0.075 | 0.023 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| LEAP vs iVAE | 4.95 | 7.26e-07 | 2.18e-06 | Reject H0 |
| LEAP vs TDRL | 4.53 | 5.83e-06 | 1.17e-05 | Reject H0 |
| LEAP vs NCTRL | 1.90 | 0.0578 | 0.0578 | Fail to Reject H0 |
| LEAP vs MOSAIC | 7.59 | <1e-12 | <1e-12 | Reject H0 |

### Temporal CTEE, harmonic (sqrt-PEHE, lower is better)

Friedman over 10 cases: chi2 = 33.68, p = 8.67e-07 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.93 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| CRN | 1.00 | 0.270 | 0.280 | 0.101 |
| CEPAE | 2.00 | 0.634 | 0.454 | 0.364 |
| CaTSG | 3.50 | 1.098 | 0.864 | 0.682 |
| GANITE | 4.10 | 1.130 | 0.891 | 0.701 |
| WuIPTW | 4.40 | 1.479 | 1.630 | 0.576 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| CRN vs CEPAE | 1.41 | 0.157 | 0.157 | Fail to Reject H0 |
| CRN vs CaTSG | 3.54 | 0.000407 | 0.000814 | Reject H0 |
| CRN vs GANITE | 4.38 | 1.16e-05 | 3.49e-05 | Reject H0 |
| CRN vs WuIPTW | 4.81 | 1.52e-06 | 6.09e-06 | Reject H0 |

### Temporal CTEE, tumour (sqrt-PEHE, lower is better)

Friedman over 10 cases: chi2 = 16.48, p = 0.00244 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.93 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| CRN | 1.70 | 6.416 | 6.299 | 0.430 |
| CaTSG | 2.60 | 6.653 | 6.741 | 0.406 |
| CEPAE | 3.10 | 6.726 | 6.697 | 0.381 |
| WuIPTW | 3.10 | 6.840 | 6.836 | 0.469 |
| GANITE | 4.50 | 7.291 | 7.224 | 0.465 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| CRN vs CEPAE | 1.98 | 0.0477 | 0.143 | Fail to Reject H0 |
| CRN vs CaTSG | 1.27 | 0.203 | 0.143 | Fail to Reject H0 |
| CRN vs GANITE | 3.96 | 7.5e-05 | 0.203 | Reject H0 |
| CRN vs WuIPTW | 1.98 | 0.0477 | 0.0003 | Fail to Reject H0 |

### Temporal CTEE, linear_sem (sqrt-PEHE, lower is better)

Friedman over 10 cases: chi2 = 30.16, p = 4.54e-06 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.93 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| CRN | 1.00 | 0.110 | 0.059 | 0.122 |
| CaTSG | 2.60 | 1.053 | 1.047 | 0.469 |
| CEPAE | 2.90 | 1.380 | 1.456 | 0.260 |
| GANITE | 3.90 | 1.411 | 1.483 | 0.280 |
| WuIPTW | 4.60 | 1.444 | 1.485 | 0.283 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| CRN vs CEPAE | 2.69 | 0.00721 | 0.0144 | Reject H0 |
| CRN vs CaTSG | 2.26 | 0.0237 | 0.0237 | Reject H0 |
| CRN vs GANITE | 4.10 | 4.11e-05 | 0.000123 | Reject H0 |
| CRN vs WuIPTW | 5.09 | 3.56e-07 | 1.42e-06 | Reject H0 |

### Static CATE, IHDP (sqrt-PEHE, lower is better)

Friedman over 100 cases: chi2 = 257.36, p = 1.45e-53 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 0.75 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| ridge_t | 1.31 | 2.397 | 1.015 | 3.952 |
| cepae | 3.14 | 3.924 | 2.187 | 4.733 |
| crn | 3.33 | 4.540 | 2.008 | 6.656 |
| ganite | 3.39 | 2.787 | 2.395 | 1.870 |
| nearest_neighbour | 4.78 | 5.475 | 2.664 | 7.892 |
| catsg | 5.05 | 5.451 | 2.898 | 7.769 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| ridge_t vs catsg | 14.14 | <1e-12 | <1e-12 | Reject H0 |
| ridge_t vs cepae | 6.92 | 4.62e-12 | 4.62e-12 | Reject H0 |
| ridge_t vs crn | 7.63 | <1e-12 | <1e-12 | Reject H0 |
| ridge_t vs ganite | 7.86 | <1e-12 | <1e-12 | Reject H0 |
| ridge_t vs nearest_neighbour | 13.12 | <1e-12 | <1e-12 | Reject H0 |

### Static CATE, ACIC (sqrt-PEHE, lower is better)

Friedman over 100 cases: chi2 = 319.76, p = 5.64e-67 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 0.75 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| catsg | 1.43 | 0.174 | 0.047 | 0.312 |
| crn | 2.32 | 0.309 | 0.255 | 0.208 |
| ridge_t | 3.25 | 0.649 | 0.596 | 0.533 |
| cepae | 3.77 | 0.694 | 0.488 | 0.596 |
| nearest_neighbour | 4.83 | 0.984 | 0.998 | 0.359 |
| ganite | 5.40 | 1.310 | 1.250 | 0.677 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| catsg vs cepae | 8.84 | <1e-12 | <1e-12 | Reject H0 |
| catsg vs crn | 3.36 | 0.000769 | 0.000769 | Reject H0 |
| catsg vs ganite | 15.01 | <1e-12 | <1e-12 | Reject H0 |
| catsg vs ridge_t | 6.88 | 6.03e-12 | 1.21e-11 | Reject H0 |
| catsg vs nearest_neighbour | 12.85 | <1e-12 | <1e-12 | Reject H0 |

### Static CATE, TWINS (sqrt-PEHE, lower is better)

Friedman over 100 cases: chi2 = 437.15, p = 2.9e-92 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 0.75 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| catsg | 1.70 | 0.319 | 0.320 | 0.007 |
| crn | 1.75 | 0.319 | 0.320 | 0.007 |
| cepae | 2.65 | 0.320 | 0.320 | 0.007 |
| ridge_t | 3.90 | 0.321 | 0.322 | 0.007 |
| ganite | 5.26 | 0.472 | 0.431 | 0.123 |
| nearest_neighbour | 5.74 | 0.536 | 0.536 | 0.010 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| catsg vs cepae | 3.59 | 0.00033 | 0.00066 | Reject H0 |
| catsg vs crn | 0.19 | 0.85 | 0.85 | Fail to Reject H0 |
| catsg vs ganite | 13.46 | <1e-12 | <1e-12 | Reject H0 |
| catsg vs ridge_t | 8.32 | <1e-12 | <1e-12 | Reject H0 |
| catsg vs nearest_neighbour | 15.27 | <1e-12 | <1e-12 | Reject H0 |

### Static CATE, NEWS (sqrt-PEHE, lower is better)

Friedman over 50 cases: chi2 = 223.74, p = 2.35e-46 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.07 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| cepae | 1.20 | 3.449 | 3.257 | 1.322 |
| crn | 2.06 | 3.802 | 3.722 | 0.974 |
| catsg | 3.32 | 4.326 | 4.130 | 1.184 |
| nearest_neighbour | 3.42 | 4.641 | 4.589 | 1.333 |
| ridge_t | 5.14 | 9.570 | 8.724 | 3.141 |
| ganite | 5.86 | 12.948 | 11.022 | 5.376 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| cepae vs catsg | 5.67 | 1.46e-08 | 2.92e-08 | Reject H0 |
| cepae vs crn | 2.30 | 0.0215 | 0.0215 | Reject H0 |
| cepae vs ganite | 12.45 | <1e-12 | <1e-12 | Reject H0 |
| cepae vs ridge_t | 10.53 | <1e-12 | <1e-12 | Reject H0 |
| cepae vs nearest_neighbour | 5.93 | 2.97e-09 | 8.91e-09 | Reject H0 |

### Real panels, one-step forecasting (MSE, lower is better)

Friedman over 11 cases: chi2 = 17.60, p = 0.00148 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.84 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| LEAP | 2.36 | 0.438 | 0.312 | 0.361 |
| MOSAIC | 2.36 | 0.490 | 0.261 | 0.453 |
| NCTRL | 2.55 | 0.411 | 0.299 | 0.335 |
| TDRL | 3.00 | 0.446 | 0.312 | 0.366 |
| iVAE | 4.73 | 1.060 | 0.637 | 1.234 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| LEAP vs iVAE | 3.51 | 0.000455 | 0.00182 | Reject H0 |
| LEAP vs TDRL | 0.94 | 0.345 | 1 | Fail to Reject H0 |
| LEAP vs NCTRL | 0.27 | 0.787 | 1 | Fail to Reject H0 |
| LEAP vs MOSAIC | 0.00 | 1 | 1 | Fail to Reject H0 |
