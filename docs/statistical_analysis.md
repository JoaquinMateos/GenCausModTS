# Statistical analysis (StaTDS)

Friedman + Nemenyi + Holm, computed with `statds` 1.1.7.

### CRL representation quality (MCC, higher is better)

Friedman over 18 cases: chi2 = 42.09, p = 1.6e-08 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.44 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| NCTRL | 1.78 | 0.782 | 0.816 | 0.127 |
| LEAP | 2.28 | 0.761 | 0.756 | 0.124 |
| TDRL | 2.67 | 0.741 | 0.761 | 0.147 |
| MOSAIC | 3.39 | 0.730 | 0.748 | 0.141 |
| iVAE | 4.89 | 0.581 | 0.546 | 0.191 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| NCTRL vs iVAE | 5.90 | 3.57e-09 | 1.43e-08 | Reject H0 |
| NCTRL vs LEAP | 0.95 | 0.343 | 0.343 | Fail to Reject H0 |
| NCTRL vs TDRL | 1.69 | 0.0917 | 0.183 | Fail to Reject H0 |
| NCTRL vs MOSAIC | 3.06 | 0.00224 | 0.00671 | Reject H0 |

### CRL one-step forecasting (MSE, lower is better)

Friedman over 18 cases: chi2 = 34.27, p = 6.57e-07 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.44 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| LEAP | 2.06 | 0.189 | 0.008 | 0.577 |
| NCTRL | 2.06 | 0.192 | 0.009 | 0.587 |
| TDRL | 2.67 | 0.227 | 0.008 | 0.709 |
| MOSAIC | 3.67 | 0.275 | 0.017 | 0.830 |
| iVAE | 4.56 | 0.274 | 0.021 | 0.820 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| LEAP vs iVAE | 4.74 | 2.1e-06 | 8.41e-06 | Reject H0 |
| LEAP vs TDRL | 1.16 | 0.246 | 0.493 | Fail to Reject H0 |
| LEAP vs NCTRL | 0.00 | 1 | 1 | Fail to Reject H0 |
| LEAP vs MOSAIC | 3.06 | 0.00224 | 0.00671 | Reject H0 |

### 18-problem grid, MCC (higher is better)

Not testable: 0 complete cases, 5 methods (0 rows dropped).

### 18-problem grid, one-step forecasting (MSE, lower is better)

Not testable: 0 complete cases, 5 methods (0 rows dropped).

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

Friedman over 30 cases: chi2 = 73.50, p = 1.91e-14 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.38 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| ridge_t | 1.40 | 3.230 | 1.113 | 5.981 |
| cepae | 3.07 | 4.711 | 1.960 | 6.635 |
| crn | 3.27 | 5.509 | 1.977 | 9.233 |
| ganite | 3.47 | 3.295 | 2.447 | 3.029 |
| catsg | 4.87 | 7.027 | 2.908 | 12.411 |
| nearest_neighbour | 4.93 | 6.906 | 2.713 | 11.683 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| ridge_t vs catsg | 7.18 | <1e-12 | 2.86e-12 | Reject H0 |
| ridge_t vs cepae | 3.45 | 0.00056 | 0.00056 | Reject H0 |
| ridge_t vs crn | 3.86 | 0.000111 | 0.000223 | Reject H0 |
| ridge_t vs ganite | 4.28 | 1.88e-05 | 5.65e-05 | Reject H0 |
| ridge_t vs nearest_neighbour | 7.31 | <1e-12 | 1.29e-12 | Reject H0 |

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
