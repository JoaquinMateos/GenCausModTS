# Statistical analysis (StaTDS)

Friedman + Nemenyi + Holm, computed with `statds` 1.1.7.

### CRL representation quality (MCC, higher is better)

Friedman over 18 cases: chi2 = 35.73, p = 3.28e-07 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.44 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| NCTRL | 2.06 | 0.770 | 0.780 | 0.130 |
| LEAP | 2.17 | 0.757 | 0.756 | 0.127 |
| TDRL | 2.89 | 0.730 | 0.723 | 0.149 |
| MOSAIC | 3.06 | 0.728 | 0.751 | 0.139 |
| iVAE | 4.83 | 0.572 | 0.501 | 0.201 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| NCTRL vs iVAE | 5.27 | 1.36e-07 | 5.44e-07 | Reject H0 |
| NCTRL vs LEAP | 0.21 | 0.833 | 0.833 | Fail to Reject H0 |
| NCTRL vs TDRL | 1.58 | 0.114 | 0.228 | Fail to Reject H0 |
| NCTRL vs MOSAIC | 1.90 | 0.0578 | 0.173 | Fail to Reject H0 |

### CRL one-step forecasting (MSE, lower is better)

Friedman over 18 cases: chi2 = 38.18, p = 1.03e-07 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.44 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| NCTRL | 1.89 | 0.310 | 0.009 | 0.865 |
| LEAP | 2.00 | 0.307 | 0.008 | 0.855 |
| TDRL | 2.83 | 0.438 | 0.008 | 1.264 |
| MOSAIC | 3.67 | 0.538 | 0.017 | 1.543 |
| iVAE | 4.61 | 0.536 | 0.019 | 1.529 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| NCTRL vs iVAE | 5.17 | 2.4e-07 | 9.61e-07 | Reject H0 |
| NCTRL vs LEAP | 0.21 | 0.833 | 0.833 | Fail to Reject H0 |
| NCTRL vs TDRL | 1.79 | 0.0731 | 0.146 | Fail to Reject H0 |
| NCTRL vs MOSAIC | 3.37 | 0.000743 | 0.00223 | Reject H0 |

### 18-problem grid, MCC (higher is better)

Friedman over 18 cases: chi2 = 5.02, p = 0.285 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.44 ranks. Fail to Reject H0 with alpha = 0.05 (Same distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| LEAP | 2.56 | 0.752 | 0.750 | 0.085 |
| TDRL | 2.72 | 0.751 | 0.757 | 0.084 |
| iVAE | 2.83 | 0.739 | 0.755 | 0.093 |
| NCTRL | 3.44 | 0.738 | 0.726 | 0.083 |
| MOSAIC | 3.44 | 0.733 | 0.734 | 0.095 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| LEAP vs iVAE | 0.53 | 0.598 | 1 | Fail to Reject H0 |
| LEAP vs TDRL | 0.32 | 0.752 | 1 | Fail to Reject H0 |
| LEAP vs NCTRL | 1.69 | 0.0917 | 0.367 | Fail to Reject H0 |
| LEAP vs MOSAIC | 1.69 | 0.0917 | 0.367 | Fail to Reject H0 |

### 18-problem grid, one-step forecasting (MSE, lower is better)

Friedman over 18 cases: chi2 = 72.00, p = 8.58e-15 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.44 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| LEAP | 1.00 | 4.515 | 4.503 | 1.587 |
| NCTRL | 2.00 | 4.622 | 4.588 | 1.582 |
| TDRL | 3.00 | 5.586 | 5.563 | 1.531 |
| iVAE | 4.00 | 9.110 | 9.070 | 3.258 |
| MOSAIC | 5.00 | 9.198 | 9.153 | 3.287 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| LEAP vs iVAE | 5.69 | 1.25e-08 | 3.76e-08 | Reject H0 |
| LEAP vs TDRL | 3.79 | 0.000148 | 0.000296 | Reject H0 |
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

Friedman over 100 cases: chi2 = 264.07, p = 5.24e-55 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 0.75 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| catsg | 1.95 | 0.354 | 0.162 | 0.499 |
| crn | 2.16 | 0.311 | 0.271 | 0.245 |
| cepae | 3.28 | 0.554 | 0.421 | 0.463 |
| ridge_t | 3.45 | 0.656 | 0.627 | 0.650 |
| nearest_neighbour | 5.06 | 0.980 | 0.998 | 0.378 |
| ganite | 5.10 | 1.066 | 0.966 | 0.493 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| catsg vs cepae | 5.03 | 4.98e-07 | 9.97e-07 | Reject H0 |
| catsg vs crn | 0.79 | 0.427 | 0.427 | Fail to Reject H0 |
| catsg vs ganite | 11.91 | <1e-12 | <1e-12 | Reject H0 |
| catsg vs ridge_t | 5.67 | 1.43e-08 | 4.3e-08 | Reject H0 |
| catsg vs nearest_neighbour | 11.75 | <1e-12 | <1e-12 | Reject H0 |

### Static CATE, TWINS (sqrt-PEHE, lower is better)

Friedman over 100 cases: chi2 = 429.24, p = 1.47e-90 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 0.75 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| catsg | 1.81 | 0.319 | 0.320 | 0.007 |
| crn | 2.07 | 0.319 | 0.320 | 0.007 |
| cepae | 2.26 | 0.320 | 0.321 | 0.007 |
| ridge_t | 3.88 | 0.321 | 0.322 | 0.007 |
| ganite | 4.98 | 0.329 | 0.328 | 0.011 |
| nearest_neighbour | 6.00 | 0.536 | 0.536 | 0.010 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| catsg vs cepae | 1.70 | 0.089 | 0.178 | Fail to Reject H0 |
| catsg vs crn | 0.98 | 0.326 | 0.326 | Fail to Reject H0 |
| catsg vs ganite | 11.98 | <1e-12 | <1e-12 | Reject H0 |
| catsg vs ridge_t | 7.82 | <1e-12 | <1e-12 | Reject H0 |
| catsg vs nearest_neighbour | 15.84 | <1e-12 | <1e-12 | Reject H0 |

### Static CATE, NEWS (sqrt-PEHE, lower is better)

Friedman over 50 cases: chi2 = 224.53, p = 1.59e-46 (StaTDS statistic; p recomputed with the exact chi-square tail). Nemenyi critical distance = 1.07 ranks. Reject H0 with alpha = 0.05 (Different distributions) (0 rows dropped for missing values.)

| Method | Mean rank | Mean | Median | SD |
| --- | --- | --- | --- | --- |
| cepae | 1.20 | 3.432 | 3.185 | 1.319 |
| crn | 2.04 | 3.805 | 3.658 | 0.977 |
| catsg | 3.32 | 4.326 | 4.130 | 1.184 |
| nearest_neighbour | 3.44 | 4.641 | 4.589 | 1.333 |
| ridge_t | 5.14 | 9.570 | 8.724 | 3.141 |
| ganite | 5.86 | 12.728 | 12.167 | 4.388 |

Holm post-hoc against the best-ranked method:

| Comparison | Z | p | adjusted p | Result |
| --- | --- | --- | --- | --- |
| cepae vs catsg | 5.67 | 1.46e-08 | 2.92e-08 | Reject H0 |
| cepae vs crn | 2.24 | 0.0248 | 0.0248 | Reject H0 |
| cepae vs ganite | 12.45 | <1e-12 | <1e-12 | Reject H0 |
| cepae vs ridge_t | 10.53 | <1e-12 | <1e-12 | Reject H0 |
| cepae vs nearest_neighbour | 5.99 | 2.14e-09 | 6.43e-09 | Reject H0 |

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
