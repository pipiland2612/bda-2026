Exploratory forward-time validation; not an untouched final test.
Train: dates before 2024-10-01. Test: 2024-10-01 through 2024-12-31.
All stations on each date are kept in the same partition.
Weather transforms are estimated on training rows and reused unchanged on test rows.
Both models use the same rows, fixed effects, spline basis, and half-normal sigma prior.
Weekend labels and spline boundaries match the hourly report; timestamp semantics remain unresolved.
