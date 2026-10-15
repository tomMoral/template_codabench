# Reference data

Intentionally empty. All evaluation happens in the benchopt objective during
the ingestion run; the scoring program only parses the results parquet (the
metric is already computed). No ground-truth labels reach the scoring step —
the counts live in the staged benchmark data the objective reads.
