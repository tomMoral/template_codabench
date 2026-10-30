# Reference data

The hidden test labels: one `<test set>.csv` with a `target` column per test
set whose labels submissions must not see (here, `private`). `benchopt
prepare` writes them to `$BENCHOPT_DATA_HOME/reference`, and
`tools/create_bundle.py` ships them here. Only the scoring program reads
them.
