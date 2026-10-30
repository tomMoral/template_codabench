# Reference data

The test labels, one `<test set>.csv` with a `target` column per test set.
`benchopt prepare` writes them to `$BENCHOPT_DATA_HOME/reference`, and
`tools/create_bundle.py` ships them here. Only the scoring program reads
them: submissions never see them.
