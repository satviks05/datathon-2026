# Person 1 — Signal complaints and shared cleaner

The existing cleaner is `../clean_data.py`; see [pipeline instructions](../README.md). The committed complaint output is `data/processed/signal_complaints_clean.csv`, using the source field names in [the contract](../../docs/DATA_CONTRACTS.md).

Own the shared cleaner, its dependency file, and complaint-specific extensions here. Keep the signal-only filter explicit. Add machine-readable complaint-quality reporting as needed. Coordinate any crash-cleaning changes with Person 2.

Starting branch: `team/311-cleaning`.
