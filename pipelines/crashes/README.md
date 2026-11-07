# Person 2 — Crash validation and data quality

Crash cleaning already exists in `../clean_data.py`. The committed output is `data/processed/collisions_clean.csv`; see [the contract](../../docs/DATA_CONTRACTS.md).

Own independent crash validation and quality-report tooling in this directory. Verify missing/zero coordinates, unique IDs, combined date/time, injury counts, and retained records with unknown boroughs. Do not concurrently edit the shared cleaner owned by Person 1; propose coordinated changes through a pull request.

Document any added dependencies, commands, and validation here. Starting branch: `team/crash-cleaning`.
