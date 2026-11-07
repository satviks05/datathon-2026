# Existing cleaning pipeline

`clean_data.py` is the team-provided script, preserved without modifications. It cleans both inputs in a single run and filters 311 to **Traffic Signal Condition**. It does not retain every DOT complaint category.

From the repository root, with Python 3.10 or newer:

```bash
python -m venv .venv
# Activate: macOS/Linux: source .venv/bin/activate
# Activate: Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r pipelines/requirements.txt
python pipelines/clean_data.py data/raw data/processed
```

Outputs committed as small shared snapshots:

- `data/processed/signal_complaints_clean.csv`: 6,119 complaints, from 6,534 signal reports selected out of 34,676 DOT reports. The coordinate screen excludes 415 signal reports.
- `data/processed/collisions_clean.csv`: 14,066 crashes, from 14,526 source records; excludes 119 missing coordinate pairs and 341 zero-coordinate pairs. Includes 6,504 crashes with injury and 6,536 with injury or fatality.

The script reproduced both uploaded snapshots byte for byte using pandas 2.2.3. Other generated artifacts remain ignored.

## Checks already implemented

Exact duplicate rows are removed; missing/conflicting IDs cause an error. Event dates must fall within August–September 2025. Coordinate strings are converted to numbers and screened with a rough bounding box (latitude 40.4–41.0, longitude -74.4–-73.6). Borough labels are normalized, but usable coordinates do not require a known borough. Injury counts must be nonnegative integers. Injury/fatality flags are derived from counts. Signal tickets with closure dates preceding creation raise an error.

## Important limits and next steps

- The coordinate box is a coarse screen, not a city-boundary check. It is slightly wider than the starter's initial inspection box; output counts agree for these files.
- The script prints counts but does not generate a machine-readable exclusion report.
- A closed ticket does not establish that a signal was repaired. Closure dates are not constrained to the historical window.
- Crash date and time remain separate columns. Analysis must combine them before comparing event order with complaint creation times.
- Interpret timestamps as America/New_York local time; the CSVs do not include timezone offsets.

## Ownership

Person 1 owns `clean_data.py`, `requirements.txt`, this README, and complaint-specific work in `complaints/`. Person 2 owns crash validation and quality-report code in `crashes/`. Coordinate changes to shared crash-cleaning logic with Person 1 instead of editing the same script concurrently. Person 2 can contribute changes through a separate pull request.
