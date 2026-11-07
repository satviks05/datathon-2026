# Data sources

Only the following two government CSVs from the NYC in Motion participant bundle are included.

| File | Official source | Unit |
| --- | --- | --- |
| `data/raw/nyc311_dot_aug_sep_2025.csv` | [NYC 311 Service Requests](https://data.cityofnewyork.us/d/erm2-nwe9) | Individual service request, filtered to DOT |
| `data/raw/nyc_crashes_aug_sep_2025.csv` | [NYPD Motor Vehicle Collisions](https://data.cityofnewyork.us/d/h9gi-nx95) | Police-reported collision |

Both cover August 1–September 30, 2025 based on creation/crash dates. Preserve originals as received. Request closure dates may fall outside this period and require validation before any response-time analysis.

311 reports are not verified incidents, and closed requests do not establish that a condition was repaired. Crash records may be incomplete. Coordinate precision varies across sources, and a spatial match is an analytical association rather than evidence of causation. No traffic-exposure denominator is included.
