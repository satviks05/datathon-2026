# NYC Street Safety — Datathon 2026

Interactive NYC map exploring the geographic overlap between DOT 311 traffic signal complaints and police-reported crashes, August–September 2025.

**Status:** Existing team cleaning code and reproducible cleaned CSVs are included. Spatial matching and the map remain to be implemented.

## Project question

Where do traffic signal complaints and injury crashes overlap, and which locations deserve closer review?

## Four independent work areas

| Person | Ownership | Starting branch | Deliverable |
| --- | --- | --- | --- |
| 1 | `pipelines/complaints/` | `team/311-cleaning` | Shared cleaner maintenance, signal complaints, and complaint-quality report |
| 2 | `pipelines/crashes/` | `team/crash-cleaning` | Crash validation and coordinate-quality report |
| 3 | `analysis/` | `team/spatial-analysis` | Distance-based matches, location summaries, findings |
| 4 | `web/`, presentation, and integration | `team/interactive-map` | Interactive map with layers, filters, and exclusions shown |

Assign actual names before starting. See [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration and [data contracts](docs/DATA_CONTRACTS.md) for shared formats. Each work-area README defines its scope.

## Getting started

```bash
git clone https://github.com/satviks05/datathon-2026.git
cd datathon-2026
git switch team/311-cleaning
```

Replace the branch with your assigned branch. The existing cleaner is runnable; see [pipeline setup and commands](pipelines/README.md). Each owner documents dependencies and commands for additional work in their directory. The map is not implemented yet.

## Data

Original provided CSVs are committed under `data/raw/` and must remain unchanged. The two small cleaned CSV snapshots are committed under `data/processed/` so everyone can use them immediately. Other generated outputs are ignored. Regenerate snapshots with the committed cleaning script. Only 311 and crash data are used.

| Initial inspection | 311 | Crashes |
| --- | ---: | ---: |
| Total records | 34,676 | 14,526 |
| Missing coordinate pairs | 4,252 | 119 |
| Zero coordinate pairs | 0 | 341 |
| Records inside a rough NYC coordinate box | 30,424 | 14,066 |

These counts come from an initial inspection, not completed cleaning. The screening box is latitude 40.45–40.95 and longitude -74.30–-73.65; it is not an NYC boundary polygon. Missing coordinates cannot be recovered from these files using street addresses because address fields are absent.

The cleaner selects 6,534 traffic signal reports from the 311 source and retains **6,119** after coordinate filtering. The cleaned crash file has **14,066** records. See [pipeline details](pipelines/README.md) for its coordinate rule and validation.

[Sources and interpretation](docs/DATA_SOURCES.md)

## Analysis principles

- Match using distances in meters, not exact coordinate equality; evaluate 25, 50, and 100 meter radii.
- Preserve unique complaint and collision IDs. Count distinct crashes when summarizing pairs.
- Exclude invalid locations from the map and spatial joins, but include exclusions in data-quality summaries.
- Distinguish complaints created before a crash from complaints created afterward.
- Complaint volume reflects reporting behavior. Spatial overlap does not establish causation or crash risk per traveler.
- Two months of observations limit historical comparisons and conclusions.

## Planned map

Separate complaint and crash layers, point clustering, signal-descriptor/date/borough filters, injury-severity filters, and record details. Show the number of records excluded for unusable coordinates.

## Submission checklist

- Reproducible cleaning and analysis with run instructions and dependencies
- Working interactive map and documented launch instructions
- Evidence, limitations, data citations, and presentation
- Public repository submitted by the event deadline
