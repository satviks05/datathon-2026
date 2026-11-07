# NYC Street Safety — Datathon 2026

Interactive NYC map exploring the geographic overlap between DOT 311 complaints and police-reported crashes, August–September 2025.

**Status:** Team starter repository. Cleaning pipelines, spatial matching, and the map are still to be implemented.

## Project question

Where do transportation complaints and injury crashes overlap, and which locations deserve closer review?

## Four independent work areas

| Person | Ownership | Starting branch | Deliverable |
| --- | --- | --- | --- |
| 1 | `pipelines/complaints/` | `team/311-cleaning` | Clean 311 CSV and coordinate-quality report |
| 2 | `pipelines/crashes/` | `team/crash-cleaning` | Clean crash CSV and coordinate-quality report |
| 3 | `analysis/` | `team/spatial-analysis` | Distance-based matches, location summaries, findings |
| 4 | `web/`, presentation, and integration | `team/interactive-map` | Interactive map with layers, filters, and exclusions shown |

Assign actual names before starting. See [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration and [data contracts](docs/DATA_CONTRACTS.md) for shared formats. Each work-area README defines its scope.

## Getting started

```bash
git clone https://github.com/satviks05/datathon-2026.git
cd datathon-2026
git switch team/311-cleaning
```

Replace the branch with your assigned branch. Each owner documents dependencies and execution commands inside their directory. No application or data pipeline is runnable yet.

## Data

Original provided CSVs are committed under `data/raw/` and must remain unchanged. Generated outputs belong in `data/processed/` and are ignored by Git; commit code that reproduces them instead. Only 311 and crash data are used.

| Initial inspection | 311 | Crashes |
| --- | ---: | ---: |
| Total records | 34,676 | 14,526 |
| Missing coordinate pairs | 4,252 | 119 |
| Zero coordinate pairs | 0 | 341 |
| Records inside a rough NYC coordinate box | 30,424 | 14,066 |

These counts come from an initial inspection, not completed cleaning. The screening box is latitude 40.45–40.95 and longitude -74.30–-73.65; it is not an NYC boundary polygon. Missing coordinates cannot be recovered from these files using street addresses because address fields are absent.

[Sources and interpretation](docs/DATA_SOURCES.md)

## Analysis principles

- Match using distances in meters, not exact coordinate equality; evaluate 25, 50, and 100 meter radii.
- Preserve unique complaint and collision IDs. Count distinct crashes when summarizing pairs.
- Exclude invalid locations from the map and spatial joins, but include exclusions in data-quality summaries.
- Distinguish complaints created before a crash from complaints created afterward.
- Complaint volume reflects reporting behavior. Spatial overlap does not establish causation or crash risk per traveler.
- Two months of observations limit historical comparisons and conclusions.

## Planned map

Separate complaint and crash layers, point clustering, complaint-category/date/borough filters, injury-severity filters, and record details. Show the number of records excluded for unusable coordinates.

## Submission checklist

- Reproducible cleaning and analysis with run instructions and dependencies
- Working interactive map and documented launch instructions
- Evidence, limitations, data citations, and presentation
- Public repository submitted by the event deadline
