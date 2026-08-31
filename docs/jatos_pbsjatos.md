# pbsjatos migration

BOOST behavioral pipeline pulls from **pbsjatos** (`https://pbsjatos.psychology.uiowa.edu`).
Until all observational (OA/OB/OC) studies are imported, nightlies also pull missing OBS
sites from the **legacy** server when `JATOS_LEGACY_TOKEN` is set.

## Study layout

- Titles: `{SITE}_{TASK}` — e.g. `IA_AF`, `OA_NF`, `OC_VNB`
- Full battery: 13 tasks × 6 sites = **78** studies
- pbsjatos (2026-08-31): **44** studies — IA/IB/IC complete; OA/OB/OC mostly absent

## Code map

| File | Role |
|------|------|
| `code/jatos_study_ids.py` | Registry, discovery scan, legacy OBS fallback |
| `code/jatos_discovered_ids.json` | Cached pbsjatos title → studyId map |
| `code/data_processing/pull_handler.py` | API pull + zip extract |
| `scripts/refresh_jatos_ids.py` | Re-probe pbsjatos after new imports |

## Nightly env (GitHub Actions / vosslink)

| Variable | Default | Purpose |
|----------|---------|---------|
| `JATOS_TOKEN` | (required) | pbsjatos API token |
| `JATOS_BASE_URL` | pbsjatos URL | Primary server |
| `JATOS_LEGACY_TOKEN` | — | Old server token for OA/OB/OC backfill |
| `JATOS_LEGACY_BASE_URL` | `https://jatos.psychology.uiowa.edu` | Legacy host |
| `JATOS_LEGACY_PULL` | `1` | Set `0` when OBS fully migrated |
| `JATOS_LEGACY_DAYS_AGO` | `2000` | Wide window for historical OBS |
| `JATOS_DAYS_AGO` | `127` | Primary pull window |
| `JATOS_DISCOVER_IDS` | `1` | Scan studyIds 1–200 on pbsjatos at startup |

## After importing OA/OB/OC studies on pbsjatos

1. Run `python scripts/refresh_jatos_ids.py` (updates `jatos_discovered_ids.json`)
2. Confirm `migration_status` shows fewer missing OBS titles
3. When all 78 titles present, set `JATOS_LEGACY_PULL=0` in Actions

## Legacy site order

Old-server Handler arrays use order **IA, IB, IC, OA, OB, OC** (indices 0–5).
OBS fallback pulls only OA/OB/OC slots whose title is not yet on pbsjatos.

## Dump-recover (parallel track)

Server-side raw not reachable via API → `bahaa jatos dump-recover` from old-server
zip dump. Does not replace nightly pull; complements it for wrong-ID / FINISHED gaps.
