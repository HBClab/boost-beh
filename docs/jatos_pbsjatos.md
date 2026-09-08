# pbsjatos migration

BOOST behavioral pipeline primary pull target is **pbsjatos**
(`https://pbsjatos.psychology.uiowa.edu`).

**Legacy host (`jatos.psychology.uiowa.edu`):** still reachable (HTTP 303 as of
2026-09-08; Sydney can use the UI). Prefer pbsjatos for new collection + nightly.
Legacy pull env (`JATOS_LEGACY_*`) is optional for OBS studies not yet imported.

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
| `JATOS_LEGACY_PULL` | `0` | Set `1` only if intentionally pulling OBS from old host |
| `JATOS_DAYS_AGO` | `127` | Primary pull window |
| `JATOS_DISCOVER_IDS` | `1` | Scan studyIds 1–200 on pbsjatos at startup |

Legacy env vars (`JATOS_LEGACY_TOKEN`, etc.) remain in code for reference only; unused while old server is down.

## After importing OA/OB/OC studies on pbsjatos

1. Run `python scripts/refresh_jatos_ids.py` (updates `jatos_discovered_ids.json`)
2. Confirm `migration_status` shows fewer missing OBS titles
3. When all 78 titles present, migration complete (no legacy fallback needed)

## Legacy site order

Old-server Handler arrays use order **IA, IB, IC, OA, OB, OC** (indices 0–5).
OBS fallback pulls only OA/OB/OC slots whose title is not yet on pbsjatos.

## Dump-recover (parallel track)

Server-side raw not reachable via API → `bahaa jatos dump-recover` from old-server
zip dump. Does not replace nightly pull; complements it for wrong-ID / FINISHED gaps.
