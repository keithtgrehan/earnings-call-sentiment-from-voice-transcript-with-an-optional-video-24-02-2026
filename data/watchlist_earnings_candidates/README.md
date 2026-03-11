# Watchlist Earnings Candidates

This package is a metadata-first candidate pool built only from the supplied watchlist, while excluding companies already used in the frozen benchmark and current holdout benchmark unless explicitly requested elsewhere.

## Purpose
- keep future candidate sourcing separate from the active frozen benchmark and current holdout benchmark
- record official or primary-source references before any future transcript collection or gold labeling
- leave `labels.csv` non-canonical and empty until transcript or excerpt evidence is supportable

## Current package contents
- `call_manifest.csv`: candidate rows drawn from the watchlist and not already used in the active benchmark packages
- `official_source_manifest.csv`: source-of-truth metadata for candidate dates and official URLs
- `transcription_status.csv`: excerpt/transcript collection state for each candidate row
- `labels.csv`: header-only placeholder; no new gold labels were appended in this run

## Watchlist classification
| ticker | resolved company name | candidate type | notes |
|---|---|---|---|
| NVDA | NVIDIA | duplicate_already_used | Already used in the frozen benchmark and holdout. Handled separately in `data/nvda_2025_historical_calls/`. |
| TSFA | Taiwan Semiconductor Manufacturing Co. Ltd. (Frankfurt listing likely) | ambiguous | Exchange-specific symbol without a clean local investor-relations convention in this repo. |
| PLTR | Palantir | duplicate_already_used | Already used in the frozen benchmark. |
| EUED | iShares € UltraShort Bond ESG SRI UCITS ETF | non_equity / fund / ETF / ETP | ETF, not a normal earnings-call operating company. |
| GOOG | Alphabet | duplicate_already_used | Already represented in the frozen benchmark/holdout via Alphabet (`GOOGL` share-class noted). |
| 4GLD | Xetra-Gold | non_equity / fund / ETF / ETP | Exchange-traded commodity, not an operating-company earnings candidate. |
| AMZN | Amazon.com, Inc. | operating_company | Added as a new watchlist candidate. |
| NBIS | Nebius Group | operating_company | Added as a new watchlist candidate. |
| ETN | Eaton | operating_company | Added as a new watchlist candidate. |
| MSFT | Microsoft | duplicate_already_used | Already used in the frozen benchmark and holdout. |
| LEU | Centrus Energy Corp | duplicate_already_used | Already used in the frozen benchmark. |
| IREN | IREN | operating_company | Added as a new watchlist candidate. |
| FEIM | Frequency Electronics | operating_company | Added as a new watchlist candidate, but direct page access was blocked during this run. |
| LMND | Lemonade | operating_company | Added as a new watchlist candidate. |
| BE | Bloom Energy | operating_company | Added as a new watchlist candidate. |
| OKLO | Oklo | operating_company | Real operating company, but skipped in this pass because the locally surfaced source material is a quarterly company update deck rather than a standard earnings-call path. |
| VRT | Vertiv | operating_company | Added as a new watchlist candidate. |
| SIE | Siemens AG (likely Xetra ticker) | ambiguous | Likely Siemens AG, but the bare ticker is exchange-specific and was left out to avoid identity drift. |

## Notes
- No new labels were appended in this package during this run.
- Several rows were collected only as official-source excerpts because direct media or transcript paths were not available through the current environment.
