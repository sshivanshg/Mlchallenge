# Dataset audit

**Provenance:** `synthetic_dev_fixture_v2`

> If provenance is synthetic, metrics are for research-engine development only — not official challenge scores.

## Sources

### S1
- records: 400 (unique IDs 400, duplicate IDs 0)
- countries: `{'US': 200, 'India': 200}`
- name empty rate: 0.000; addr empty: 0.000
- name non-ASCII rate: 0.000
- addr digit-token mean: 3.00

### S2
- records: 658 (unique IDs 658, duplicate IDs 0)
- countries: `{'US': 328, 'India': 330}`
- name empty rate: 0.000; addr empty: 0.000
- name non-ASCII rate: 0.000
- addr digit-token mean: 2.94

### S3
- records: 553 (unique IDs 553, duplicate IDs 0)
- countries: `{'India': 271, 'US': 282}`
- name empty rate: 0.000; addr empty: 0.000
- name non-ASCII rate: 0.000
- addr digit-token mean: 2.95

## Ground truth
- total S1: 400
- singletons: 135 (33.8%)
- exactly-one: 136; multi: 129
- S2 links: 241; S3 links: 153; both-sources entities: 129
- match-count histogram: `{'0': 135, '1': 136, '2': 129}`
- max matches/S1: 2

## Noise examples (true pairs)

- `S1-00002` ↔ `S3-00001` (India): name `Metro Consultants Ltd` vs `Consultants Metro Ltd`
- `S1-00003` ↔ `S2-00003` (US): name `Velvet Orchid 3 Services Pvt Ltd` vs `Velvet Orchid 3 Services Pvt Ltd`
- `S1-00004` ↔ `S2-00004` (India): name `Marble Badger 4 Holdings LLC` vs `Marbe Badger 4 Holdings LLC`
- `S1-00005` ↔ `S2-00006` (US): name `Premier Services LLC` vs `Premier Services LLC`
- `S1-00007` ↔ `S2-00012` (US): name `National Solutions Inc` vs `National Solutions Inc`
