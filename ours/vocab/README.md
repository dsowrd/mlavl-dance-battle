# Breaking move vocabulary (draft, 2026-10-11)

`action-label_BREAKING.csv` replaces the figure-skating / rhythmic-gymnastics action list that MLAVL feeds
into the MAG² graph. Layout follows upstream `action-label_RG.csv` exactly so the same slicing code works:

| rows | content | count |
| --- | --- | --- |
| 0–35 | `a video of <move>` (visual action nodes) | M = 36 |
| 36–71 | `a music suitable for <move>` (audio action nodes) | M = 36 |
| 72–83 | `a video of <grade> breaking performance with/without fitting music` (AV grades) | 2N = 12 |
| 84–89 | `a video of <grade> breaking performance` (V grades) | N = 6 |

Total 90 rows → run with `--n_text 36 --n_query 6`.

The `category` column is for us only (not read by the model):
toprock · transition (go-downs) · footwork · freeze · power · trick · negative (crash) · battle (burn/callout) · musicality.

**Status: draft written by Claude Code. 정현도 (a street dancer) reviews and finalises the list before
CLIP text features are extracted.** Points to decide:

- Keep or drop the three non-move nodes (crash, burn, hitting the beat). They are what judges actually look at,
  but they are not "actions" in the MLAVL sense.
- Granularity: 36 moves is between RG (32) and FS (42). More nodes = bigger graph, slower MAG² loop.
- Spelling of move names in English matters for CLIP (e.g. "air chair" vs "airchair", "turtles" vs "crickets").

Text features are extracted once with the CLIP / ViFi-CLIP text encoder (`--use_clip`) and saved as
`BREAKING_text_feature.npy` (torch.save format, like upstream).
