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

**Status (2026-10-14): v1, finalised by Claude Code on 정현도's instruction ("어휘도 넣어줘"; he delegated the
choice).** Changes from the 2026-10-11 draft (kept as `action-label_BREAKING_draft0.csv` /
`BREAKING_draft0_text_feature.npy` for comparison runs):

- Visual nodes say what the clip shows in dance context: `a video of a breakdancer doing <move>`
  (draft: `a video of <move>`; e.g. "a swipe" alone is ambiguous for CLIP).
- Audio nodes name the music breaking is danced to: `breakbeat music suitable for <move>`.
- Grades use `breakdancing performance` (CLIP knows "breakdancing" better than "breaking").
- Nodes: "shuffle footwork" → "pretzel footwork"; "a dancer burning the opponent" → "a burn gesture mocking the
  opponent"; "a dancer crashing" → "a crash, losing balance and falling" (execution); added
  "a freeze on the break of the music" (musicality, the WDSF criterion judges cite most), removed nothing else.
  Categories: toprock 4 · go-down 3 · footwork 6 · freeze 6 · power 10 · trick 3 · execution 1 · battle 1 · musicality 2.

Node separation (mean pairwise cosine of the 36 visual move nodes): draft0 0.79, v1 0.91; the shipped ViFi-CLIP
RG/FS features are 0.95, so v1 is in the range MLAVL was trained with. Which vocabulary helps is decided by the battle
experiments, not by this number.

Text features: `tools/battle/make_text_features.py` (vanilla CLIP ViT-B/16 text encoder, raw sentences, rescaled to
the mean row norm of `RG_text_feature.npy`) → `BREAKING_text_feature.npy`, torch.save format like upstream.
