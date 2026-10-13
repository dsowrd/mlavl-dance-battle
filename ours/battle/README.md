# Battle metadata and winner labels (ours)

Built by `tools/battle/build_battle_labels.py` from public sources. Videos themselves are **not** in this repository
(BRACE is S-Lab License 1.0, non-commercial; YouTube videos are used for research only and never redistributed).

## Files

- `battles.csv`: one row per BRACE video. Every BRACE video is one 1-vs-1 Red Bull BC One battle (81 battles).
  `dancer_a` is the dancer who danced first in the BRACE sequences, `sequence_uids` lists the BRACE sequences
  (one per round/turn), `round` is Top 16 / Quarterfinal / Semifinal / Final.
- `winners.csv`: winner of each battle, parsed from the Wikipedia brackets, plus the bracket score.
  `verified_by` is empty until a human (정현도) has checked the row.

## Sources

| What | Source |
| --- | --- |
| Battles, dancers, sequences, keypoints | BRACE annotations, https://github.com/dmoltisanti/brace (`annotations/sequences.csv`, `videos_info.csv`) |
| Winners and judge votes | English Wikipedia "Red Bull BC One", revision in the `source` column (`oldid=1376654426`, 2026-09-25) |
| Round names, cross-check | Official Red Bull BC One YouTube titles (`yt-dlp --print title`), e.g. "Battle 9" = quarterfinal |

Parse check (2026-10-13): 81 / 81 battles matched a bracket entry, round from the bracket position agrees with the
round in the YouTube title for all 81, and the parsed champions agree with the known champions
(2011 Roxrite, 2013 Hong 10, 2014 Menno, 2017 Menno, 2018 Lil Zoo, 2020 Shigekix / B-girl Kastet).

## Notes for using the labels

- Scores are judge votes. 2011–2013 and 2017–2020: one number per dancer (e.g. 3–2 of 5 judges).
  **2014 used round-by-round judging**: `5/4` means votes per round, the battle ends when one dancer wins 2 rounds;
  `*` (stripped here) marked tie-break rounds on Wikipedia. A vote margin can be used as a soft label only for
  the non-2014 years.
- Split: BRACE's own train/test split is per sequence and puts most videos in both halves, so it cannot be used for
  battle-level evaluation. Use leave-one-year-out (6 folds); the same dancers appear in several years
  (e.g. Menno, Lil Zoo, Wing), so dancer identity leakage across folds still exists and should be reported.
- `winner_is_a` is 1 in 59% of battles (the first dancer wins slightly more often), so a "first dancer wins"
  baseline is 59%, not 50%.
- 2020 contains 7 B-boy and 7 B-girl battles (8-dancer brackets). All other years are B-boy only.
