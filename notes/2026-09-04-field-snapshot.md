# Field snapshot — 2026-09-04 (day 4 of 41)

Pulled with `python -m prc.leaderboard`. Latest scored submission at the time:
`2026-09-04T17:33:43Z`.

731 scored submissions from **37 teams**, against **256 registered**. So ~86% of
registrations have not submitted anything yet. The 2024 edition drew 132 teams
in total, so registrations are already running well ahead of that; whether
*active* teams do is the open question.

| rank | RMSE (s) | team | subs |
|-----:|---------:|------|-----:|
| 1 | 249.4600 | enthusiastic-daisy | **542** |
| 2 | 250.5233 | quick-boat | 17 |
| 3 | 253.1779 | jovial-uniform | 8 |
| 4 | 258.4399 | merry-rose | 8 |
| 5 | 259.8263 | intelligent-ladder | 4 |
| 6 | 260.1320 | reliable-hamburger | 10 |
| 7 | 264.6420 | upbeat-goblin | 6 |
| 8 | 264.8846 | optimistic-panda | 9 |

Team-best percentiles: p0 249.5 · p10 259.8 · p25 275.5 · **p50 319.5** · p75 481.9
· p90 561.5 · p100 726.3.

## Reading

- **The top is tight, the middle is not.** 1st→3rd is 3.7 s; 1st→median is 70 s.
  Most of the field has not done the obvious work yet, so early placement is
  cheap and late placement will not be.
- **542 submissions in four days is an automated sweep**, not a person iterating.
  With no private split and no visible submission cap, that team is probing the
  ranking set directly. Some of their margin over 2nd (1.06 s) is likely fitting
  the 215,876 evaluation rows rather than modelling taxi-out.
- **Consequence for us:** a genuine top-10 is reachable on modelling alone. First
  place is a grinding contest we would have to opt into deliberately, and it is
  worth deciding *up front* whether we are willing to, because it changes what
  the last two weeks look like.
- Baseline sense-check to run once data lands: per-airport-hour mean taxi-out.
  If that lands near the p75 of 482 s, the field's median team is barely beating
  a group-mean, which would say a lot about how much headroom is real.
