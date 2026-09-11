# depth-4 was fitting seven rows, and the gate reparameterisation fails outright

2026-09-11

Both results came from the same run: every arm scored twice, once on all no-plan
rows and once with `y >= 80,000` removed. That second column is new, and it
killed a submission that was already building.

## depth-4 does not survive the exclusion

    === ALL no-plan rows ===
      depth4   mean 1574.2   vs ctrl  -38.5    -59.8   -59.2    +3.5   wins 2/3

    === EXCLUDING y >= 80,000 ===
      depth4   mean 1043.2   vs ctrl  +17.3     +6.1   +48.5    -2.9   wins 1/3

The folds hold 6, 2 and 3 of those rows. A shallower tree averages harder at the
top of the `gap_sched` staircase, which flatters it on exactly the rows the
scored months appear not to contain.

**v12 was at iteration 500 when this landed and has been killed.** It would have
shipped a change whose entire measured gain was fitting seven rows.

The 2026-09-06 grid had depth 4 losing; the sweep three days later had it
winning by 38s; with the monsters removed it loses again. The middle reading was
the anomaly, and only the new second column distinguishes them.

## The gate reparameterisation is worse, by a lot

    gate      mean 2280.3   vs ctrl  +667.6
    gate_d4   mean 2284.2   vs ctrl  +671.4     both 0/3, and 0/3 excluding monsters

Predicting `G = BLOCK - SCHED` and returning `gap_sched - G` is algebraically
exact -- verified to 0.000000 on all 1,488 LIRF no-plan rows -- and it is a
catastrophe in practice. An error in G passes straight through to y at
coefficient 1, and where `gap_sched` is large and stale the model cannot predict
G at all.

`likable-eagle` reports it taking their live score 316.97 -> 288.90, so it works
for them. Two differences worth noting before writing it off completely: they fit
the G-model **only on unmatched rows**, and they apply it **only at LIRF**. We
applied it across all no-plan rows including the nine airports where `gap_sched`
is often a stale long-haul schedule and D - G is meaningless. A LIRF-only variant
is the version their evidence actually supports; this test does not refute that.

## What the second column is worth

It cost 40 seconds a fold and it caught a submission mid-build. Every fold
comparison from here reports both numbers. The rule is in
`2026-09-11-the-offset-is-seven-rows.md`: a change whose gain disappears when
seven rows are removed is a change that fits seven rows.
