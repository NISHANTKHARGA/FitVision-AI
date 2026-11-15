# Manual E2E & Edge-Case Test Plan

Legend: ✅ pass · note the observed behaviour for regressions.

## Functional

| # | Scenario | Expected |
|---|---|---|
| 1 | Upload valid person (front-facing) | Pose detected pill; quality badges; overlay not shown until generate |
| 2 | Upload dark/blurry image | "Soft"/"Lighting" warning pill appears; generation still allowed |
| 3 | Upload image with no person | "No pose found" pill; generation proceeds but alignment check will be "unverified" |
| 4 | Upload > 15 MB file | Rejected with 413 before queueing |
| 5 | Upload .gif / .pdf renamed .png | Rejected with 400 ("Could not decode" / unsupported type) |
| 6 | Generate with sample photos, 20 steps | Job completes in ~6 min; three fit images served; analysis sections populated |
| 7 | Generate with both measurement sets filled | Fit analysis verdicts appear; "rules used" expandable |
| 8 | Generate with no measurements | Fit analysis shows "Not enough measurements…" honestly; no invented values |
| 9 | Before/after slider | Drag works; images aligned 1:1 |
| 10 | Save to history → Profile page | Item appears with thumbnail; survives 24 h cleanup |
| 11 | Without saving → restart backend | Result folder deleted on startup (TTL cleanup); history unchanged |
| 12 | Download button | File saved as `fitvision-tryon-<fit>.png` |
| 13 | Delete result | File + DB record removed |
| 14 | GPU OOM (close VRAM hogs) | Job enters `error` state with message; no fake result shown |
| 15 | Network offline after weights downloaded | Generation still works fully offline |
| 16 | Lower-body category | Warp band and region crop move to hips→ankles |

## Visual / layout

| # | Scenario | Expected |
|---|---|---|
| A | 375 px mobile | Stacked layout, no horizontal scroll |
| B | 1280 / 1536 px desktop | Two-column studio; reports beside images |
| C | Results page with long analysis text | No overflow; cards wrap |
| D | Reduced motion | Animations respect `prefers-reduced-motion` (CSS only uses subtle fades) |
