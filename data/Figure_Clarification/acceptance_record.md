# Mesh clarification acceptance

Date: 2026-09-07. Status: COMPLETE_WITH_SCOPE_LIMITS.

- Registered before execution: five spacings 0.4/0.2/0.1/0.05/0.025 m, with
  0.0125 m reference, for RECT_FULL, RECT_PARTIAL and GRADED.
- Completed 18 mesh/profile combinations and stored 15 scored comparisons.
  Original load states are retained within each profile. No new inverse
  observation set or parameter fit is counted.
- Maximum strain relative L2 at 0.05 m: 5.850305572357825e-5, scored over
  [-12,12] m against the finer mesh across the original load states.
- Inputs are the unchanged Step08A FE support arrays; their SHA-256 values are
  stored in each result row. The script hash matches `registration.json`.
- The source CSV and manuscript plot export have identical rows; Figure 2
  plots actual values, not fitted or interpolated evidence points.
- This is a discretization check within the same linear beam--foundation
  physics, not independent physical validation or convergence-order proof.
- QA: Step12 `qa/verify_draft.py` checks record counts, mesh levels, input and
  script hashes and figure-data identity. Current status: PASS.
