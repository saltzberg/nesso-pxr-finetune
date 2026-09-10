# Both-member capture and supervised training

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Can both actual affinity members be captured and adapted reproducibly?

## Design and fitted/frozen flow

Eight predetermined FIT records → frozen trunk capture → each member’s tensors/RNG → continuous heads [fitted] with/without rank-4 adapters at esm_proj.1, esm_proj.3 and first transition fc1.

## Results

Eight of eight records and both supervised arms completed. Native/differentiable, zero-update and saved-state replay discrepancies were exactly zero. Trainable parameters: 592,130 head-only; 616,706 with adapters.

## Implication and limits

Two SGD updates on eight records are engineering feasibility, not held-out utility. Binary weights stay frozen, but upstream changes can change binary outputs.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
