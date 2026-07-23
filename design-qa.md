# Design QA — Open-rPPG ambient component upgrade

Status: **PASSED**

## Reference-to-implementation comparison

| Reference | Implemented target | Result |
| --- | --- | --- |
| React Bits Line Waves | Hero `.line-waves` WebGL layer | Signal-flow character retained; color, opacity, masking, frame rate, and pointer response were reduced to fit the existing instrument interface. |
| React Bits Border Glow | `#pulseLab` state border | Border response retained and mapped to idle, calibration, live, and error semantics without covering controls. |
| Aceternity Card Hover Effect | `.pipeline-focus` shared focus layer | Shared moving highlight retained for pointer and keyboard focus, with no layout movement. |
| Fade Content behavior | Section and card entry animation | Reimplemented with IntersectionObserver and Web Animations API; real-time BPM, waveform, spectrum, and status values are excluded. |

The implementation intentionally keeps the existing Open-rPPG dark instrument system rather than copying the source sites' page layouts.

## Automated verification

- Chrome desktop viewport: 1440 × 1024.
- Chrome narrow viewport: 390 × 844.
- Line Waves initialized with `data-status="ready"` and injected its canvas.
- Chrome console: 0 matched module, syntax, reference, type, or resource-loading errors.
- Reduced-motion mode switched Line Waves to `data-status="static"` and left its container without a WebGL canvas.
- JavaScript syntax checks passed for `app.js`, `ambient-ui.js`, and the local OGL bundle.
- `git diff --check` passed.
- OGL, React Bits attribution, and local license files are present.

## Quality gates

- Visual hierarchy and legibility: passed.
- Desktop and narrow responsive layout: passed.
- Keyboard focus and reduced-motion behavior: passed.
- Decorative layers are non-interactive and hidden from assistive technology: passed.
- Camera/FacePhys lifecycle integration: passed by code-path inspection; capture start pauses Line Waves, and stop/error restores the correct state.
- Static Vercel architecture and local-only model/data behavior: unchanged.

No blocking or high-severity design discrepancies remain.
