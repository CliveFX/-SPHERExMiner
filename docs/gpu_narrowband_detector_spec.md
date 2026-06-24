# GPU Narrowband Detector — Build Specification

**Detection core:** matched-filter trigger → profile-likelihood test statistic → Gross–Vitells look-elsewhere correction → empirically-calibrated global FAP.

**Target:** replace `blind_classifier_*_warp` as the science candidate generator.

**Status:** build spec. Every section is intended to be implemented as written unless marked *[future]*.

---

## 0. Design thesis

The detector emits, for every candidate, **one number with a defendable meaning: the probability that noise alone would produce a peak this strong *anywhere* in the searched wavelength range, for this target.** That global, look-elsewhere-corrected false-alarm probability (FAP) is the promotion criterion. Everything upstream exists to compute that statistic; everything downstream exists to verify it against measured noise.

This replaces the prior design's additive `rank_score` (which mixed incommensurable units and hid its weights) and its Bayesian evidence framing (which hides the answer inside a prior on line amplitude). A profile-likelihood bump-hunt with a trials factor is the standard machinery for "is there a line *somewhere* in this range," it controls the global false-alarm rate directly — which is the quantity the survey's FP budget is denominated in — and its only free knob (the trials factor) is *measured*, not chosen.

---

## 1. Statistical core

### 1.1 Local model

Work on native per-measurement data, not the assembled spectrum. Index measurements contributing to a local wavelength neighborhood by `i = 1 … n`. Each measurement has flux `d_i`, a noise estimate `σ_i` (Section 3.A), and a known response weight `T_i(λ)` — the predicted *relative contribution* of an intrinsically narrow line at wavelength `λ` to measurement `i`, produced by the same response model as the injector (Section 3.B).

The local data model at trial wavelength `λ`:

```
d_i = c_i + A · T_i(λ) + n_i,     n_i ~ N(0, σ_i²),     A ≥ 0
```

- `c_i` — local continuum, removed by the CFAR estimator (Section 3.A). After continuum subtraction, treat `c_i ≈ 0` in the residual and carry its uncertainty in `σ_i`.
- `A` — line amplitude (µJy). **Physically constrained `A ≥ 0`**: a narrowband source can only add flux. This constraint is not cosmetic; it changes the null distribution of the test statistic (Section 1.4).
- `T_i(λ)` — response weights, normalized so that `Σ_i T_i(λ)² / σ_i²` is the squared template norm used below.

### 1.2 Profiled amplitude and the per-wavelength statistic

Define the whitened inner products:

```
S(λ) = Σ_i  d_i · T_i(λ) / σ_i²          (matched-filter score)
N(λ) = Σ_i  T_i(λ)² / σ_i²               (template norm²)
```

The unconstrained maximum-likelihood amplitude is the matched-filter estimator:

```
Â_unc(λ) = S(λ) / N(λ)
```

with variance `Var(Â) = 1 / N(λ)`. The **non-negativity constraint** gives the profiled estimator:

```
Â(λ) = max( 0 , S(λ) / N(λ) )
```

The matched-filter SNR (the trigger statistic, Stage B) is:

```
ρ(λ) = S(λ) / sqrt(N(λ)) = Â_unc(λ) · sqrt(N(λ))
```

The **profile likelihood ratio** test statistic at `λ`, comparing `H1(λ)` (line of amplitude `A ≥ 0`) against `H0` (no line):

```
q(λ) = -2 [ ln L(H0) − ln L(H1 | Â(λ)) ]
```

For Gaussian noise this reduces to the cleaned-up form:

```
        ⎧ ρ(λ)²      if S(λ) > 0   (Â pinned to interior, Â = Â_unc)
q(λ) =  ⎨
        ⎩ 0          if S(λ) ≤ 0   (Â pinned to boundary, Â = 0)
```

i.e. **`q(λ) = ρ(λ)² · 1[ρ(λ) > 0]`**. This is the half-chi-square that the boundary constraint produces, and it is what every downstream formula assumes. Computing `ρ(λ)` (cheap, linear) and squaring the positive part *is* the profile-likelihood scan for the one-line Gaussian case — the expensive non-linear machinery (Stage E) is only needed when the continuum is jointly refit or when broadened/multi-parameter templates are profiled.

### 1.3 Target statistic — the scan maximum

The detection statistic for the target is the maximum of `q(λ)` over the searched grid:

```
q_max = max_λ  q(λ)
```

`q_max` is large for *some* `λ` even under pure noise, because the scan searched many wavelengths. The job of Section 1.5 is to convert `q_max` into a global false-alarm probability that accounts for that search.

### 1.4 Local significance — Chernoff boundary correction

Because `A ≥ 0` is an active boundary under `H0`, Wilks' theorem does **not** apply in its textbook form. By Chernoff's theorem, the asymptotic null distribution of `q(λ)` at a *fixed* `λ` is a 50/50 mixture of a point mass at zero and a chi-square with one degree of freedom:

```
q(λ) | H0   ~   ½ · δ(0)  +  ½ · χ²₁
```

The point mass at 0 is the probability `S(λ) ≤ 0` (line pinned to the boundary, no evidence). The **local** p-value for an observed value `q` at a single, pre-specified `λ` is therefore:

```
p_local(q) = ½ · P(χ²₁ > q) = ½ · erfc( sqrt(q/2) )
```

Equivalently, in terms of the matched-filter SNR `ρ = sqrt(q)`, the one-sided Gaussian tail `p_local = Φ(−ρ)`. This is the significance you would quote if you had searched exactly one wavelength. You did not.

### 1.5 Global significance — Gross–Vitells look-elsewhere correction

The scan over `λ` introduces a trials factor. The **Gross–Vitells** method estimates the global tail of `q_max` from the geometry of the `q(λ)` random field rather than from a chosen trials count.

The key object is `⟨N(u)⟩`, the *expected number of up-crossings* of level `u` by the field `q(λ)` under `H0` — the number of times the scan curve crosses `u` going upward. For a chi-square field with one degree of freedom this decays exponentially in `u`:

```
⟨N(u)⟩  =  ⟨N_0⟩ · exp( −(u − u_0) / 2 )
```

where `⟨N_0⟩` is the mean up-crossing count at a low reference level `u_0`, **measured once per detector/band from baseline data** (Section 4). `⟨N_0⟩` is the trials factor — it encodes how many effectively-independent resolution elements the scan covered, including the smoothing imposed by the response width and the LVF sampling, none of which a naive "number of grid points" count would get right.

The global p-value (probability that noise produces `q_max ≥ u` *somewhere* in the scan) follows from the expected-Euler-characteristic / up-crossing bound:

```
p_global(u)  ≲  P( ½ χ²₁ > u )  +  ⟨N_0⟩ · exp( −(u − u_0) / 2 )
             =  ½ · erfc( sqrt(u/2) )  +  ⟨N_0⟩ · exp( −(u − u_0) / 2 )
```

The first term is the single-test floor; the second is the look-elsewhere inflation. At the high thresholds relevant for a SETI claim the second term dominates, and `p_global ≈ ⟨N_0⟩ · exp(−q_max/2)` to good approximation.

**Measuring `⟨N_0⟩`.** On baseline (injection-free) spectra, run the scan, pick a reference level `u_0` (Section 8, open question 4), and count up-crossings of `u_0` per scan; average over targets within a detector/band. One scalar per detector/band. The Gross–Vitells prediction is then *testable*: the observed `q_max` tail on baseline data must match `⟨N_0⟩ · exp(−q_max/2)`. Section 7 makes this a validation metric; if it fails, fall back to a fully empirical per-detector `q_max` threshold (already supported by the Section 4 null), so a Gross–Vitells breakdown degrades gracefully rather than breaking the pipeline.

### 1.6 Promotion criterion

```
promote(target)  iff  p_global(target) < α_global
```

`α_global` is fixed once, from the survey FP budget (Section 7), against the **empirical** null (Section 4) — never against the asymptotic χ² forms above. The asymptotics are the *trigger*; the measured null is the *decision*.

> **Why this over Bayesian evidence.** The Occam factor in the evidence integral is set by the amplitude prior — a quantity you invent. The trials factor `⟨N_0⟩` here is set by the number of resolution elements you actually searched — a quantity you measure. For a detection you intend to *claim*, a measured trials factor is defensible in a way a chosen prior is not.

---

## 2. Pipeline stages

Cheap, decisive gates first; the expensive scan and calibration only on survivors.

| Stage | Operation | Scope |
|-------|-----------|-------|
| A | Continuum + local noise (CFAR) | GPU, all targets |
| B | Matched-filter trigger | GPU, all targets → sparse top-K peaks |
| C | Shape gate (response-bin χ²) | GPU, peaks only |
| D | Aperture/PSF agreement gate | GPU, peaks only |
| E | Profile-likelihood scan `q(λ)` | GPU, gated targets, local grid |
| F | Look-elsewhere → `p_global` | GPU/CPU, gated targets |
| G | Source-frame recurrence gate | CPU join, ≥2-visit targets |
| H | Write sparse candidate parquet + debug | CPU |

The matched filter (B) is a fast linear pre-filter that finds *where* to do the profile scan (E). They are not redundant: B triggers, E decides, F corrects, G confirms. For the pure one-line Gaussian case, E collapses to `q(λ) = ρ(λ)²·1[ρ>0]` (Section 1.2) and is nearly free; E earns its cost only when the continuum is jointly refit or broadened/multi-parameter templates are profiled.

---

## 3. Stage detail

### 3.A Continuum and local noise (CFAR)

GPU-friendly local robust background, radar-CFAR structure:

- exclude **guard cells** spanning the candidate response support;
- estimate continuum from **training cells** in a surrounding window (median or trimmed mean);
- estimate local scatter from MAD and, separately, from propagated `VARIANCE`.

**Size windows in units of template support, not fixed nm** — guard ≈ 1.5× template FWHM, training ≈ N independent resolution elements — because channel density varies across the LVF and a fixed-nm window is a variable-element window. Verify training cells are line-free; near a real cluster they contaminate the continuum and bias it high, self-suppressing signal.

Emit **two** noise estimates per measurement (`sigma_mad`, `sigma_var`). Their ratio is a diagnostic for non-Gaussian local noise / continuum mismodeling; large divergence flags the measurement.

> ⚠️ Matched-filter optimality and the look-elsewhere asymptotics both assume the whitened residuals are independent. If neighboring support measurements share systematics (common readout, overlapping PSF wings), off-diagonal covariance is nonzero and diagonal whitening overstates `q`. Either whiten with a local covariance estimate or — the cheaper path — let the **empirical null (Section 4) absorb the correlation**, which it does automatically because `⟨N_0⟩` and the `q_max` tail are measured on the same correlated data. This is a second reason the empirical null is load-bearing.

### 3.B Matched-filter trigger

Per (target, flux_kind, template): local continuum subtraction → whitening → matched-template amplitude `Â_unc(λ)` → matched-filter SNR `ρ(λ)` (Section 1.2). Emit **sparse top-K peaks per target**, never a dense target×λ×template cube (except explicit debug).

Templates predict **relative contribution across native measurements** `T_i(λ)`, not nearest-channel membership. Bank families: unresolved line at native response width; slightly broadened; one-sided/undersampled (edge, sparse support). Grid: 1 nm validation / coarser preview / dense local refinement at peaks.

**Single source of truth:** the template bank and the FITS injector call the *same* response-generation function, version-hashed, hash written into both the candidate table and injection truth. If they drift, injection/recovery measures a shared bug, not sensitivity.

### 3.C Shape gate — response-bin χ²

Partition template support into bins of equal *expected* contribution; observed excess must distribute power across bins like the template. A single hot pixel or flag-adjacent spike fails this. Cheap, runs on every triggered peak, kills the dominant within-visit artifact class before the expensive scan.

### 3.D Aperture/PSF agreement gate

Score aperture and PSF independently. Require agreement for the science tier; allow single-channel as a lower review tier; penalize extreme flux ratios; retain both amplitudes and SNRs. A real source appears in both photometries; many detector artifacts do not.

### 3.E Profile-likelihood scan

On targets surviving C+D, scan `q(λ)` over a **local refined grid** around each surviving peak (not the whole band — the gates already localized it). Profile `Â(λ) ≥ 0` at each grid point. For the one-line Gaussian case this is `ρ(λ)²·1[ρ>0]` and nearly free; the non-linear cost appears only for joint continuum refit or broadened/multi-parameter templates. The survivor set is small (C+D ensure that), which is what makes the gate-first ordering pay off.

### 3.F Look-elsewhere → global FAP

Apply Section 1.5 with the per-detector/band `⟨N_0⟩` from calibration. Emit `q_max`, `p_local`, `p_global`, and the `⟨N_0⟩` used. Threshold `p_global` against the empirical null.

### 3.G Source-frame recurrence gate

For targets with ≥2 independent visits, **require recurrence at the same source-frame wavelength** for science-tier promotion — hard gate, not a bonus. Under barycentric correction across visits:

- a **beacon / astrophysical line** is stationary in source-frame wavelength;
- a **terrestrial laser** is stationary in *observed* wavelength (drifts in source frame);
- a **detector artifact** is stationary in *detector* coordinates (drifts in both).

This single physical test subsumes the prior design's separate detector-frame-recurrence veto, most flag penalties, and the artifact-template family.

> ⚠️ **Hard dependency:** correctness of the barycentric/source-frame correction, applied identically across visits, with correction parameters logged per measurement. This is a **blocker** for the gate, not optional. Single-visit targets cannot use the gate → routed to lower tier, can never reach top science tier on shape alone. Correct conservative posture for a SETI claim.

---

## 4. Empirical null calibration — do this before any threshold work

Every threshold (`α_global`, the trigger SNR floor, the reference level `u_0` for `⟨N_0⟩`) is meaningless without a measured null.

1. Run Stages A–F on **injection-free real baseline** spectra.
2. Build the empirical distribution of `q_max` and the up-crossing count `⟨N_0⟩` **per detector / camera / band** — noise is non-stationary across the LVF.
3. Express `α_global` as a tail probability against that measured `q_max` distribution.
4. **Negative control / anti-injection:** inject signal at wavelengths/detectors where no real source can exist; confirm Stage G kills them. Directly tests the strongest gate.

Because `⟨N_0⟩` and the `q_max` tail are *measured* on correlated, systematics-bearing baseline data, they absorb the covariance and non-Gaussianity that the asymptotic formulas in Section 1 ignore. The asymptotics are only ever used as the GPU trigger; the measured null is the truth the decision is made against.

> ⚠️ The empirical null is only as good as the baseline's coverage of rare artifacts — the worst false positives live in the tail, where any finite baseline is thinnest. *[future: model the tail with peaks-over-threshold / generalized-Pareto rather than extrapolating the bulk.]*

---

## 5. GPU architecture

Warp-first: ragged data, existing photometry stack is Warp.

```
load compact arrays
  → build/load template metadata
  → [A] GPU CFAR continuum + dual noise
  → [B] GPU matched-filter trigger (aperture + PSF) → top-K
  → [C][D] GPU shape + aperture/PSF gates on peaks
  → [E] GPU profile-likelihood scan on survivors (local grid)
  → [F] GPU/CPU look-elsewhere → p_global
  → [G] CPU source-frame join (≥2-visit)
  → [H] write sparse parquet + local debug windows
```

Never move dense score cubes to CPU; save local windows for the viewer or recompute on demand. CuPy acceptable for CFAR/convolution prototyping; production Warp-first absent a benchmark.

> ⚠️ **The real implementation difficulty is ragged support.** Per-target support counts vary (edge/one-sided cases), so a thread-per-(target,template) kernel suffers warp divergence and load imbalance. Bucket targets by support size, or pad-and-mask to fixed width; benchmark before committing the kernel layout. The non-negative-amplitude profiling in E is a per-peak 1-D constrained fit — cheap individually, but it is a *non-linear* kernel; keep the survivor set small (C+D do that) so E never dominates.

---

## 6. Inputs and outputs

**Inputs:** `target_id`; RA/Dec + Gaia metadata; aperture & PSF flux arrays; `VARIANCE`-derived uncertainties; **per-measurement** `cwave`/`cband`; field/image IDs; detector/camera/band/frame metadata; pixel coords + edge distance; raw **and** decoded flags; **per-measurement barycentric-correction parameters**; injection truth (validation only). Wavelength modeled per camera/detector/measurement — never a global grid.

**Candidate table:** `target_id`, `line_nm_source_frame`, `template_id`, `tier`, `q_max`, `p_local`, `p_global`, `N0_used`, `aperture_snr`, `psf_snr`, `aperture_amp_uJy`, `psf_amp_uJy`, `response_bin_chi2`, `n_visits`, `source_frame_recurrence_pass`, `support_count`, `flagged_points_sum`, `sigma_ratio_flag`, `detectors`, `frame_ids`, `response_model_version_hash`, `reject_reasons`.

**Debug:** local `q(λ)` scan window per candidate; CFAR continuum used; template values over support; aperture/PSF residual vectors; per-visit source-frame wavelengths (makes Stage G auditable); injection-truth join when available.

**Viewer:** aperture + PSF spectra together; template overlay on measurements; the local `q(λ)` scan (not just the peak); `p_global` + `⟨N_0⟩` + visit count + source-frame recurrence plot; decoded flag explanations; visual distinction between raw science candidates, injected raw recovery, paired-delta recovery.

---

## 7. Validation — ordered to avoid circularity

1. **Measure the empirical null + baseline `q_max` distribution + `⟨N_0⟩` first** (Section 4) — before any injection sweep, so the sweep reads against a fixed false-alarm scale.
2. Negative control / anti-injection (Stage G kills impossible-location signal).
3. Paired-delta recovery still finds known injections (injector sanity).
4. **Blind raw** injected recovery finds them without subtraction.
5. Sweep injected flux, wavelength, magnitude, detector, field density.
6. Set `α_global` to the FP budget — now well-defined, the null is fixed.
7. Freeze a holdout before final threshold evaluation.

**Metrics:** recovery fraction by injected SNR/wavelength; **measured vs. predicted trials factor** (does Gross–Vitells `⟨N_0⟩` match the observed `q_max` tail? — this validates the core statistic); FP per target / wavelength-interval / field / detector; median wavelength error; amplitude bias; aperture/PSF agreement rate; **marginal** rejection rate of each gate (confirm none silently kills real signal); source-frame-gate pass rate, real vs. injected.

---

## 8. Open questions, ordered by what they block

1. **Barycentric/source-frame correction representation in QR2** — *blocks Stage G, the decisive gate.* Answer first.
2. **Field-scale FP budget** → sets `α_global` (Sections 1.6, 4, 7).
3. Is matched-filter support correlated enough to need non-diagonal whitening, or does the empirical null absorb it? (Section 3.A)
4. Reference level `u_0` for the `⟨N_0⟩` up-crossing count — low enough for good up-crossing statistics, high enough for the exponential-decay asymptotic to hold.
5. Continuum: local windows only, or stellar-template-first for bright stars? (Stellar lines are the main contaminant of both the gate and *[future]* population detrending.)
6. Source-noise update for injected FITS products.

---

## 9. Future work

- **Population detrending** (robust PCA / low-rank-plus-sparse in detector space) as a pre-conditioning stage owning systematics removal, *gated on the SVD singular-value diagnostic confirming the systematics are actually low-rank in LVF geometry.* Would clean the null upstream and answer "is this candidate also on unrelated targets" natively rather than by downstream coincidence. Carries a SETI-specific risk — a real signal common across targets is coherent and could be absorbed as a systematic — so it would run twice (detector-frame and source-frame) and subtract only what is detector-coherent *and* source-frame-incoherent.
- **Model-agnostic excess-power channel** as a second discovery path with its own null, to catch beacon morphologies the template family cannot represent (pulsed, drifting, structured). The matched filter answers "is the line I imagined here"; an excess-power channel answers "is there anything here I didn't imagine."
- **Peaks-over-threshold tail modeling** of the empirical null (Section 4), for the rare-artifact tail a finite baseline under-samples.

---

## Appendix — symbol reference

| Symbol | Meaning |
|--------|---------|
| `d_i` | continuum-subtracted flux of measurement `i` |
| `σ_i` | per-measurement noise (`sigma_mad` / `sigma_var`, Section 3.A) |
| `T_i(λ)` | response weight: relative contribution of a line at `λ` to measurement `i` |
| `A`, `Â(λ)` | line amplitude (µJy); profiled non-negative estimator |
| `S(λ)` | matched-filter score, `Σ d_i T_i / σ_i²` |
| `N(λ)` | template norm², `Σ T_i² / σ_i²` |
| `ρ(λ)` | matched-filter SNR, `S/√N` |
| `q(λ)` | profile likelihood-ratio statistic, `ρ²·1[ρ>0]` |
| `q_max` | scan maximum of `q(λ)` — the target detection statistic |
| `p_local` | single-wavelength p-value, `½·erfc(√(q/2))` |
| `⟨N(u)⟩` | expected up-crossings of level `u` by the `q(λ)` field |
| `⟨N_0⟩` | up-crossing count at reference level `u_0` — the measured trials factor |
| `p_global` | look-elsewhere-corrected FAP — the promotion quantity |
| `α_global` | promotion threshold, set from the FP budget against the empirical null |
