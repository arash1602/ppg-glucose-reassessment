# Additions (pulsatile signal, confounders, extended sweep)

## New / changed files
| File | What |
|---|---|
| `src/analysis/physio_confounders.py` | DC and pulsatile (AC) glucose-equivalent errors, net analyte signal, scattering-assumption scan |
| `src/main_extended.py` | sweep 660, 760, 810, 850, 880, 940, 970 nm + 1000-1700 nm, 5 batches |
| `Water_Absorption_Data_Public/` | Segelstein (1981) water spectrum (public, CC0) used when the git-ignored `Water_Absorption_Data/` is absent |
| `src/tissue_model/exp_data.py` | falls back to the public water file if Kou/Fewell files are missing |
| `run_extended_sim.bat` | Windows one-click run |

## Method (first-order / Jacobian)
Mean partial pathlength in layer k = dA/dmu_a,k (A = -ln R); perturbation-MC gives dA/dmu_s,k.
Any small change of absorption/scattering -> dA without new simulations. Each confounder's dA is
divided by the glucose sensitivity dA/dCg -> glucose-equivalent error in mg/dL (no arbitrary weights).

Net analyte signal (Lorber 1986): the part of the glucose sensitivity vector (over wavelengths x
detector rings) orthogonal to all confounder vectors. Best achievable error = noise / |NAS|.

## Key results from the existing 1000-1700 nm runs (assumptions in results/physio_analysis/assumptions.txt)
* Single wavelength, DC: best ~70 mg/dL (1090 nm, 2.9 mm), and this relies entirely on the assumed
  glucose scattering coefficient (-0.29 %/mmol/L). With glucose ABSORPTION only: >= ~790 mg/dL.
* Glucose's scattering effect lies in the same direction as any other dermal scattering change ->
  removed completely by the net-analyte-signal analysis: only absorption is glucose-specific.
* Per wavelength, all dermal absorbers give the same pattern across detector rings, so separating
  glucose from water + blood volume + SO2 needs spectral diversity: >= 4 wavelengths;
  5 wavelengths reach ~18 mg/dL in this (idealised, linear, 0.05 % noise) model.
* Pulsatile (AC) channel: at 1550-1700 nm blood and water absorb almost equally, so the pulse nearly
  vanishes and a 0.5 % change in blood water content swamps glucose (~400-550 mg/dL equivalent).
  Below 1300 nm the glucose effect on the pulse is negligible.
* Photon shot noise is negligible (<0.01 mg/dL): the limit is physiology and drift, not detectors.

## Not yet included
Temperature-dependent water spectrum (shifts the 1450 nm band; likely the largest confounder),
melanin / epidermal thickness / dermal thickness variation, contact pressure, glucose absorption < 1000 nm.
