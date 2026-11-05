"""Photometric definitions of an impact-flash luminous energy.

Published luminous energies are not one quantity. They differ in bandwidth, isotropy factor and
integration window, and the luminous efficiency eta is only meaningful for the definition it was
derived with (Madiedo et al. 2019, MNRAS 486, 3380, sec. 4.1). The engine's eta range
(5e-4 to 6e-3; the 6e-3 upper bound is not independently verified) is applied in the
'luminous_range_integrated' definition. An energy given in another definition is converted with an uncertain factor F, log10 F ~ uniform(lo, hi):

    E_luminous_range = F * E_reported

The factor ranges are:

* msfc_peak_frame_R (Suggs et al. 2014, Icarus 238, 23, eqs. 5-6): energy in the brightest video
  frame from the peak R magnitude, dlambda = 1607 A. Converting it needs a band factor (blackbody
  flash, 2000-5000 K: 1.07 to 3.2) and the ratio of total to peak-frame energy. That ratio is an
  assumption of 1.5 to 5; it is 3.0 for the 11 Sep 2013 light curve.
* integrated_0.3um (Sheward et al. 2022, MNRAS 514, 4320, eq. 2): whole flash, dlambda = 0.3 um.
  The factor is 0.5/0.3 if the magnitudes are V-calibrated; the band is not stated, hence 1.4 to 2.4.
"""

from __future__ import annotations

FLASH_DEFINITIONS: dict[str, dict] = {
    "luminous_range_integrated": {
        "label": "Integrated over the flash, 0.5 µm luminous range, f = 2",
        "log10_factor": (0.0, 0.0),
        "source": "Ortiz et al. 2000; Madiedo et al. 2014, 2019",
    },
    "msfc_peak_frame_R": {
        "label": "Peak video frame, R band (NASA MSFC, Suggs et al. 2014)",
        "log10_factor": (0.206, 1.199),
        "source": "Suggs et al. 2014 eqs. 5-6",
    },
    "integrated_0.3um": {
        "label": "Integrated over the flash, 0.3 µm bandwidth, f = 2",
        "log10_factor": (0.146, 0.380),
        "source": "Sheward et al. 2022 eq. 2",
    },
}
DEFAULT_FLASH_DEFINITION = "luminous_range_integrated"
