BasinHoppingCatalyst and example3.py Todo:
- make sure mask in the predict method works properly to ensure only top few layers + catalyst atoms are elegible for perturbations
- ensure all resulting perturbations are nonnegative / valid coordinates
- make sure it only accepts a step when it is lower in energy. seems like there is a systematic shift in the atom positions that could be caused by an issue in this regard