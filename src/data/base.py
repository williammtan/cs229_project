"""Dataset-base placeholder.

WAY-EEG-GAL is the only dataset wired up today. The interface is intentionally
left as a docstring rather than an ABC until a second dataset (Jeong 2020,
Müller-Putz Graz) is integrated, so we don't pre-commit to an API.

A second dataset module should expose:
    load_subject(raw_dir, subject, ...) -> SubjectData
    load_dataset(raw_dir, subjects, ...) -> dict[int, SubjectData]
where SubjectData.trials is a list of ``src.data.way_eeg_gal.Trial`` (or a
compatible dataclass with the same fields). Splits and protocols will then
work unchanged.
"""
