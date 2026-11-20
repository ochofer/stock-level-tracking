"""
Code copied verbatim from the source repository, github.com/ochofer/optimal-vs-naive-diversification at commit
5414f14, so that this study runs the same estimators as the one it extends. paired.py, the paired test of two
realised tracking errors, is copied from the same repository at commit ca0ec64, on 20 November 2026. It was
added there at commit 0e651a0 on 18 November 2026, and the comment above constants.COPIED_AT_COMMIT says what
has changed in it since.

Each copy's first line names its source file and the commit; every line after it is the source file unchanged,
and tests/test_source_copies.py hashes it against constants.COPIED_FILES. The copies import their constants as
`from . import constants as C`, by the source's names, so constants.py in this folder holds those names and
nothing else, each at the source's value. The study's own modules call the copies with the study's own
parameters and never read the names below.
"""
