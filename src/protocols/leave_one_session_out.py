"""Within-subject leave-one-session-out protocol — the canonical EEGMMI
within-subject metric.

EEGMMI's imagery runs come in 3 sessions, each containing one L/R-fist run and
one BothFists/BothFeet run:

    Session 1: runs {4, 6}
    Session 2: runs {8, 10}
    Session 3: runs {12, 14}

We hold out one session at a time per subject, train on the other two
sessions, and test on the held-out session. This:

  * Keeps all 4 classes balanced in train and test (each session covers all
    four MI conditions);
  * Preserves the temporal structure between recording sessions — train and
    test never come from the same recording block, so the metric reflects
    realistic within-subject generalisation across sessions;
  * Yields 3 folds × N_subjects splits, all subject-dependent.

This is what EEGNet (Lawhern 2018) and most subsequent EEGMMI papers report as
the "within-subject" decoding accuracy.
"""
from __future__ import annotations

from typing import Iterator

from src.core.registry import register
from src.core.types import Split
from src.data.eegmmi import SubjectData
from src.protocols.base import ProtocolBase

# Canonical EEGMMI imagery sessions: each tuple is the (L/R-run, hands/feet-run)
# pair recorded in one session.
EEGMMI_SESSIONS: tuple[tuple[int, ...], ...] = (
    (4, 6),
    (8, 10),
    (12, 14),
)


@register("protocol", "leave_one_session_out")
class LeaveOneSessionOutProtocol(ProtocolBase):
    name = "lso"

    def __init__(
        self,
        sessions: tuple[tuple[int, ...], ...] = EEGMMI_SESSIONS,
        seed: int = 0,
    ):
        self.sessions = tuple(tuple(s) for s in sessions)
        self.seed = seed

    def iter_splits(self, data: dict[int, SubjectData]) -> Iterator[Split]:
        for subject_id in sorted(data.keys()):
            sd = data[subject_id]
            for fold_i, held_out_runs in enumerate(self.sessions):
                held_set = set(held_out_runs)
                train_trials = [t for t in sd.trials if t.run not in held_set]
                eval_trials = [t for t in sd.trials if t.run in held_set]
                if not train_trials or not eval_trials:
                    continue
                yield Split(
                    train=train_trials,
                    calib=[],
                    eval=eval_trials,
                    meta={
                        "subject": subject_id,
                        "fold": fold_i,
                        "n_folds": len(self.sessions),
                        "held_out_runs": list(held_out_runs),
                        "n_train_trials": len(train_trials),
                        "n_eval_trials": len(eval_trials),
                    },
                )
