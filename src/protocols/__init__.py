"""Evaluation protocols: yield Splits, runner iterates."""
from src.protocols.base import ProtocolBase
from src.protocols.loso import LOSOProtocol
from src.protocols.within_subject_cv import WithinSubjectCVProtocol
from src.protocols.kmin_calibration import KTrialsCalibrationProtocol
from src.protocols.leave_one_session_out import LeaveOneSessionOutProtocol

__all__ = [
    "ProtocolBase",
    "LOSOProtocol",
    "WithinSubjectCVProtocol",
    "KTrialsCalibrationProtocol",
    "LeaveOneSessionOutProtocol",
]
