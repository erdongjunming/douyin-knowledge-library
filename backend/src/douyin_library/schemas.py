from enum import StrEnum


class JobStage(StrEnum):
    DISCOVER = "discover"
    DOWNLOAD = "download"
    EXTRACT = "extract"
    TRANSCRIBE = "transcribe"
    ANALYZE = "analyze"
    COMPLETE = "complete"


class WorkAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNFAVORITED = "unfavorited"
    UNKNOWN = "unknown"
