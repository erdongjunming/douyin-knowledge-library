from douyin_library.schemas import JobStage, WorkAvailability


def test_persisted_state_values_are_stable() -> None:
    assert [stage.value for stage in JobStage] == [
        "discover",
        "download",
        "extract",
        "transcribe",
        "analyze",
        "complete",
    ]
    assert [state.value for state in WorkAvailability] == [
        "available",
        "unavailable",
        "unfavorited",
        "unknown",
    ]
