from collections import Counter

from src.build_balanced_zip_dataset import Record, added_group_key, evenly_spaced, strict_official_group


def make_record(name: str) -> Record:
    return Record(
        name=name,
        label_name=name.rsplit(".", 1)[0] + ".txt",
        source="added",
        split="train",
        scene_id="sequence:RURU02",
        cluster_id="added:sequence:RURU02",
        image_entry=name,
        label_entry=name.rsplit(".", 1)[0] + ".txt",
        archive="combined",
        classes=Counter({24: 1}),
    )


def test_added_group_key_recovers_numeric_and_sequence_sources() -> None:
    assert added_group_key("1_3_105_12041.jpg") == "numeric:1_3_105"
    assert added_group_key("AUAU010001.jpg") == "sequence:AUAU01"
    assert added_group_key("RURU020571.jpg") == "sequence:RURU02"


def test_strict_official_group_joins_products_and_map_sources() -> None:
    assert strict_official_group("01-PAN-20240420-113-325-L00000010882-CCD3_5_crop4.jpg") == (
        "satellite:01-PAN-20240420-113-325-L00000010882"
    )
    assert strict_official_group("E103.9_N1.2_20200419_L1A0004749016-PAN20_crop1.jpg") == (
        "l1a:E103.9_N1.2_20200419_L1A0004749016"
    )
    assert strict_official_group("fsc_AGZ-N24.15-E120.73-lv20-Google_crop0001.jpg") == (
        "fsc:N24.15:E120.73"
    )


def test_evenly_spaced_is_deterministic_and_keeps_endpoints() -> None:
    records = [make_record(f"RURU02{index:04d}.jpg") for index in range(100)]
    selected = evenly_spaced(records, 15)
    assert len(selected) == 15
    assert selected[0].name == "RURU020000.jpg"
    assert selected[-1].name == "RURU020099.jpg"
    assert len({record.name for record in selected}) == 15


def test_evenly_spaced_does_not_drop_small_groups() -> None:
    records = [make_record(f"AUES01{index:04d}.jpg") for index in range(4)]
    assert evenly_spaced(records, 15) == records
