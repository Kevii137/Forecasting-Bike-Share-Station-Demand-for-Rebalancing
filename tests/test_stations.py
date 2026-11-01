"""Tests for station label normalisation and merging."""
import pandas as pd
import pytest

from bikeshare.data.stations import group_names, normalise_name


@pytest.mark.parametrize(("raw", "expected"), [
    ("[Archived] Wilton Ave & Diversey Pkwy", "Wilton Ave & Diversey Pkwy"),
    ("Larrabee St & Kingsbury St 1", "Larrabee St & Kingsbury St"),
    ("Clark St & Elm St*", "Clark St & Elm St"),
    ("North Ave Public Rack - 2", "North Ave Public Rack"),
    ("Clark St & 47th St", "Clark St & 47th St"),
])
def test_normalise_name(raw, expected):
    assert normalise_name(raw) == expected


def pairs(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["norm_name", "station_id", "trips", "lat", "lng"])


def test_nearby_names_sharing_an_id_are_renames():
    groups = group_names(pairs([
        ("Rainbow Beach", "CHI1", 200, 41.75, -87.55),
        ("Rainbow - Beach", "CHI1", 5, 41.7502, -87.55),
    ]), merge_distance_metres=100)
    assert groups["Rainbow - Beach"] == "Rainbow Beach"


def test_distant_names_sharing_an_id_stay_separate():
    groups = group_names(pairs([
        ("Ashland Ave & Division St", "CHI2", 100, 41.903, -87.667),
        ("Paulina St & Division St", "CHI2", 50, 41.903, -87.670),
    ]), merge_distance_metres=100)
    assert groups["Paulina St & Division St"] == "Paulina St & Division St"


def test_unrelated_stations_stay_separate():
    groups = group_names(pairs([
        ("A", "CHI3", 10, 41.8, -87.6),
        ("B", "CHI4", 10, 41.8, -87.6),
    ]), merge_distance_metres=100)
    assert groups["A"] != groups["B"]
