"""Tests for :mod:`impectPy.id_linking`."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("requests")

from impectPy.id_linking import _provider_columns


@pytest.mark.parametrize(
    "provider,expected",
    [
        (
            "heimSpiel",
            {
                "match": "heimSpielId",
                "home": "homeSquadHeimSpielId",
                "away": "awaySquadHeimSpielId",
            },
        ),
        (
            "HeimSpiel",
            {
                "match": "heimSpielId",
                "home": "homeSquadHeimSpielId",
                "away": "awaySquadHeimSpielId",
            },
        ),
        (
            "HEIM_SPIEL",
            {
                "match": "heimSpielId",
                "home": "homeSquadHeimSpielId",
                "away": "awaySquadHeimSpielId",
            },
        ),
        (
            "wyscout",
            {
                "match": "wyscoutId",
                "home": "homeSquadWyscoutId",
                "away": "awaySquadWyscoutId",
            },
        ),
        (
            "SkillCorner",
            {
                "match": "skillCornerId",
                "home": "homeSquadSkillCornerId",
                "away": "awaySquadSkillCornerId",
            },
        ),
    ],
)
def test_provider_columns_supports_common_variations(provider, expected):
    columns = _provider_columns(provider)

    assert columns.match == expected["match"]
    assert columns.home_team == expected["home"]
    assert columns.away_team == expected["away"]


@pytest.mark.parametrize("value", ["", "   ", "__", "-", "."])
def test_provider_columns_rejects_missing_names(value):
    with pytest.raises(ValueError):
        _provider_columns(value)
