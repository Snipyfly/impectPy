"""Utilities to link Impect matches and squads to DFL identifiers.

The Impect Customer API already embeds provider specific identifiers in
its responses (e.g. HeimSpiel/DFL, Wyscout, SkillCorner).  This module
collects these mappings in consolidated lookup tables that can be reused
whenever downstream workflows need to translate an Impect identifier into
the corresponding DFL identifier.

Typical usage
-------------

>>> import impectPy as ip  # doctest: +SKIP
>>> token = ip.getAccessToken("username@example.com", "password")  # doctest: +SKIP
>>> iterations = [518, 522]  # doctest: +SKIP
>>> team_lookup = ip.getDflTeamLookup(iterations, token)  # doctest: +SKIP
>>> match_lookup = ip.getDflMatchLookup(iterations, token)  # doctest: +SKIP

Both helper functions return pandas ``DataFrame`` objects that contain
Impect and DFL identifiers side-by-side.  These lookup tables can then be
used to translate Impect ids into DFL ids whenever a downstream workflow
needs to call either API.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable, Optional

import pandas as pd
import requests

if __package__ in (None, ""):
    from config import Config
    from helpers import RateLimitedAPI
    from matches import getMatchesFromHost
else:
    from .config import Config
    from .helpers import RateLimitedAPI
    from .matches import getMatchesFromHost


@dataclass
class _ProviderColumns:
    """Collect column names used for a provider mapping."""

    match: str
    home_team: str
    away_team: str


def _provider_columns(provider: str) -> _ProviderColumns:
    """Return the column names used by the API for a given provider."""

    if not provider or not provider.strip():
        raise ValueError("Provider name must not be empty.")

    provider_lower_camel, provider_upper_camel = _normalise_provider_name(provider)
    return _ProviderColumns(
        match=f"{provider_lower_camel}Id",
        home_team=f"homeSquad{provider_upper_camel}Id",
        away_team=f"awaySquad{provider_upper_camel}Id",
    )


def _collect_matches(
    iterations: Iterable[int],
    connection: RateLimitedAPI,
    host: str,
) -> pd.DataFrame:
    """Fetch matches for the supplied iterations and return a dataframe."""

    frames = []
    for iteration in iterations:
        matches = getMatchesFromHost(iteration=iteration, connection=connection, host=host)
        matches = matches.copy()
        matches["iterationId"] = iteration
        frames.append(matches)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _normalise_text(value: Optional[str]) -> Optional[str]:
    """Create a comparable, accent-free representation of a string."""

    if value is None:
        return None
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized.lower().strip()


def getDflMatchLookup(
    iterations: Iterable[int],
    token: str,
    session: Optional[requests.Session] = None,
    host: Optional[str] = None,
    provider: str = "heimSpiel",
) -> pd.DataFrame:
    """Return a dataframe mapping Impect match ids to DFL/Heimspiel ids."""

    host = host or Config().HOST
    connection = RateLimitedAPI(session=session or requests.Session())
    connection.session.headers.update({"Authorization": f"Bearer {token}"})

    provider_columns = _provider_columns(provider)
    matches = _collect_matches(iterations=iterations, connection=connection, host=host)

    if matches.empty:
        return pd.DataFrame(
            columns=[
                "iterationId",
                "impectMatchId",
                "dflMatchId",
                "scheduledDate",
                "homeSquadId",
                "homeSquadName",
                "homeDflTeamId",
                "awaySquadId",
                "awaySquadName",
                "awayDflTeamId",
            ]
        )

    for column in [provider_columns.match, provider_columns.home_team, provider_columns.away_team]:
        if column not in matches.columns:
            raise KeyError(
                f"Column '{column}' is not available in the matches response. "
                "Verify that the provider name is correct and that the API "
                "response includes the desired id mapping."
            )

    lookup = matches[
        [
            "iterationId",
            "id",
            provider_columns.match,
            "scheduledDate",
            "homeSquadId",
            "homeSquadName",
            provider_columns.home_team,
            "awaySquadId",
            "awaySquadName",
            provider_columns.away_team,
        ]
    ].rename(
        columns={
            "id": "impectMatchId",
            provider_columns.match: "dflMatchId",
            provider_columns.home_team: "homeDflTeamId",
            provider_columns.away_team: "awayDflTeamId",
        }
    )

    lookup["homeSquadNameNormalized"] = lookup["homeSquadName"].map(_normalise_text)
    lookup["awaySquadNameNormalized"] = lookup["awaySquadName"].map(_normalise_text)

    return lookup.drop_duplicates(subset=["impectMatchId"]).reset_index(drop=True)


def getDflTeamLookup(
    iterations: Iterable[int],
    token: str,
    session: Optional[requests.Session] = None,
    host: Optional[str] = None,
    provider: str = "heimSpiel",
) -> pd.DataFrame:
    """Return a dataframe mapping Impect squad ids to DFL/Heimspiel ids."""

    matches_lookup = getDflMatchLookup(
        iterations=iterations,
        token=token,
        session=session,
        host=host,
        provider=provider,
    )

    if matches_lookup.empty:
        return pd.DataFrame(
            columns=[
                "iterationId",
                "impectSquadId",
                "impectSquadName",
                "dflTeamId",
                "squadRole",
            ]
        )

    home = matches_lookup[
        [
            "iterationId",
            "homeSquadId",
            "homeSquadName",
            "homeDflTeamId",
        ]
    ].rename(
        columns={
            "homeSquadId": "impectSquadId",
            "homeSquadName": "impectSquadName",
            "homeDflTeamId": "dflTeamId",
        }
    )
    home["squadRole"] = "HOME"

    away = matches_lookup[
        [
            "iterationId",
            "awaySquadId",
            "awaySquadName",
            "awayDflTeamId",
        ]
    ].rename(
        columns={
            "awaySquadId": "impectSquadId",
            "awaySquadName": "impectSquadName",
            "awayDflTeamId": "dflTeamId",
        }
    )
    away["squadRole"] = "AWAY"

    squads = pd.concat([home, away], ignore_index=True)
    squads = squads.dropna(subset=["impectSquadId"]).drop_duplicates(
        subset=["impectSquadId", "dflTeamId", "iterationId"]
    )

    squads["impectSquadNameNormalized"] = squads["impectSquadName"].map(_normalise_text)

    return squads.reset_index(drop=True)


_CANONICAL_PROVIDER_NAMES = {
    "heimspiel": ("heimSpiel", "HeimSpiel"),
}


def _normalise_provider_name(provider: str) -> tuple[str, str]:
    """Return lower and upper camel-case representations of a provider name."""

    stripped = provider.strip()
    if not stripped:
        raise ValueError("Provider name must not be empty.")

    sanitized = re.sub(r"[^0-9A-Za-z]+", "", stripped)
    if not sanitized:
        raise ValueError("Provider name must contain alphanumeric characters.")

    canonical = _CANONICAL_PROVIDER_NAMES.get(sanitized.lower())
    if canonical:
        return canonical

    parts = re.findall(r"[A-Z]?[a-z]+|[0-9]+|[A-Z]+(?=[A-Z]|$)", stripped)
    if not parts:
        parts = [sanitized]

    parts = [part.lower() for part in parts if part]
    if not parts:
        raise ValueError("Provider name must contain alphanumeric characters.")

    lower_camel = parts[0] + "".join(part.capitalize() for part in parts[1:])
    upper_camel = "".join(part.capitalize() for part in parts)
    return lower_camel, upper_camel


__all__ = ["getDflMatchLookup", "getDflTeamLookup"]
