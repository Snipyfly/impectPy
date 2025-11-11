"""Example script for generating an interactive match overview dashboard.

The script mirrors the workflow from the README and demonstrates how to
combine the high-level helper functions provided by :mod:`impectPy` to load a
match, compute a handful of metrics and visualise them with Plotly.

Steps
-----
1. Authenticate via :func:`impectPy.getAccessToken`.
2. Fetch the list of available iterations with :func:`impectPy.getIterations`.
3. Ask the user which iteration and match should be analysed.
4. Load the event data, match sums and iteration averages.
5. Aggregate the raw data into team, phase, time-line and player level KPIs.
6. Plot the result as an interactive Plotly figure.

Running the example
-------------------
Set the ``USERNAME`` and ``PASSWORD`` constants below or export the matching
environment variables before executing the script::

    export IMPLECT_USERNAME="you@example.com"
    export IMPLECT_PASSWORD="secret"
    python examples/match_overview.py

When run, the script provides a guided command line interface that allows you
to pick a season/iteration and match from the available data.  The final step
is an interactive Plotly window summarising team KPIs, xG by phase, an xG
timeline and individual player ratings.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

import impectPy as ip

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USERNAME = os.getenv("IMPLECT_USERNAME", "")
PASSWORD = os.getenv("IMPLECT_PASSWORD", "")
DEFAULT_SEASON = "25/26"
DEFAULT_PROFILE_POSITIONS = [
    "GOALKEEPER",
    "LEFT_WINGBACK_DEFENDER",
    "RIGHT_WINGBACK_DEFENDER",
    "CENTRAL_DEFENDER",
    "DEFENSE_MIDFIELD",
    "CENTRAL_MIDFIELD",
    "ATTACKING_MIDFIELD",
    "LEFT_WINGER",
    "RIGHT_WINGER",
    "CENTER_FORWARD",
]


# ---------------------------------------------------------------------------
# Helper data-classes & functions
# ---------------------------------------------------------------------------


def _prompt(prompt: str, default: str) -> str:
    """Return the users input or a default when the reply is empty."""

    reply = input(prompt).strip()
    return reply or default


def get_access_token(username: str, password: str) -> str:
    """Authenticate the user and return an API token."""

    print("🔐 Hole Access-Token...")
    return ip.getAccessToken(username=username, password=password)


def choose_iteration(token: str) -> int:
    """Interactively prompt the user for an iteration ID."""

    print("📚 Hole Iterations-Übersicht...")
    iterations = pd.DataFrame(ip.getIterations(token))
    iterations["season"] = iterations["season"].astype(str)

    seasons = sorted(iterations["season"].unique())
    print("\nVerfügbare Saisons:", seasons)

    season = _prompt(
        f"Saison wählen (z.B. {DEFAULT_SEASON}) [Enter = {DEFAULT_SEASON}]: ",
        DEFAULT_SEASON,
    )

    iteration_table = iterations[iterations["season"] == season]
    if iteration_table.empty:
        print(f"⚠️ Keine Iterationen für Saison {season} gefunden, nutze alle Iterationen.")
        iteration_table = iterations

    print("\n📋 Verfügbare Wettbewerbe / Iterationen:")
    iteration_table = iteration_table.sort_values(["competitionName", "id"])
    for idx, row in enumerate(iteration_table.itertuples(), start=1):
        print(f"{idx:2d}) id={row.id} | {row.competitionName} | Saison {row.season}")

    default_index = 1
    selected_index = _prompt(
        f"\nNummer der gewünschten Iteration (z.B. {default_index}) "
        f"[Enter = {default_index}]: ",
        str(default_index),
    )

    try:
        selected_index = int(selected_index)
    except ValueError:
        selected_index = default_index

    selected_index = max(1, min(selected_index, len(iteration_table)))
    selection = iteration_table.iloc[selected_index - 1]

    print(
        "\n✅ Gewählte Iteration: id={id} | {competition} | Saison {season}\n".format(
            id=int(selection["id"]),
            competition=selection["competitionName"],
            season=selection["season"],
        )
    )
    return int(selection["id"])


def get_matchplan(token: str, iteration_id: int) -> pd.DataFrame:
    """Fetch the match plan for the provided iteration."""

    print(f"📅 Hole Matchplan für Iteration (Saison) {iteration_id}...")
    matchplan = pd.DataFrame(ip.getMatches(iteration_id, token))
    print("   Matchplan-Spalten:", list(matchplan.columns))
    return matchplan


def choose_match(matchplan: pd.DataFrame) -> int:
    """Prompt the user to choose a match ID from the match plan."""

    matchplan = matchplan.copy()
    sort_columns = [
        column
        for column in ["matchDayIndex", "scheduledDate", "dateTime", "id"]
        if column in matchplan.columns
    ]
    if sort_columns:
        matchplan = matchplan.sort_values(sort_columns)

    print("\n📋 Verfügbare Spiele:")
    for row in matchplan.itertuples():
        matchday = getattr(row, "matchDayIndex", "?")
        home = getattr(row, "homeSquadName", "?")
        away = getattr(row, "awaySquadName", "?")
        date = getattr(row, "scheduledDate", getattr(row, "dateTime", "?"))
        match_id = getattr(row, "id")
        print(f"- MD {matchday}: {home} vs {away} (matchId={match_id}, Datum={date})")

    default_match_id = int(matchplan.iloc[0]["id"])
    selection = _prompt(
        f"\nGewünschte matchId eingeben [Enter = {default_match_id}]: ",
        str(default_match_id),
    )

    try:
        match_id = int(selection)
    except ValueError:
        match_id = default_match_id

    print(f"\n✅ Gewähltes Spiel: matchId={match_id}\n")
    return match_id


def derive_minute_str(game_time: str) -> float:
    """Convert the API ``gameTime`` string into minutes as float."""

    if not isinstance(game_time, str):
        try:
            return float(game_time) / 60.0
        except Exception:  # pragma: no cover - defensive casting
            return np.nan

    if ":" in game_time:
        minutes, seconds = game_time.split(":", 1)
        try:
            return float(minutes) + float(seconds) / 60.0
        except Exception:  # pragma: no cover - defensive casting
            return np.nan

    try:
        return float(game_time) / 60.0
    except Exception:  # pragma: no cover - defensive casting
        return np.nan


def add_minutes_column(events: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``events`` that contains an additional ``minute`` column."""

    events = events.copy()
    if "gameTime" in events.columns:
        events["minute"] = events["gameTime"].astype(str).apply(derive_minute_str)
    else:
        events["minute"] = np.nan
    return events


TEAM_KPI_LABELS: Sequence[Tuple[str, str]] = (
    ("GOALS", "Goals"),
    ("SHOT_XG", "Shot-based xG"),
    ("PACKING_XG", "Packing xG"),
    ("POSTSHOT_XG", "Post-shot xG"),
    ("SHOT_AT_GOAL_NUMBER", "Total Shots"),
    ("CRITICAL_BALL_LOSS_NUMBER", "Critical Ball Losses"),
    ("WON_GROUND_DUELS", "Won Ground Duels"),
    ("LOST_GROUND_DUELS", "Lost Ground Duels"),
    ("WON_AERIAL_DUELS", "Won Aerial Duels"),
    ("LOST_AERIAL_DUELS", "Lost Aerial Duels"),
    ("EXPECTED_PASSES", "Expected Passes"),
    ("BYPASSED_OPPONENTS_RAW", "Bypassed Opponents (raw)"),
    ("BYPASSED_DEFENDERS_RAW", "Bypassed Defenders (raw)"),
    ("BYPASSED_OPPONENTS_TO_PITCH_POSITION_FINAL_THIRD", "Bypassed Opponents – Final Third"),
    ("BYPASSED_OPPONENTS_NUMBER_TO_PITCH_POSITION_FINAL_THIRD", "Bypassed Opponent Actions – Final Third"),
    ("BYPASSED_OPPONENTS_NUMBER_TO_PITCH_POSITION_OPPONENT_BOX", "Bypassed Opponent Actions – Opponent Box"),
    ("BYPASSED_DEFENDERS_BY_ACTION_LOW_PASS", "Bypassed Defenders – Low Pass"),
    ("BYPASSED_DEFENDERS_BY_ACTION_DIAGONAL_PASS", "Bypassed Defenders – Diagonal Pass"),
    ("BYPASSED_DEFENDERS_BY_ACTION_CHIPPED_PASS", "Bypassed Defenders – Chipped Pass"),
    ("BYPASSED_DEFENDERS_BY_ACTION_SHORT_AERIAL_PASS", "Bypassed Defenders – Short Aerial Pass"),
    ("BYPASSED_DEFENDERS_BY_ACTION_LOW_CROSS", "Bypassed Defenders – Low Cross"),
    ("BYPASSED_DEFENDERS_BY_ACTION_HIGH_CROSS", "Bypassed Defenders – High Cross"),
    ("BYPASSED_DEFENDERS_BY_ACTION_CLEARANCE", "Bypassed Defenders – Clearance"),
    ("BYPASSED_DEFENDERS_BY_ACTION_HEADER", "Bypassed Defenders – Header"),
    ("BYPASSED_DEFENDERS_BY_ACTION_BLOCK", "Bypassed Defenders – Block"),
    ("BYPASSED_DEFENDERS_BY_ACTION_SAVE", "Bypassed Defenders – Save"),
    ("BYPASSED_DEFENDERS_BY_ACTION_GOAL_KICK", "Bypassed Defenders – Goal Kick"),
    ("BYPASSED_DEFENDERS_BY_ACTION_THROW_IN", "Bypassed Defenders – Throw-in"),
    ("BYPASSED_DEFENDERS_BY_ACTION_CORNER", "Bypassed Defenders – Corner"),
    ("BYPASSED_DEFENDERS_BY_ACTION_FREE_KICK", "Bypassed Defenders – Free Kick"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_LOW_PASS", "Bypassed Opponent Actions – Low Pass"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_DIAGONAL_PASS", "Bypassed Opponent Actions – Diagonal Pass"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_CHIPPED_PASS", "Bypassed Opponent Actions – Chipped Pass"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_SHORT_AERIAL_PASS", "Bypassed Opponent Actions – Short Aerial Pass"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_LOW_CROSS", "Bypassed Opponent Actions – Low Cross"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_HIGH_CROSS", "Bypassed Opponent Actions – High Cross"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_CLEARANCE", "Bypassed Opponent Actions – Clearance"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_HEADER", "Bypassed Opponent Actions – Header"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_BLOCK", "Bypassed Opponent Actions – Block"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_SAVE", "Bypassed Opponent Actions – Save"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_GOAL_KICK", "Bypassed Opponent Actions – Goal Kick"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_THROW_IN", "Bypassed Opponent Actions – Throw-in"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_CORNER", "Bypassed Opponent Actions – Corner"),
    ("BYPASSED_OPPONENTS_NUMBER_BY_ACTION_FREE_KICK", "Bypassed Opponent Actions – Free Kick"),
    ("BYPASSED_OPPONENTS_NUMBER_TO_PITCH_POSITION_FIRST_THIRD", "Bypassed Opponent Actions – First Third"),
    ("BYPASSED_OPPONENTS_NUMBER_TO_PITCH_POSITION_MIDDLE_THIRD", "Bypassed Opponent Actions – Middle Third"),
    ("BYPASSED_OPPONENTS_NUMBER_TO_PITCH_POSITION_FINAL_THIRD", "Bypassed Opponent Actions – Final Third"),
    ("BYPASSED_OPPONENTS_NUMBER_TO_PITCH_POSITION_OPPONENT_BOX", "Bypassed Opponent Actions – Opponent Box"),
    ("SUCCESSFUL_PASSES_TO_PITCH_POSITION_FIRST_THIRD", "Successful Passes – First Third"),
    ("SUCCESSFUL_PASSES_TO_PITCH_POSITION_MIDDLE_THIRD", "Successful Passes – Middle Third"),
    ("SUCCESSFUL_PASSES_TO_PITCH_POSITION_FINAL_THIRD", "Successful Passes – Final Third"),
    ("SUCCESSFUL_PASSES_TO_PITCH_POSITION_OPPONENT_BOX", "Successful Passes – Opponent Box"),
    ("OFFENSIVE_TOUCHES_IN_PITCH_POSITION_OWN_BOX", "Offensive Touches – Own Box"),
    ("OFFENSIVE_TOUCHES_IN_PITCH_POSITION_FIRST_THIRD", "Offensive Touches – First Third"),
    ("OFFENSIVE_TOUCHES_IN_PITCH_POSITION_MIDDLE_THIRD", "Offensive Touches – Middle Third"),
    ("OFFENSIVE_TOUCHES_IN_PITCH_POSITION_FINAL_THIRD", "Offensive Touches – Final Third"),
    ("OFFENSIVE_TOUCHES_IN_PITCH_POSITION_OPPONENT_BOX", "Offensive Touches – Opponent Box"),
    ("REVERSE_PLAY_NUMBER_FROM_PITCH_POSITION_FIRST_THIRD", "Reverse Plays – First Third"),
    ("REVERSE_PLAY_NUMBER_FROM_PITCH_POSITION_MIDDLE_THIRD", "Reverse Plays – Middle Third"),
    ("REVERSE_PLAY_NUMBER_FROM_PITCH_POSITION_FINAL_THIRD", "Reverse Plays – Final Third"),
    ("REVERSE_PLAY_NUMBER_FROM_PITCH_POSITION_OPPONENT_BOX", "Reverse Plays – Opponent Box"),
    ("REVERSE_PLAY_NUMBER_AT_PHASE_IN_POSSESSION", "Reverse Plays – In Possession"),
    ("REVERSE_PLAY_NUMBER_AT_PHASE_ATTACKING_TRANSITION", "Reverse Plays – Attacking Transition"),
    ("REVERSE_PLAY_NUMBER_AT_PHASE_SET_PIECE", "Reverse Plays – Set Piece"),
    ("REVERSE_PLAY_NUMBER_AT_PHASE_SECOND_BALL", "Reverse Plays – Second Ball"),
    ("WON_GROUND_DUELS_IN_PITCH_POSITION_OWN_BOX", "Won Ground Duels – Own Box"),
    ("WON_GROUND_DUELS_IN_PITCH_POSITION_FIRST_THIRD", "Won Ground Duels – First Third"),
    ("WON_GROUND_DUELS_IN_PITCH_POSITION_MIDDLE_THIRD", "Won Ground Duels – Middle Third"),
    ("WON_GROUND_DUELS_IN_PITCH_POSITION_FINAL_THIRD", "Won Ground Duels – Final Third"),
    ("WON_GROUND_DUELS_IN_PITCH_POSITION_OPPONENT_BOX", "Won Ground Duels – Opponent Box"),
    ("PXT_PASS", "pxT – Pass"),
    ("PXT_DRIBBLE", "pxT – Dribble"),
    ("PXT_SETPIECE", "pxT – Set Piece"),
    ("PXT_BLOCK", "pxT – Block"),
    ("PXT_SHOT", "pxT – Shot"),
    ("PXT_BALL_WIN", "pxT – Ball Win"),
    ("PXT_FOUL", "pxT – Foul"),
    ("PXT_NO_VIDEO", "pxT – No Video"),
    ("PXT_REC", "pxT – Receiving"),
    ("NUMBER_OF_PRESSES", "Number of Presses"),
    ("NUMBER_OF_PRESSES_BUILD_UP", "Presses – Build-up"),
    ("NUMBER_OF_PRESSES_BETWEEN_THE_LINES", "Presses – Between the Lines"),
    ("NUMBER_OF_PRESSES_COUNTER_PRESS", "Presses – Counter Press"),
)


def most_common_value(values: pd.Series, default: str = "Unknown") -> str:
    """Return the mode of ``values`` with graceful fallbacks."""

    if values.empty:
        return default

    filtered = values.dropna()
    if filtered.empty:
        return default

    modes = filtered.mode()
    if not modes.empty:
        return str(modes.iloc[0])

    return str(filtered.iloc[0])


def select_top_numeric_columns(
    dataframe: pd.DataFrame,
    exclude: Sequence[str],
    limit: int,
) -> List[str]:
    """Return the most relevant numeric columns for display."""

    numeric_columns = [
        column
        for column in dataframe.columns
        if column not in exclude and is_numeric_dtype(dataframe[column])
    ]
    numeric_columns = sorted(
        numeric_columns,
        key=lambda column: dataframe[column].sum(skipna=True),
        reverse=True,
    )
    return numeric_columns[:limit]


def prepare_table(
    dataframe: Optional[pd.DataFrame],
    base_columns: Sequence[str],
    extra_candidates: Optional[Sequence[str]] = None,
    numeric_limit: int = 3,
    top_n: int = 10,
    sort_by: Optional[str] = None,
) -> pd.DataFrame:
    """Return a compact table with the selected columns formatted for display."""

    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    df = dataframe.copy()
    selected: List[str] = [column for column in base_columns if column in df.columns]

    if extra_candidates:
        for column in extra_candidates:
            if column in df.columns and column not in selected:
                selected.append(column)

    exclude = set(selected)
    numeric_columns = select_top_numeric_columns(df, exclude=selected, limit=numeric_limit)
    selected.extend([column for column in numeric_columns if column not in selected])

    if not selected:
        return pd.DataFrame()

    for column in selected:
        if column in df.columns and is_datetime64_any_dtype(df[column]):
            df[column] = pd.to_datetime(df[column]).dt.strftime("%Y-%m-%d")

    table = df[selected]

    sort_candidates = [column for column in [sort_by, *numeric_columns, *selected] if column in table.columns]
    sort_column = sort_candidates[0]
    table = table.sort_values(sort_column, ascending=False, na_position="last").head(top_n)
    table = table.reset_index(drop=True)

    numeric_subset = [column for column in table.columns if is_numeric_dtype(table[column])]
    if numeric_subset:
        table[numeric_subset] = table[numeric_subset].applymap(
            lambda value: round(float(value), 2) if pd.notna(value) else np.nan
        )

    return table


def summarise_player_scores(player_scores: Optional[pd.DataFrame], match_id: int) -> pd.DataFrame:
    """Return a compact table containing the most relevant player scores for the match."""

    if player_scores is None or player_scores.empty:
        return pd.DataFrame()

    filtered = player_scores[player_scores.get("matchId") == match_id]
    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.copy()
    return prepare_table(
        filtered,
        base_columns=["playerName", "squadName"],
        extra_candidates=["position", "positions", "matchShare", "playDuration"],
        numeric_limit=4,
    )


def summarise_player_profile_scores(
    profile_scores: Optional[pd.DataFrame],
    squad_ids: Sequence[int],
) -> pd.DataFrame:
    """Return player profile scores for the participating squads."""

    if profile_scores is None or profile_scores.empty or not squad_ids:
        return pd.DataFrame()

    filtered = profile_scores[profile_scores.get("squadId").isin(squad_ids)]
    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.copy()
    return prepare_table(
        filtered,
        base_columns=["playerName", "squadName"],
        extra_candidates=["positions", "matchShare", "playDuration"],
        numeric_limit=4,
    )


def summarise_squad_scores(squad_scores: Optional[pd.DataFrame], match_id: int) -> pd.DataFrame:
    """Return aggregated squad scores for the selected match."""

    if squad_scores is None or squad_scores.empty:
        return pd.DataFrame()

    filtered = squad_scores[squad_scores.get("matchId") == match_id]
    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.copy()
    return prepare_table(
        filtered,
        base_columns=["squadName"],
        extra_candidates=["matchShare", "playDuration"],
        numeric_limit=4,
        top_n=4,
    )


def parse_match_date(match_meta: Dict[str, object]) -> Optional[pd.Timestamp]:
    """Return the scheduled match date if available."""

    for key in ("scheduledDate", "dateTime"):
        if key in match_meta and match_meta[key]:
            try:
                return pd.to_datetime(match_meta[key])
            except Exception:  # pragma: no cover - defensive conversion
                continue
    return None


def summarise_squad_ratings(
    squad_ratings: Optional[pd.DataFrame],
    squad_ids: Sequence[int],
    match_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """Return the latest squad ratings for the teams involved in the match."""

    if squad_ratings is None or squad_ratings.empty or not squad_ids:
        return pd.DataFrame()

    filtered = squad_ratings[squad_ratings.get("squadId").isin(squad_ids)]
    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.copy()
    if "date" in filtered.columns and match_date is not None:
        filtered["date_ts"] = pd.to_datetime(filtered["date"], errors="coerce", utc=True)
        filtered = filtered[filtered["date_ts"].notna()]

        filtered["date_ts"] = filtered["date_ts"].dt.tz_localize(None)
        match_cutoff = (
            match_date.tz_localize(None)
            if getattr(match_date, "tzinfo", None) is not None
            else match_date
        )

        filtered = filtered[filtered["date_ts"] <= match_cutoff]
        filtered = filtered.sort_values("date_ts", ascending=False).drop_duplicates("squadId")

    return prepare_table(
        filtered,
        base_columns=["squadName", "date" if "date" in filtered.columns else "iterationId"],
        extra_candidates=["value"],
        numeric_limit=1,
        top_n=4,
    )


def summarise_squad_coefficients(
    squad_coefficients: Optional[pd.DataFrame],
    squad_ids: Sequence[int],
    match_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """Return the predictive coefficients for the squads involved."""

    if squad_coefficients is None or squad_coefficients.empty or not squad_ids:
        return pd.DataFrame()

    filtered = squad_coefficients[squad_coefficients.get("squadId").isin(squad_ids)]
    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.copy()
    if "date" in filtered.columns and match_date is not None:
        filtered["date_ts"] = pd.to_datetime(filtered["date"], errors="coerce", utc=True)
        filtered = filtered[filtered["date_ts"].notna()]

        filtered["date_ts"] = filtered["date_ts"].dt.tz_localize(None)
        match_cutoff = (
            match_date.tz_localize(None)
            if getattr(match_date, "tzinfo", None) is not None
            else match_date
        )

        filtered = filtered[filtered["date_ts"] <= match_cutoff]
        filtered = filtered.sort_values("date_ts", ascending=False).drop_duplicates("squadId")

    return prepare_table(
        filtered,
        base_columns=["squadName", "date" if "date" in filtered.columns else "iterationId"],
        extra_candidates=[
            "attackCoefficient",
            "defenseCoefficient",
            "homeCoefficient",
            "competitionCoefficient",
        ],
        numeric_limit=4,
        top_n=4,
    )


def summarise_set_pieces(set_pieces: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return a simple breakdown of the most common set piece categories."""

    if set_pieces is None or set_pieces.empty:
        return pd.DataFrame()

    if "attackingSquadName" not in set_pieces.columns:
        return pd.DataFrame()

    category_column = next(
        (
            column
            for column in [
                "setPieceType",
                "setPieceCategory",
                "setPieceName",
                "setPieceSubPhaseType",
            ]
            if column in set_pieces.columns
        ),
        None,
    )

    if category_column is None:
        return pd.DataFrame()

    dataframe = set_pieces.copy()
    dataframe["category"] = dataframe[category_column].astype(str)

    xg_columns = [column for column in dataframe.columns if "xg" in column.lower()]
    if xg_columns:
        dataframe["xg_total"] = dataframe[xg_columns].sum(axis=1)
    else:
        dataframe["xg_total"] = np.nan

    count_source = "setPieceId" if "setPieceId" in dataframe.columns else "category"
    summary = (
        dataframe.groupby(["attackingSquadName", "category"])
        .agg(
            Anzahl=(count_source, "nunique") if count_source == "setPieceId" else (count_source, "size"),
            xG=("xg_total", "sum"),
        )
        .reset_index()
    )

    summary = summary.rename(columns={"attackingSquadName": "Team", "category": "Kategorie"})
    summary["Kategorie"] = summary["Kategorie"].str.replace("_", " ").str.title()
    summary["Anzahl"] = summary["Anzahl"].astype(int)

    focus_mask = summary["Kategorie"].str.contains("Free|Freistoß|Corner|Ecke", case=False, na=False)
    if focus_mask.any():
        summary = summary[focus_mask]

    if "xG" in summary.columns:
        summary["xG"] = summary["xG"].round(2)

    return summary.sort_values(["Team", "Kategorie"]).reset_index(drop=True)


def safe_api_call(description: str, func, *args, **kwargs):
    """Execute ``func`` and capture exceptions with a concise console message."""

    try:
        print(f"📦 Lade {description}...")
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            print(f"✅ {description} geladen ({len(result)} Zeilen)")
        else:
            print(f"✅ {description} geladen")
        return result
    except Exception as error:  # pragma: no cover - defensive logging
        print(f"⚠️ {description} konnte nicht geladen werden: {error}")
        return None


def load_additional_match_data(
    match_id: int,
    iteration_id: int,
    token: str,
    events: pd.DataFrame,
) -> Dict[str, Optional[pd.DataFrame]]:
    """Fetch optional match level datasets used for the extended dashboard."""

    additional: Dict[str, Optional[pd.DataFrame]] = {}

    additional["player_scores"] = safe_api_call(
        "Player Match Scores",
        ip.getPlayerMatchScores,
        matches=[match_id],
        token=token,
    )

    unique_positions: List[str] = []
    position_series = events.get("position")
    if position_series is not None:
        unique_positions = sorted(
            {
                str(position).upper()
                for position in position_series.dropna().unique()
                if isinstance(position, str)
            }
        )
    positions = [position for position in unique_positions if position in DEFAULT_PROFILE_POSITIONS]
    if not positions:
        positions = DEFAULT_PROFILE_POSITIONS

    additional["player_profile_scores"] = safe_api_call(
        "Player Profile Scores",
        ip.getPlayerProfileScores,
        iteration=iteration_id,
        positions=positions,
        token=token,
    )

    additional["squad_scores"] = safe_api_call(
        "Squad Match Scores",
        ip.getSquadMatchScores,
        matches=[match_id],
        token=token,
    )

    additional["set_pieces"] = safe_api_call(
        "Set Pieces",
        ip.getSetPieces,
        matches=[match_id],
        token=token,
    )

    additional["squad_ratings"] = safe_api_call(
        "Squad Ratings",
        ip.getSquadRatings,
        iteration=iteration_id,
        token=token,
    )

    additional["squad_coefficients"] = safe_api_call(
        "Squad Coefficients",
        ip.getSquadCoefficients,
        iteration=iteration_id,
        token=token,
    )

    return additional


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def map_phase(phase: str) -> str:
    """Bucket the impect phase into broader xG categories."""

    if not isinstance(phase, str):
        return "Sonstige"

    phase_upper = phase.upper()
    if "SET_PIECE" in phase_upper:
        return "Standard"
    if "ATTACKING_TRANSITION" in phase_upper or "COUNTER" in phase_upper:
        return "Konter"
    if "SECOND_BALL" in phase_upper:
        return "Pressing / 2. Bälle"
    if "IN_POSSESSION" in phase_upper or "BUILD_UP" in phase_upper:
        return "Spielaufbau"
    return "Sonstige"


def get_team_colors(home_name: str, away_name: str) -> Dict[str, str]:
    """Return distinct colours for both squads used in the figure."""

    palette = {
        "SpVgg Greuther Fürth": ("#008C45", "#CCCCCC"),
        "Preußen Münster": ("#000000", "#888888"),
        "1. FC Kaiserslautern": ("#B00030", "#999999"),
        "SV Werder Bremen": ("#008A3A", "#CCCCCC"),
        "VfL Wolfsburg": ("#63B22F", "#999999"),
        "Hertha BSC": ("#004C9B", "#FFB300"),
        "FC Schalke 04": ("#1F4FA3", "#FFB300"),
    }

    def pick(name: str, default_main: str, default_alt: str) -> Iterable[str]:
        if name in palette:
            return palette[name]
        return default_main, default_alt

    home_color, home_alt = pick(home_name, "#1f77b4", "#aec7e8")
    away_color, away_alt = pick(away_name, "#d62728", "#ff9896")

    return {
        "home_main": home_color,
        "home_alt": home_alt,
        "away_main": away_color,
        "away_alt": away_alt,
    }


def infer_minutes_from_matchsums(matchsums: List[Dict[str, float]]) -> pd.DataFrame:
    """Infer the played minutes from the match sums payload."""

    df = pd.DataFrame(matchsums).copy()
    if df.empty:
        return pd.DataFrame({"playerId": [], "minutes": []})

    columns_lower = {column.lower(): column for column in df.columns}

    if "minutes" in columns_lower:
        source_column = columns_lower["minutes"]
        df["minutes"] = pd.to_numeric(df[source_column], errors="coerce")
    elif "minutesplayed" in columns_lower:
        source_column = columns_lower["minutesplayed"]
        df["minutes"] = pd.to_numeric(df[source_column], errors="coerce")
    else:
        minute_columns = [column for column in df.columns if "minute" in column.lower()]
        if minute_columns:
            source_column = minute_columns[0]
            df["minutes"] = pd.to_numeric(df[source_column], errors="coerce")
        else:
            share_columns = [column for column in df.columns if "matchshare" in column.lower()]
            if share_columns:
                source_column = share_columns[0]
                df["minutes"] = pd.to_numeric(df[source_column], errors="coerce") * 90.0
            else:
                df["minutes"] = 90.0

    df["minutes"] = df["minutes"].fillna(0.0)
    return df[["playerId", "minutes"]]


def group_position(position: str) -> str:
    """Map the detailed position into broader buckets."""

    if not isinstance(position, str):
        return "Other"

    position_upper = position.upper()
    if "KEEPER" in position_upper:
        return "Goalkeeper"
    if "DEFENDER" in position_upper or "BACK" in position_upper:
        return "Defender"
    if "MIDFIELD" in position_upper:
        return "Midfielder"
    if "FORWARD" in position_upper or "STRIKER" in position_upper or "WINGER" in position_upper:
        return "Forward"
    return "Other"


def normalize_by_pos(raw: pd.Series, positions: pd.Series) -> pd.Series:
    """Normalise scores to a 1..10 scale within each positional group."""

    frame = pd.DataFrame({"value": raw, "position": positions})
    scores = pd.Series(index=frame.index, dtype=float)

    for position, subset in frame.groupby("position"):
        values = subset["value"]
        if values.max() == values.min():
            scores.loc[subset.index] = 5.0 if values.max() == 0 else 10.0
        else:
            scores.loc[subset.index] = 1.0 + 9.0 * (values - values.min()) / (values.max() - values.min())

    return scores.clip(0, 10)


def compute_team_kpis(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a selection of team level KPIs for the figure."""

    events = events.copy()
    groups = events.groupby(["squadId", "squadName"])

    rows = []
    for (_squad_id, squad_name), subset in groups:
        row: Dict[str, float] = {"Team": squad_name}

        if "GOALS" in subset:
            row["Goals"] = subset["GOALS"].sum()
        elif "score" in subset:
            row["Goals"] = subset["score"].diff().clip(lower=0).sum()
        else:
            row["Goals"] = np.nan

        shot_xg = subset.get("SHOT_XG", pd.Series(dtype=float)).sum()
        packing_xg = subset.get("PACKING_XG", pd.Series(dtype=float)).sum()
        row["Shot-based xG"] = shot_xg
        row["Packing xG"] = packing_xg
        row["xGoals (Shot+Packing)"] = shot_xg + packing_xg

        pxt_columns = [
            column
            for column in [
                "PXT_PASS",
                "PXT_DRIBBLE",
                "PXT_SETPIECE",
                "PXT_SHOT",
                "PXT_BALL_WIN",
                "PXT_BLOCK",
            ]
            if column in subset
        ]
        row["Goal Threat gesamt (pxT)"] = (
            subset[pxt_columns].sum(axis=1).sum() if pxt_columns else np.nan
        )

        for column, label in TEAM_KPI_LABELS:
            if column == "GOALS":
                continue
            if column in subset:
                row[label] = subset[column].sum()

        if "OFFENSIVE_TOUCHES_IN_PITCH_POSITION_FINAL_THIRD" in subset:
            row.setdefault(
                "Offensive Touches – Final Third",
                subset["OFFENSIVE_TOUCHES_IN_PITCH_POSITION_FINAL_THIRD"].sum(),
            )

        rows.append(row)

    team_kpis = pd.DataFrame(rows)
    column_order = [
        "Team",
        "Goals",
        "Shot-based xG",
        "Packing xG",
        "Post-shot xG",
        "xGoals (Shot+Packing)",
        "Total Shots",
        "Critical Ball Losses",
        "Goal Threat gesamt (pxT)",
    ]
    column_order.extend([
        label
        for _column, label in TEAM_KPI_LABELS
        if label not in column_order and label in team_kpis.columns
    ])
    return team_kpis[[column for column in column_order if column in team_kpis.columns]]


def compute_xg_by_phase(events: pd.DataFrame) -> pd.DataFrame:
    """Return the accumulated xG per phase for every squad."""

    events = add_minutes_column(events)
    events["phase_group"] = events["phase"].apply(map_phase)

    xg_columns = [column for column in ["SHOT_XG", "PACKING_XG"] if column in events]
    if not xg_columns:
        return pd.DataFrame()

    events["xg_total"] = events[xg_columns].sum(axis=1)
    return (
        events.groupby(["squadName", "phase_group"])["xg_total"].sum().reset_index()
    )


def compute_xg_timeline(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate the cumulative xG development over time."""

    events = add_minutes_column(events)
    if "minute" not in events.columns:
        return pd.DataFrame()

    minute_series = pd.to_numeric(events["minute"], errors="coerce")
    max_minute = minute_series.max()
    if pd.isna(max_minute):
        max_minute = None

    squads = sorted({squad for squad in events.get("squadName", pd.Series(dtype=str)).dropna()})
    if not squads:
        return pd.DataFrame()

    xg_columns = [column for column in ["SHOT_XG", "PACKING_XG"] if column in events]
    if not xg_columns:
        xg_events = pd.DataFrame(columns=["minute", "cum_xg", "squadName"])
    else:
        events["xg"] = events[xg_columns].sum(axis=1)
        xg_events = events[events["xg"] > 0].copy()

    curves: List[pd.DataFrame] = []

    for squad in squads:
        subset = xg_events[xg_events["squadName"] == squad].copy()
        if subset.empty:
            base_minutes = [0.0]
            if max_minute and max_minute > 0:
                base_minutes.append(float(max_minute))
            curve = pd.DataFrame(
                {
                    "minute": base_minutes,
                    "cum_xg": [0.0] * len(base_minutes),
                    "squadName": squad,
                }
            )
        else:
            subset = subset.sort_values("minute")
            subset["cum_xg"] = subset["xg"].cumsum()
            curve = subset[["minute", "cum_xg"]].copy()
            curve.insert(0, "squadName", squad)
            curve = curve[["minute", "cum_xg", "squadName"]]
            if curve.iloc[0]["minute"] > 0:
                curve = pd.concat(
                    [
                        pd.DataFrame({"minute": [0.0], "cum_xg": [0.0], "squadName": [squad]}),
                        curve,
                    ],
                    ignore_index=True,
                )
            if max_minute and float(curve.iloc[-1]["minute"]) < float(max_minute):
                last_value = float(curve.iloc[-1]["cum_xg"])
                curve = pd.concat(
                    [
                        curve,
                        pd.DataFrame(
                            {
                                "minute": [float(max_minute)],
                                "cum_xg": [last_value],
                                "squadName": [squad],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

        curves.append(curve)

    if not curves:
        return pd.DataFrame()

    return pd.concat(curves, ignore_index=True)


def compute_player_ratings(
    events: pd.DataFrame, player_matchsums: List[Dict[str, float]], iteration_player_avgs: List[Dict[str, float]]
) -> pd.DataFrame:
    """Derive per player ratings from event level data."""

    del iteration_player_avgs  # currently unused but kept for future extensions

    events = events.copy()

    if "position" not in events.columns:
        position_aliases = [
            "playerPosition",
            "player_position",
            "positionName",
        ]
        for alias in position_aliases:
            if alias in events.columns:
                events["position"] = events[alias]
                break

    required_columns = {"playerId", "playerName", "squadId", "squadName", "position"}
    if not required_columns.issubset(events.columns):
        missing = ", ".join(sorted(required_columns - set(events.columns)))
        raise ValueError(f"Spalten fehlen für die Spieleraggregation: {missing}")

    canonical_positions = (
        events.groupby(["playerId", "playerName", "squadId", "squadName"])["position"]
        .agg(lambda values: most_common_value(values))
        .reset_index()
        .rename(columns={"position": "position_canonical"})
    )
    events = events.merge(
        canonical_positions,
        on=["playerId", "playerName", "squadId", "squadName"],
        how="left",
    )
    events["position"] = events["position_canonical"].fillna(events["position"]).fillna("Unknown")
    events = events.drop(columns="position_canonical")

    aggregations = {column: "sum" for column in [
        "SHOT_XG",
        "PACKING_XG",
        "EXPECTED_GOAL_ASSISTS",
        "EXPECTED_PASSES",
        "PXT_PASS",
        "PXT_DRIBBLE",
        "PXT_SETPIECE",
        "PXT_SHOT",
        "PXT_BALL_WIN",
        "PXT_BLOCK",
        "WON_GROUND_DUELS",
        "LOST_GROUND_DUELS",
        "WON_AERIAL_DUELS",
        "LOST_AERIAL_DUELS",
        "CRITICAL_BALL_LOSS_NUMBER",
        "NUMBER_OF_PRESSES",
        "BYPASSED_OPPONENTS_DEFENDERS_RAW",
    ] if column in events}

    if not aggregations:
        raise ValueError("Keine passenden Event-KPIs für Spieleraggregation gefunden.")

    players = (
        events.groupby(["playerId", "playerName", "squadId", "squadName", "position"]).agg(aggregations).reset_index()
    )

    players["xg"] = players.get("SHOT_XG", 0.0)
    players["xa"] = players.get("EXPECTED_GOAL_ASSISTS", 0.0)
    players["off_threat"] = (
        players.get("PXT_PASS", 0.0)
        + players.get("PXT_DRIBBLE", 0.0)
        + players.get("PXT_SETPIECE", 0.0)
        + players.get("PXT_SHOT", 0.0)
        + players.get("PXT_BALL_WIN", 0.0)
    )
    players["neg_actions"] = (
        players.get("LOST_GROUND_DUELS", 0.0)
        + players.get("LOST_AERIAL_DUELS", 0.0)
        + players.get("CRITICAL_BALL_LOSS_NUMBER", 0.0)
    )
    players["won_duels"] = players.get("WON_GROUND_DUELS", 0.0) + players.get("WON_AERIAL_DUELS", 0.0)
    players["lost_duels"] = players.get("LOST_GROUND_DUELS", 0.0) + players.get("LOST_AERIAL_DUELS", 0.0)
    players["presses"] = players.get("NUMBER_OF_PRESSES", 0.0)
    players["def_threat"] = players.get("PXT_BLOCK", 0.0)

    minutes = infer_minutes_from_matchsums(player_matchsums)
    players = players.merge(minutes, on="playerId", how="left")
    players["minutes"] = players["minutes"].fillna(0.0)

    players["pos_group"] = players["position"].apply(group_position)
    minutes_factor = np.minimum(players["minutes"] / 90.0, 1.0) ** 0.5

    players["off_raw"] = (
        players["xg"] * 4.0
        + players["xa"] * 3.0
        + players["off_threat"] * 1.0
        - players["neg_actions"] * 0.5
    ) * minutes_factor

    players["def_raw"] = (
        players["won_duels"] * 0.5
        + players["presses"] * 0.1
        + players["def_threat"] * 1.0
        - players["lost_duels"] * 0.5
        - players.get("CRITICAL_BALL_LOSS_NUMBER", 0.0) * 1.0
    ) * minutes_factor

    players["off_score"] = normalize_by_pos(players["off_raw"], players["pos_group"])
    players["def_score"] = normalize_by_pos(players["def_raw"], players["pos_group"])
    players["rating"] = (players["off_score"] * 0.6 + players["def_score"] * 0.4).clip(0, 10)

    columns = [
        "playerName",
        "squadName",
        "position",
        "pos_group",
        "minutes",
        "xg",
        "xa",
        "off_threat",
        "won_duels",
        "lost_duels",
        "presses",
        "neg_actions",
        "off_score",
        "def_score",
        "rating",
    ]

    return players[columns].sort_values("rating", ascending=False)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def build_match_figure(
    match_meta: Dict[str, object],
    team_kpis: pd.DataFrame,
    phase_xg: pd.DataFrame,
    timeline: pd.DataFrame,
    players: pd.DataFrame,
    player_scores: Optional[pd.DataFrame] = None,
    player_profile_scores: Optional[pd.DataFrame] = None,
    squad_scores: Optional[pd.DataFrame] = None,
    squad_ratings: Optional[pd.DataFrame] = None,
    squad_coefficients: Optional[pd.DataFrame] = None,
    set_pieces: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Create the final Plotly figure containing all components."""

    home = match_meta.get("homeSquadName", "?")
    away = match_meta.get("awaySquadName", "?")
    match_day = match_meta.get("matchDayIndex", "?")
    title = f"{home} vs {away} – Spieltag {match_day}"

    colors = get_team_colors(home, away)

    match_id = int(match_meta.get("id", -1))
    home_squad_id = match_meta.get("homeSquadId")
    away_squad_id = match_meta.get("awaySquadId")
    squad_ids = [
        squad_id
        for squad_id in [home_squad_id, away_squad_id]
        if squad_id is not None and not pd.isna(squad_id)
    ]
    match_date = parse_match_date(match_meta)

    players_table = players.copy().rename(
        columns={
            "playerName": "Playername",
            "squadName": "Squadname",
            "position": "Position",
            "pos_group": "Pos Group",
            "minutes": "Minutes",
            "xg": "xG",
            "xa": "xA",
            "off_threat": "Off Threat",
            "won_duels": "Won Duels",
            "lost_duels": "Lost Duels",
            "presses": "Presses",
            "neg_actions": "Neg Actions",
            "off_score": "Off Rating",
            "def_score": "Def Rating",
            "rating": "Rating",
        }
    )

    table_columns = [
        "Playername",
        "Squadname",
        "Position",
        "Pos Group",
        "Minutes",
        "xG",
        "xA",
        "Off Threat",
        "Won Duels",
        "Lost Duels",
        "Presses",
        "Neg Actions",
        "Off Rating",
        "Def Rating",
        "Rating",
    ]
    table_columns = [column for column in table_columns if column in players_table.columns]
    players_table = players_table[table_columns]

    pos_group_order = {
        "Goalkeeper": 0,
        "Defender": 1,
        "Midfielder": 2,
        "Forward": 3,
        "Other": 4,
    }

    player_sections: List[Tuple[str, pd.DataFrame]] = []
    for team in [home, away]:
        subset = players_table[players_table["Squadname"] == team].copy()
        if subset.empty:
            continue
        subset["pos_group_sort"] = subset["Pos Group"].map(pos_group_order).fillna(len(pos_group_order))
        subset = subset.sort_values(["pos_group_sort", "Position", "Rating"], ascending=[True, True, False])
        subset = subset.drop(columns=["pos_group_sort", "Squadname"]).reset_index(drop=True)

        numeric_columns = [column for column in subset.columns if is_numeric_dtype(subset[column])]
        if numeric_columns:
            subset[numeric_columns] = subset[numeric_columns].applymap(
                lambda value: round(float(value), 2) if pd.notna(value) else np.nan
            )

        player_sections.append((f"Spielerratings – {team}", subset))

    additional_tables: List[Tuple[str, pd.DataFrame]] = player_sections + [
        (
            "Player Match Scores",
            summarise_player_scores(player_scores, match_id),
        ),
        (
            "Player Profile Scores",
            summarise_player_profile_scores(player_profile_scores, squad_ids),
        ),
        (
            "Squad Match Scores",
            summarise_squad_scores(squad_scores, match_id),
        ),
        (
            "Squad Ratings",
            summarise_squad_ratings(squad_ratings, squad_ids, match_date),
        ),
        (
            "Squad Coefficients",
            summarise_squad_coefficients(squad_coefficients, squad_ids, match_date),
        ),
        (
            "Set Piece Übersicht",
            summarise_set_pieces(set_pieces),
        ),
    ]

    table_sections = [(title, table) for title, table in additional_tables if table is not None and not table.empty]

    num_tables = len(table_sections)
    total_rows = 3 + num_tables

    row_heights = [0.24, 0.2, 0.28] + [0.18] * num_tables
    subplot_titles = [
        "Teamvergleich",
        "xG nach Spielphasen (Aufbau / Konter / Pressing / Standard / Sonstige)",
        "xG-Verlauf",
    ] + [title for title, _ in table_sections]

    specs = [[{"type": "bar"}], [{"type": "bar"}], [{"type": "scatter"}]] + [[{"type": "table"}] for _ in table_sections]

    figure = make_subplots(
        rows=total_rows,
        cols=1,
        vertical_spacing=0.07,
        row_heights=row_heights,
        specs=specs,
        subplot_titles=tuple(subplot_titles),
    )

    team_kpis = team_kpis.copy()
    if set(team_kpis["Team"]) == {home, away}:
        team_kpis["sort_key"] = team_kpis["Team"].apply(lambda value: 0 if value == home else 1)
        team_kpis = team_kpis.sort_values("sort_key").drop(columns="sort_key")

    metrics = [column for column in team_kpis.columns if column != "Team"]
    home_values = pd.to_numeric(team_kpis.iloc[0][metrics], errors="coerce").to_numpy(dtype=float)
    away_values = pd.to_numeric(team_kpis.iloc[1][metrics], errors="coerce").to_numpy(dtype=float)
    totals = home_values + away_values

    with np.errstate(divide="ignore", invalid="ignore"):
        home_share = np.divide(home_values, totals, out=np.zeros_like(home_values), where=totals != 0)
        away_share = np.divide(away_values, totals, out=np.zeros_like(away_values), where=totals != 0)

    home_custom = np.column_stack([home_values, totals])
    away_custom = np.column_stack([away_values, totals])

    home_text = [f"{value:.0%}" if total > 0 else "" for value, total in zip(home_share, totals)]
    away_text = [f"{value:.0%}" if total > 0 else "" for value, total in zip(away_share, totals)]

    figure.add_trace(
        go.Bar(
            x=home_share,
            y=metrics,
            orientation="h",
            name=home,
            marker_color=colors["home_main"],
            customdata=home_custom,
            text=home_text,
            textposition="inside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{home}: %{{customdata[0]:.2f}}<br>Gesamt: %{{customdata[1]:.2f}}<br>"
                "Anteil: %{x:.0%}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=away_share,
            y=metrics,
            orientation="h",
            name=away,
            marker_color=colors["away_main"],
            customdata=away_custom,
            text=away_text,
            textposition="inside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{away}: %{{customdata[0]:.2f}}<br>Gesamt: %{{customdata[1]:.2f}}<br>"
                "Anteil: %{x:.0%}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    figure.update_xaxes(range=[0, 1], tickformat=".0%", row=1, col=1)

    if not phase_xg.empty:
        phases_order = ["Spielaufbau", "Konter", "Pressing / 2. Bälle", "Standard", "Sonstige"]
        pivot = (
            phase_xg.pivot_table(
                index="phase_group",
                columns="squadName",
                values="xg_total",
                aggfunc="sum",
            )
            .reindex(phases_order)
            .fillna(0.0)
        )

        phases = pivot.index.to_list()
        home_phase_values = pivot[home].to_numpy(dtype=float) if home in pivot else np.zeros(len(phases))
        away_phase_values = pivot[away].to_numpy(dtype=float) if away in pivot else np.zeros(len(phases))
        totals_phase = home_phase_values + away_phase_values

        valid_mask = totals_phase > 0
        if valid_mask.any():
            phases = [phase for phase, keep in zip(phases, valid_mask) if keep]
            home_phase_values = home_phase_values[valid_mask]
            away_phase_values = away_phase_values[valid_mask]
            totals_phase = totals_phase[valid_mask]

            with np.errstate(divide="ignore", invalid="ignore"):
                home_phase_share = np.divide(
                    home_phase_values, totals_phase, out=np.zeros_like(home_phase_values), where=totals_phase != 0
                )
                away_phase_share = np.divide(
                    away_phase_values, totals_phase, out=np.zeros_like(away_phase_values), where=totals_phase != 0
                )

            home_phase_custom = np.column_stack([home_phase_values, totals_phase])
            away_phase_custom = np.column_stack([away_phase_values, totals_phase])

            home_phase_text = [f"{value:.0%}" if total > 0 else "" for value, total in zip(home_phase_share, totals_phase)]
            away_phase_text = [f"{value:.0%}" if total > 0 else "" for value, total in zip(away_phase_share, totals_phase)]

            figure.add_trace(
                go.Bar(
                    x=home_phase_share,
                    y=phases,
                    orientation="h",
                    name=f"{home} xG-Anteil",
                    marker_color=colors["home_main"],
                    customdata=home_phase_custom,
                    text=home_phase_text,
                    textposition="inside",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        f"{home}: %{customdata[0]:.2f} xG<br>Gesamt: %{customdata[1]:.2f} xG<br>"
                        "Anteil: %{x:.0%}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
            figure.add_trace(
                go.Bar(
                    x=away_phase_share,
                    y=phases,
                    orientation="h",
                    name=f"{away} xG-Anteil",
                    marker_color=colors["away_main"],
                    customdata=away_phase_custom,
                    text=away_phase_text,
                    textposition="inside",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        f"{away}: %{customdata[0]:.2f} xG<br>Gesamt: %{customdata[1]:.2f} xG<br>"
                        "Anteil: %{x:.0%}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

            figure.update_xaxes(range=[0, 1], tickformat=".0%", row=2, col=1)
            figure.update_yaxes(categoryorder="array", categoryarray=phases[::-1], row=2, col=1)

    if not timeline.empty:
        for team, color in [(home, colors["home_main"]), (away, colors["away_main"])]:
            subset = timeline[timeline["squadName"] == team]
            if subset.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=subset["minute"],
                    y=subset["cum_xg"],
                    mode="lines+markers",
                    name=f"{team} xG",
                    line=dict(shape="hv", color=color),
                ),
                row=3,
                col=1,
            )

    figure.update_xaxes(title_text="Minute", row=3, col=1)
    figure.update_yaxes(title_text="xGoals", row=3, col=1)

    for index, (section_title, table) in enumerate(table_sections):
        table = table.copy()
        row_index = 4 + index
        if not table.empty:
            table = table.where(pd.notna(table), None)
            figure.add_trace(
                go.Table(
                    header=dict(
                        values=list(table.columns),
                        fill_color="#222222",
                        font=dict(color="white", size=11),
                        align="left",
                    ),
                    cells=dict(
                        values=[table[column].tolist() for column in table.columns],
                        fill_color="#F5F5F5",
                        align="left",
                    ),
                ),
                row=row_index,
                col=1,
            )

    figure.update_layout(
        title=title,
        barmode="stack",
        height=850 + 220 * num_tables,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return figure


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the guided match overview workflow."""

    username = USERNAME or _prompt("Benutzername (Impect): ", "")
    password = PASSWORD or _prompt("Passwort (Impect): ", "")
    if not username or not password:
        raise ValueError("Es werden gültige Zugangsdaten benötigt, um das Beispiel auszuführen.")

    token = get_access_token(username=username, password=password)
    iteration_id = choose_iteration(token)
    matchplan = get_matchplan(token, iteration_id)
    match_id = choose_match(matchplan)

    match_meta = matchplan[matchplan["id"] == match_id].iloc[0].to_dict()
    home = match_meta.get("homeSquadName", "?")
    away = match_meta.get("awaySquadName", "?")

    print("📦 Lade Events + KPIs via impectPy.getEvents(...)")
    events = pd.DataFrame(ip.getEvents([match_id], token=token))
    print(f"✅ {len(events)} Events im DataFrame")

    print("📦 Lade PlayerMatchsums...")
    player_matchsums = ip.getPlayerMatchsums([match_id], token)
    print(f"✅ {len(player_matchsums)} PlayerMatchsums-Einträge")

    print("\n🧾 Hole Spieler-Stammdaten via getPlayerIterationAverages()...")
    iteration_players = ip.getPlayerIterationAverages(iteration_id, token)
    print(f"✅ {len(iteration_players)} Spieler-Stammdatensätze geladen.\n")

    team_kpis = compute_team_kpis(events)
    phase_xg = compute_xg_by_phase(events)
    timeline = compute_xg_timeline(events)

    print("📊 Berechne Spielerratings (Offensiv & Defensiv)...")
    player_ratings = compute_player_ratings(events, player_matchsums, iteration_players)
    print(player_ratings.head(20).to_string(index=False))

    print("\n📦 Lade zusätzliche Kennzahlen (Scores & Ratings)...")
    additional_data = load_additional_match_data(
        match_id=match_id,
        iteration_id=iteration_id,
        token=token,
        events=events,
    )

    print("\n📈 Erzeuge Match-Übersichts-Grafik...")
    figure = build_match_figure(
        match_meta,
        team_kpis,
        phase_xg,
        timeline,
        player_ratings,
        player_scores=additional_data.get("player_scores"),
        player_profile_scores=additional_data.get("player_profile_scores"),
        squad_scores=additional_data.get("squad_scores"),
        squad_ratings=additional_data.get("squad_ratings"),
        squad_coefficients=additional_data.get("squad_coefficients"),
        set_pieces=additional_data.get("set_pieces"),
    )
    figure.show()


if __name__ == "__main__":
    main()
