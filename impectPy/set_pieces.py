# load packages
import pandas as pd
import requests
from impectPy.helpers import RateLimitedAPI
from .matches import getMatchesFromHost
from .iterations import getIterationsFromHost
import re
from typing import Any, Dict, List, Tuple


def _normalize_aggregate_key(label: Any) -> str:
    """Return a clean column suffix derived from ``label``."""

    text = str(label) if label is not None else "UNSPECIFIED"
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return normalized.upper() if normalized else "UNSPECIFIED"


def _list_entry_to_dict(entries: List[Any]) -> Dict[str, Any]:
    """Convert a list of aggregate entries into a dictionary."""

    result: Dict[str, Any] = {}
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            key = None
            for candidate in [
                "category",
                "type",
                "name",
                "label",
                "key",
                "id",
            ]:
                if candidate in entry and entry[candidate] not in (None, ""):
                    key = entry[candidate]
                    break

            value = entry.get("value")
            if value is None:
                for candidate in ["amount", "sum", "total"]:
                    if candidate in entry:
                        value = entry[candidate]
                        break
            if key is None:
                if len(entry) == 1:
                    key, value = next(iter(entry.items()))
                else:
                    key = str(index)
            result[str(key)] = value
        else:
            result[str(index)] = entry
    return result


def _expand_nested_aggregate_column(
    dataframe: pd.DataFrame,
    column: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Expand nested aggregate data and return new columns."""

    if column not in dataframe.columns:
        return dataframe, []

    series = dataframe[column]
    if not series.dropna().apply(lambda value: isinstance(value, (dict, list))).any():
        return dataframe, []

    normalized = []
    for value in series:
        if isinstance(value, dict):
            normalized.append(value)
        elif isinstance(value, list):
            normalized.append(_list_entry_to_dict(value))
        else:
            normalized.append({})

    if not any(normalized):
        return dataframe, []

    expanded = pd.json_normalize(normalized).fillna(0.0)
    expanded.columns = [
        f"{column}_{_normalize_aggregate_key(name)}" for name in expanded.columns
    ]

    numeric_expanded = expanded.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    total_candidates = [
        col
        for col in numeric_expanded.columns
        if col.upper().endswith("_TOTAL")
        or col.upper().endswith("_SUM")
        or col.upper().endswith("_ALL")
    ]

    raw_total = None
    if total_candidates:
        total_column = total_candidates[0]
        raw_total = numeric_expanded[total_column]
        numeric_expanded = numeric_expanded.drop(columns=[total_column])

    if not numeric_expanded.empty:
        recomputed_total = numeric_expanded.sum(axis=1, numeric_only=True)
    else:
        recomputed_total = (
            raw_total.fillna(0.0) if raw_total is not None else pd.Series(0.0, index=dataframe.index)
        )

    result = dataframe.drop(columns=[column]).copy()
    result[column] = recomputed_total

    new_columns: List[str] = []
    if raw_total is not None:
        result[f"{column}_SOURCE_TOTAL"] = raw_total
        new_columns.append(f"{column}_SOURCE_TOTAL")

    if not numeric_expanded.empty:
        result = pd.concat([result, numeric_expanded], axis=1)
        new_columns.extend(numeric_expanded.columns.tolist())

    return result, new_columns

######
#
# This function returns a pandas dataframe that contains all set pieces for a
# given match
#
######


# define function
def getSetPieces(matches: list, token: str, session: requests.Session = requests.Session()) -> pd.DataFrame:

    # create an instance of RateLimitedAPI
    connection = RateLimitedAPI(session)

    # construct header with access token
    connection.session.headers.update({"Authorization": f"Bearer {token}"})

    return getSetPiecesFromHost(matches, connection, "https://api.impect.com")

def getSetPiecesFromHost(matches: list, connection: RateLimitedAPI, host: str) -> pd.DataFrame:

    # check input for matches argument
    if not isinstance(matches, list):
        raise Exception("Argument 'matches' must be a list of integers.")

    # get match info
    iterations = pd.concat(
        map(lambda match: connection.make_api_request_limited(
            url=f"{host}/v5/customerapi/matches/{match}",
            method="GET"
        ).process_response(
            endpoint="Iterations"
        ),
            matches),
        ignore_index=True)

    # filter for matches that are unavailable
    fail_matches = iterations[iterations.lastCalculationDate.isnull()].id.drop_duplicates().to_list()

    # drop matches that are unavailable from list of matches
    matches = [match for match in matches if match not in fail_matches]

    # raise exception if no matches remaining or report removed matches
    if len(fail_matches) > 0:
        if len(matches) == 0:
            raise Exception("All supplied matches are unavailable. Execution stopped.")
        else:
            print(f"The following matches are not available yet and were ignored:\n{fail_matches}")

    # extract iterationIds
    iterations = list(iterations[iterations.lastCalculationDate.notnull()].iterationId.unique())

    # get players
    players = pd.concat(
        map(lambda iteration: connection.make_api_request_limited(
            url=f"{host}/v5/customerapi/iterations/{iteration}/players",
            method="GET"
        ).process_response(
            endpoint="Players"
        ),
            iterations),
        ignore_index=True)[["id", "commonname"]].drop_duplicates()

    # get squads
    squads = pd.concat(
        map(lambda iteration: connection.make_api_request_limited(
            url=f"{host}/v5/customerapi/iterations/{iteration}/squads",
            method="GET"
        ).process_response(
            endpoint="Squads"
        ),
            iterations),
        ignore_index=True)[["id", "name"]].drop_duplicates()

    # get matches
    matchplan = pd.concat(
        map(lambda iteration: getMatchesFromHost(
            iteration=iteration,
            connection=connection,
            host=host
        ),
            iterations),
        ignore_index=True)

    # get iterations
    iterations = getIterationsFromHost(connection=connection, host=host)

    # get set piece data
    set_pieces = pd.concat(
        map(lambda match: connection.make_api_request_limited(
            url=f"{host}/v5/customerapi/matches/{match}/set-pieces",
            method="GET"
        ).process_response(
            endpoint="Set-Pieces"
        ),
            matches),
        ignore_index=True
    ).rename(
        columns={"id": "setPieceId"}
    ).explode("setPieceSubPhase", ignore_index=True)

    # unpack setPieceSubPhase column
    set_pieces = pd.concat(
        [
            set_pieces.drop(columns=["setPieceSubPhase"]),
            pd.json_normalize(set_pieces["setPieceSubPhase"]).add_prefix("setPieceSubPhase.")
        ],
        axis=1
    ).rename(columns=lambda x: re.sub(r"\.(.)", lambda y: y.group(1).upper(), x))

    # fix typing
    set_pieces.setPieceSubPhaseMainEventPlayerId = set_pieces.setPieceSubPhaseMainEventPlayerId.astype("Int64")
    set_pieces.setPieceSubPhaseFirstTouchPlayerId = set_pieces.setPieceSubPhaseFirstTouchPlayerId.astype("Int64")
    set_pieces.setPieceSubPhaseSecondTouchPlayerId = set_pieces.setPieceSubPhaseSecondTouchPlayerId.astype("Int64")

    # start merging dfs

    # merge events with matches
    set_pieces = set_pieces.merge(
        matchplan,
        left_on="matchId",
        right_on="id",
        how="left",
        suffixes=("", "_right")
    )

    # merge with competition info
    set_pieces = set_pieces.merge(
        iterations,
        left_on="iterationId",
        right_on="id",
        how="left",
        suffixes=("", "_right")
    )

    # determine defending squad
    set_pieces["defendingSquadId"] = set_pieces.apply(
        lambda row: row.homeSquadId if row.squadId == row.awaySquadId else row.awaySquadId,
        axis=1
    )

    # merge events with squads
    set_pieces = set_pieces.merge(
        squads[["id", "name"]].rename(columns={"id": "squadId", "name": "attackingSquadName"}),
        left_on="squadId",
        right_on="squadId",
        how="left",
        suffixes=("", "_home")
    ).merge(
        squads[["id", "name"]].rename(columns={"id": "squadId", "name": "defendingSquadName"}),
        left_on="defendingSquadId",
        right_on="squadId",
        how="left",
        suffixes=("", "_away")
    )

    # merge events with players
    set_pieces = set_pieces.merge(
        players[["id", "commonname"]].rename(
            columns={
                "id": "setPieceSubPhaseMainEventPlayerId",
                "commonname": "setPieceSubPhaseMainEventPlayerName"
            }
        ),
        left_on="setPieceSubPhaseMainEventPlayerId",
        right_on="setPieceSubPhaseMainEventPlayerId",
        how="left",
        suffixes=("", "_right")
    ).merge(
        players[["id", "commonname"]].rename(
            columns={
                "id": "setPieceSubPhasePassReceiverId",
                "commonname": "setPieceSubPhasePassReceiverName"
            }
        ),
        left_on="setPieceSubPhasePassReceiverId",
        right_on="setPieceSubPhasePassReceiverId",
        how="left",
        suffixes=("", "_right")
    ).merge(
        players[["id", "commonname"]].rename(
            columns={
                "id": "setPieceSubPhaseFirstTouchPlayerId",
                "commonname": "setPieceSubPhaseFirstTouchPlayerName"
            }
        ),
        left_on="setPieceSubPhaseFirstTouchPlayerId",
        right_on="setPieceSubPhaseFirstTouchPlayerId",
        how="left",
        suffixes=("", "_right")
    ).merge(
        players[["id", "commonname"]].rename(
            columns={
                "id": "setPieceSubPhaseSecondTouchPlayerId",
                "commonname": "setPieceSubPhaseSecondTouchPlayerName"
            }
        ),
        left_on="setPieceSubPhaseSecondTouchPlayerId",
        right_on="setPieceSubPhaseSecondTouchPlayerId",
        how="left",
        suffixes=("", "_right")
    )

    # rename some columns
    set_pieces = set_pieces.rename(columns={
        "scheduledDate": "dateTime",
        "squadId": "attackingSquadId",
        "phaseIndex": "setPiecePhaseIndex",
        "setPieceSubPhaseAggregatesSHOT_XG": "setPieceSubPhase_SHOT_XG",
        "setPieceSubPhaseAggregatesPACKING_XG": "setPieceSubPhase_PACKING_XG",
        "setPieceSubPhaseAggregatesPOSTSHOT_XG": "setPieceSubPhase_POSTSHOT_XG",
        "setPieceSubPhaseAggregatesSHOT_AT_GOAL_NUMBER": "setPieceSubPhase_SHOT_AT_GOAL_NUMBER",
        "setPieceSubPhaseAggregatesGOALS": "setPieceSubPhase_GOALS",
        "setPieceSubPhaseAggregatesPXT_POSITIVE": "setPieceSubPhase_PXT_POSITIVE",
        "setPieceSubPhaseAggregatesBYPASSED_OPPONENTS": "setPieceSubPhase_BYPASSED_OPPONENTS",
        "setPieceSubPhaseAggregatesBYPASSED_DEFENDERS": "setPieceSubPhase_BYPASSED_DEFENDERS"
    })

    aggregate_columns = [
        "setPieceSubPhase_SHOT_XG",
        "setPieceSubPhase_PACKING_XG",
        "setPieceSubPhase_POSTSHOT_XG",
    ]

    dynamic_columns: List[str] = []
    for column in aggregate_columns:
        set_pieces, new_columns = _expand_nested_aggregate_column(set_pieces, column)
        dynamic_columns.extend(new_columns)

    # define desired column order
    order = [
        "matchId",
        "dateTime",
        "competitionName",
        "competitionId",
        "competitionType",
        "iterationId",
        "season",
        "attackingSquadId",
        "attackingSquadName",
        "defendingSquadId",
        "defendingSquadName",
        "setPieceId",
        "setPiecePhaseIndex",
        "setPieceCategory",
        "adjSetPieceCategory",
        "setPieceExecutionType",
        "setPieceSubPhaseId",
        "setPieceSubPhaseIndex",
        "setPieceSubPhaseStartZone",
        "setPieceSubPhaseCornerEndZone",
        "setPieceSubPhaseCornerType",
        "setPieceSubPhaseFreeKickEndZone",
        "setPieceSubPhaseFreeKickType",
        "setPieceSubPhaseMainEventPlayerId",
        "setPieceSubPhaseMainEventPlayerName",
        "setPieceSubPhaseMainEventOutcome",
        "setPieceSubPhasePassReceiverId",
        "setPieceSubPhasePassReceiverName",
        "setPieceSubPhaseFirstTouchPlayerId",
        "setPieceSubPhaseFirstTouchPlayerName",
        "setPieceSubPhaseFirstTouchWon",
        "setPieceSubPhaseIndirectHeader",
        "setPieceSubPhaseSecondTouchPlayerId",
        "setPieceSubPhaseSecondTouchPlayerName",
        "setPieceSubPhaseSecondTouchWon",
        "setPieceSubPhase_SHOT_XG",
        "setPieceSubPhase_PACKING_XG",
        "setPieceSubPhase_POSTSHOT_XG",
        "setPieceSubPhase_SHOT_AT_GOAL_NUMBER",
        "setPieceSubPhase_GOALS",
        "setPieceSubPhase_PXT_POSITIVE",
        "setPieceSubPhase_BYPASSED_OPPONENTS",
        "setPieceSubPhase_BYPASSED_DEFENDERS",
    ]

    if dynamic_columns:
        for column in dict.fromkeys(dynamic_columns):
            if column not in order:
                order.append(column)

    # reorder data
    set_pieces = set_pieces[order]

    # reorder rows
    set_pieces = set_pieces.sort_values(["matchId", "setPiecePhaseIndex"])

    # return events
    return set_pieces
