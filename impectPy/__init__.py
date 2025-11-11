# define version attribute
__version__ = "2.5.2"

import sys
from types import ModuleType

# import modules
from .access_token import getAccessToken
from .iterations import getIterations
from .matches import getMatches
from .events import getEvents
from .matchsums import getPlayerMatchsums, getSquadMatchsums
from .iteration_averages import getPlayerIterationAverages, getSquadIterationAverages
from .player_scores import getPlayerMatchScores, getPlayerIterationScores
from .squad_scores import getSquadMatchScores, getSquadIterationScores
from .player_profile_scores import getPlayerProfileScores
from .xml_generator import generateXML
from .set_pieces import getSetPieces
from .squad_ratings import getSquadRatings
from .squad_coefficients import getSquadCoefficients
from .match_info import getFormations, getSubstitutions, getStartingPositions
from .config import Config as Config
from .impect import Impect as Impect


if "impectPy.xml" not in sys.modules:
    _xml_module = ModuleType("impectPy.xml")
    _xml_module.generateXML = generateXML
    sys.modules["impectPy.xml"] = _xml_module