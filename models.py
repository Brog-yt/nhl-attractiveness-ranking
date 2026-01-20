from pydantic import BaseModel
from typing import Optional, List


class Name(BaseModel):
    default: str
    cs: Optional[str] = None
    fi: Optional[str] = None
    sk: Optional[str] = None
    de: Optional[str] = None
    es: Optional[str] = None
    sv: Optional[str] = None
    
    class Config:
        # Exclude None values when serializing
        exclude_none = True


class Player(BaseModel):
    id: int
    headshot: Optional[str] = ""
    firstName: Name
    lastName: Name
    sweaterNumber: Optional[int] = None
    positionCode: str
    shootsCatches: str
    heightInInches: Optional[int] = None
    weightInPounds: Optional[int] = None
    heightInCentimeters: Optional[int] = None
    weightInKilograms: Optional[int] = None
    birthDate: str
    birthCity: Name
    birthCountry: str
    birthStateProvince: Optional[Name] = None

class SimplePlayer(BaseModel):
    id: int
    headshot: str
    firstName: Name
    lastName: Name

class TeamRoster(BaseModel):
    forwards: List[Player]
    defensemen: List[Player]
    goalies: List[Player]


class PlayerAttractiveAnalysis(BaseModel):
    rank: int
    player: SimplePlayer
    ridgeAttractivenessScore: float


class Badge(BaseModel):
    logoUrl: Name
    title: Name


class DraftDetails(BaseModel):
    year: int
    teamAbbrev: str
    round: int
    pickInRound: int
    overallPick: int


class SeasonStats(BaseModel):
    goals: Optional[int] = None
    assists: Optional[int] = None
    points: Optional[int] = None
    pim: Optional[int] = None
    plusMinus: Optional[int] = None
    gamesPlayed: Optional[int] = None
    avgToi: Optional[str] = None


class RegularSeasonStats(BaseModel):
    subSeason: Optional[SeasonStats] = None
    career: Optional[SeasonStats] = None


class FeaturedStats(BaseModel):
    season: int
    regularSeason: Optional[RegularSeasonStats] = None


class CareerStats(BaseModel):
    assists: Optional[int] = None
    avgToi: Optional[str] = None
    faceoffWinningPctg: Optional[float] = None
    gameWinningGoals: Optional[int] = None
    gamesPlayed: Optional[int] = None
    goals: Optional[int] = None
    otGoals: Optional[int] = None
    pim: Optional[int] = None
    plusMinus: Optional[int] = None
    points: Optional[int] = None
    powerPlayGoals: Optional[int] = None
    powerPlayPoints: Optional[int] = None
    shootingPctg: Optional[float] = None
    shorthandedGoals: Optional[int] = None
    shorthandedPoints: Optional[int] = None
    shots: Optional[int] = None


class CareerTotals(BaseModel):
    regularSeason: Optional[CareerStats] = None
    playoffs: Optional[CareerStats] = None


class GameLog(BaseModel):
    assists: Optional[int] = None
    gameDate: str
    gameId: int
    gameTypeId: int
    goals: Optional[int] = None
    homeRoadFlag: str
    opponentAbbrev: str
    pim: Optional[int] = None
    plusMinus: Optional[int] = None
    points: Optional[int] = None
    powerPlayGoals: Optional[int] = None
    shifts: Optional[int] = None
    shorthandedGoals: Optional[int] = None
    shots: Optional[int] = None
    teamAbbrev: str
    toi: str


class SeasonTotal(BaseModel):
    assists: Optional[int] = None
    avgToi: Optional[str] = None
    gameTypeId: int
    gameWinningGoals: Optional[int] = None
    gamesPlayed: Optional[int] = None
    goals: Optional[int] = None
    leagueAbbrev: str
    pim: Optional[int] = None
    plusMinus: Optional[int] = None
    points: Optional[int] = None
    powerPlayGoals: Optional[int] = None
    powerPlayPoints: Optional[int] = None
    season: int
    sequence: int
    shorthandedGoals: Optional[int] = None
    shorthandedPoints: Optional[int] = None
    shots: Optional[int] = None
    teamName: Name
    teamCommonName: Optional[Name] = None
    teamPlaceNameWithPreposition: Optional[Name] = None


class SpecificPlayerInfo(BaseModel):
    playerId: int
    isActive: bool
    currentTeamId: Optional[int] = None
    currentTeamAbbrev: Optional[str] = None
    fullTeamName: Optional[Name] = None
    teamCommonName: Optional[Name] = None
    teamPlaceNameWithPreposition: Optional[Name] = None
    firstName: Name
    lastName: Name
    badges: Optional[List[Badge]] = None
    teamLogo: Optional[str] = None
    sweaterNumber: Optional[int] = None
    position: str
    headshot: str
    heroImage: Optional[str] = None
    heightInInches: Optional[int] = None
    heightInCentimeters: Optional[int] = None
    weightInPounds: Optional[int] = None
    weightInKilograms: Optional[int] = None
    birthDate: str
    birthCity: Name
    birthStateProvince: Optional[Name] = None
    birthCountry: str
    shootsCatches: str
    draftDetails: Optional[DraftDetails] = None
    playerSlug: str
    inTop100AllTime: int
    inHHOF: int
    featuredStats: Optional[FeaturedStats] = None
    careerTotals: Optional[CareerTotals] = None
    shopLink: Optional[str] = None
    twitterLink: Optional[str] = None
    watchLink: Optional[str] = None
    last5Games: Optional[List[GameLog]] = None
    seasonTotals: Optional[List[SeasonTotal]] = None


class SimpleSpecificPlayerData(PlayerAttractiveAnalysis):
    playerId: int
    isActive: bool
    currentTeamAbbrev: Optional[str] = None
    position: str
    birthCountry: str
    shootsCatches: str
    birthDate: str
    thisSeasonTotals: Optional[SeasonStats] = None
