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
