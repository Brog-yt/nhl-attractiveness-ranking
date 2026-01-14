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


class Player(BaseModel):
    id: int
    headshot: str
    firstName: Name
    lastName: Name
    sweaterNumber: int
    positionCode: str
    shootsCatches: str
    heightInInches: int
    weightInPounds: int
    heightInCentimeters: int
    weightInKilograms: int
    birthDate: str
    birthCity: Name
    birthCountry: str
    birthStateProvince: Optional[Name] = None

class SimplePlayer(BaseModel):
    headshot: str
    firstName: Name
    lastName: Name

class TeamRoster(BaseModel):
    forwards: List[Player]
    defensemen: List[Player]
    goalies: List[Player]
