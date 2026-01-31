import requests
from pathlib import Path
import json
from models import SimplePlayer, TeamRoster, SpecificPlayerInfo

allActiveTeams = [
    "ANA",  # Anaheim Ducks
    "BOS",  # Boston Bruins
    "BUF",  # Buffalo Sabres
    "CAR",  # Carolina Hurricanes
    "CBJ",  # Columbus Blue Jackets
    "CGY",  # Calgary Flames
    "CHI",  # Chicago Blackhawks
    "COL",  # Colorado Avalanche
    "DAL",  # Dallas Stars
    "DET",  # Detroit Red Wings
    "EDM",  # Edmonton Oilers
    "FLA",  # Florida Panthers
    "LAK",  # Los Angeles Kings
    "MIN",  # Minnesota Wild
    "MTL",  # Montreal Canadiens
    "NJD",  # New Jersey Devils
    "NSH",  # Nashville Predators
    "NYI",  # New York Islanders
    "NYR",  # New York Rangers
    "OTT",  # Ottawa Senators
    "PHI",  # Philadelphia Flyers
    "PIT",  # Pittsburgh Penguins
    "SEA",  # Seattle Kraken
    "SJS",  # San Jose Sharks
    "STL",  # St. Louis Blues
    "TBL",  # Tampa Bay Lightning
    "TOR",  # Toronto Maple Leafs
    "UTA",  # Utah Hockey Club (formerly Arizona)
    "VAN",  # Vancouver Canucks
    "VGK",  # Vegas Golden Knights
    "WPG",  # Winnipeg Jets
    "WSH",  # Washington Capitals
]

class NhleGithub:
    def __init__(self):
        self.season = "20252026"
        self.base_url = "https://api-web.nhle.com/v1/roster"

    def gat_all_players_on_all_teams(self) -> dict[str, TeamRoster]:
        all_teams_rosters = {}
        for team_code in allActiveTeams:
            roster = self.get_players_on_team(team_code)
            all_teams_rosters[team_code] = roster
        return all_teams_rosters

    # Return SimplePlayer list from TeamRoster
    def get_simplifiedPlayers(self, team_code: str) -> list[SimplePlayer]:
        roster = self.get_players_on_team(team_code)
        simple_players = []
        for player in roster.forwards + roster.defensemen + roster.goalies:
            # Skip players without valid headshots
            if not player.headshot or not player.headshot.strip():
                continue
            try:
                simple_player = SimplePlayer(
                    id=player.id,
                    headshot=player.headshot,
                    firstName=player.firstName,
                    lastName=player.lastName,
                )
                simple_players.append(simple_player)
            except Exception as e:
                # Skip players with invalid data
                print(f"    Skipping player {player.id}: {e}")
                continue
        return simple_players

    # https://api-web.nhle.com/v1/roster/TOR/20252026
    def get_players_on_team(self, team_code: str) -> TeamRoster:
        url = f"{self.base_url}/{team_code}/{self.season}"
        response = requests.get(url)
        response.raise_for_status()
        
        # Validate and parse the response
        roster = TeamRoster(**response.json())
        return roster

    # Get specific player stats
    # https://api-web.nhle.com/v1/player/8478402/landing
    def get_player_stats(self, player_id: int) -> SpecificPlayerInfo:
        url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
        response = requests.get(url)
        response.raise_for_status()
        return SpecificPlayerInfo(**response.json())
    
    def get_num_wins_for_team(self, team_code: str) -> float:
        """
        Get the points percentage for a specific team from league-standings.json.
        
        Args:
            team_code: Three-letter team code (e.g., 'TOR')
            
        Returns:
            Points percentage as a float (e.g., 0.625 for 62.5%)
        """
        # Load standings data from JSON file
        standings_file = Path(__file__).parent / "nhle" / "league-standings.json"
        
        try:
            with open(standings_file, 'r') as f:
                standings_data = json.load(f)
            
            # Search through standings for the team
            for standing in standings_data.get("standings", []):
                if standing.get("teamAbbrev", {}).get("default") == team_code or \
                   standing.get("teamAbbrev") == team_code:
                    return standing.get("pointPctg", 0.0)
            
            return 0.0
        except FileNotFoundError:
            print(f"Warning: league-standings.json not found at {standings_file}")
            return 0.0
        except Exception as e:
            print(f"Error reading league standings: {e}")
            return 0.0


# Main function, print how many teams are there
if __name__ == "__main__":
    print("Fetching NHL team rosters from nhle-github...")
    print(len(allActiveTeams))
