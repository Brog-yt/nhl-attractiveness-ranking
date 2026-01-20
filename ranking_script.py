import json
from pathlib import Path
from typing import List
from models import SimpleSpecificPlayerData


def load_attractive_players_with_stats() -> List[SimpleSpecificPlayerData]:
    """
    Load player data from attractive_players_with_stats.json and cast to SimpleSpecificPlayerData
    
    Returns:
        List[SimpleSpecificPlayerData]: List of players with their stats and attractiveness scores
    """
    # Load the JSON file
    data_file = Path(__file__).parent / "players" / "attractive_players_with_stats.json"
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Cast each dict to SimpleSpecificPlayerData
    players: List[SimpleSpecificPlayerData] = [
        SimpleSpecificPlayerData(**player_dict) for player_dict in data
    ]
    
    return players


if __name__ == "__main__":
    # Load the data
    players = load_attractive_players_with_stats()
    
    print(f"Loaded {len(players)} players with stats")
    
    # Display first few players as example
    for i, player in enumerate(players[:5], 1):
        print(f"\n{i}. Rank #{player.rank}: {player.player.firstName.default} {player.player.lastName.default}")
        print(f"   Attractiveness Score: {player.ridgeAttractivenessScore:.2f}")
        print(f"   Position: {player.position} | Team: {player.currentTeamAbbrev}")
        if player.thisSeasonTotals:
            stats = player.thisSeasonTotals
            print(f"   This Season: {stats.gamesPlayed} GP, {stats.goals}G {stats.assists}A {stats.points}P")
            if stats.avgToi:
                print(f"   Avg TOI: {stats.avgToi}")
