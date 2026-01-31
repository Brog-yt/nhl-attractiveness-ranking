
# Get the large list from attractiveness_analysis.py
from typing import List, cast
import json
import time
from pathlib import Path
from players.attractiveness_analysis import attractive_players_data
from nhle_github import NhleGithub, SimplePlayer
from models import PlayerAttractiveAnalysis, SpecificPlayerInfo, SimpleSpecificPlayerData, SeasonStats

def convert_to_simple_player_data(
    specific_info: SpecificPlayerInfo, 
    player_analysis: PlayerAttractiveAnalysis
) -> SimpleSpecificPlayerData:
    """
    Convert SpecificPlayerInfo to SimpleSpecificPlayerData
    
    Args:
        specific_info: Full player information from the API
        player_analysis: Player attractiveness analysis data
        
    Returns:
        SimpleSpecificPlayerData: Simplified player data with attractiveness info
    """
    # Search for current season (20252026) in seasonTotals
    this_season_stats = None
    if specific_info.seasonTotals:
        for season_total in specific_info.seasonTotals:
            if season_total.season == 20252026:
                # Convert SeasonTotal to SeasonStats
                this_season_stats = SeasonStats(
                    goals=season_total.goals,
                    assists=season_total.assists,
                    points=season_total.points,
                    pim=season_total.pim,
                    plusMinus=season_total.plusMinus,
                    gamesPlayed=season_total.gamesPlayed,
                    avgToi=season_total.avgToi
                )
                break
    
    return SimpleSpecificPlayerData(
        # From PlayerAttractiveAnalysis (parent class)
        rank=player_analysis.rank,
        player=player_analysis.player,
        ridgeAttractivenessScore=player_analysis.ridgeAttractivenessScore,
        # From SimpleSpecificPlayerData
        playerId=specific_info.playerId,
        isActive=specific_info.isActive,
        currentTeamAbbrev=specific_info.currentTeamAbbrev,
        position=specific_info.position,
        birthCountry=specific_info.birthCountry,
        shootsCatches=specific_info.shootsCatches,
        birthDate=specific_info.birthDate,
        thisSeasonTotals=this_season_stats
    )

def get_attractive_players_with_stats() -> List[SimpleSpecificPlayerData]:
    # Convert the imported data from dicts to PlayerAttractiveAnalysis objects
    # Need to convert string names to Name objects
    players_with_attractive_scores: List[PlayerAttractiveAnalysis] = []
    for player_dict in attractive_players_data:
        # Convert firstName and lastName strings to Name objects
        player_data = player_dict.copy()
        player_data['player']['firstName'] = {'default': player_data['player']['firstName']}
        player_data['player']['lastName'] = {'default': player_data['player']['lastName']}
        players_with_attractive_scores.append(PlayerAttractiveAnalysis(**player_data))

    nhle = NhleGithub()
    players_stats_list: List[SimpleSpecificPlayerData] = []

    print(f"Fetching stats for {len(players_with_attractive_scores)} players...")
    
    for i, player_analysis in enumerate(players_with_attractive_scores, 1):
        player_id = player_analysis.player.id
        try:
            # Fetch player stats from nhle_github
            specific_info = nhle.get_player_stats(player_id)
            # Convert to simplified format with attractiveness data
            simple_data = convert_to_simple_player_data(specific_info, player_analysis)
            players_stats_list.append(simple_data)
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(players_with_attractive_scores)} players")
            
            # Wait for 1 second between requests to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"  Error fetching stats for player ID {player_id}: {e}")
    
    return players_stats_list

# Main function to test, write the results to a file called players/attractive_players_with_stats.json
if __name__ == "__main__":
    print("Starting to collect player data...")
    
    # Get all players with stats
    players_data = get_attractive_players_with_stats()
    
    print(f"\nSuccessfully collected data for {len(players_data)} players")
    
    # Create output directory
    output_dir = Path("players")
    output_dir.mkdir(exist_ok=True)
    
    # Write to JSON file
    output_file = output_dir / "attractive_players_with_stats.json"
    
    # Convert to JSON-serializable format
    output_data = [player.model_dump(exclude_none=True) for player in players_data]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Data written to: {output_file}")
    print("Done!")

