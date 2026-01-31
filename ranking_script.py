import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from models import SimpleSpecificPlayerData
from nhle_github import NhleGithub
import numpy as np
from scipy import stats


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


def get_position_group(position: str) -> str:
    """
    Map individual positions to position groups.
    C, L, R -> Forward
    D -> Defense
    G -> Goalie
    """
    if position in ['C', 'L', 'R']:
        return 'Forward'
    elif position == 'D':
        return 'Defense'
    elif position == 'G':
        return 'Goalie'
    else:
        return 'Unknown'


if __name__ == "__main__":
    # Load the data
    players = load_attractive_players_with_stats()
    
    print(f"Loaded {len(players)} players with stats\n")
    
    # Group players by country
    countries: Dict[str, List[SimpleSpecificPlayerData]] = defaultdict(list)
    for player in players:
        countries[player.birthCountry].append(player)
    
    # Calculate average attractiveness score for each country
    country_stats = []
    for country, country_players in countries.items():
        avg_score = sum(p.ridgeAttractivenessScore for p in country_players) / len(country_players)
        country_stats.append({
            'country': country,
            'avg_score': avg_score,
            'player_count': len(country_players)
        })
    
    # Sort countries by average attractiveness score (descending)
    country_stats.sort(key=lambda x: x['avg_score'], reverse=True)
    
    # Print the rankings
    print("=" * 60)
    print("NHL PLAYER ATTRACTIVENESS RANKINGS BY COUNTRY")
    print("=" * 60)
    print(f"{'Rank':<6} {'Country':<10} {'Avg Score':<12} {'Players':<10}")
    print("-" * 60)
    
    for i, stat in enumerate(country_stats, 1):
        print(f"{i:<6} {stat['country']:<10} {stat['avg_score']:<12.4f} {stat['player_count']:<10}")
    
    print("=" * 60)
    print()
    
    # Group players by position group
    positions: Dict[str, List[SimpleSpecificPlayerData]] = defaultdict(list)
    for player in players:
        position_group = get_position_group(player.position)
        positions[position_group].append(player)
    
    # Calculate average attractiveness score for each position group
    position_stats = []
    for position, position_players in positions.items():
        avg_score = sum(p.ridgeAttractivenessScore for p in position_players) / len(position_players)
        position_stats.append({
            'position': position,
            'avg_score': avg_score,
            'player_count': len(position_players)
        })
    
    # Sort positions by average attractiveness score (descending)
    position_stats.sort(key=lambda x: x['avg_score'], reverse=True)
    
    # Print the position rankings
    print("=" * 60)
    print("NHL PLAYER ATTRACTIVENESS RANKINGS BY POSITION")
    print("=" * 60)
    print(f"{'Rank':<6} {'Position':<12} {'Avg Score':<12} {'Players':<10}")
    print("-" * 60)
    
    for i, stat in enumerate(position_stats, 1):
        print(f"{i:<6} {stat['position']:<12} {stat['avg_score']:<12.4f} {stat['player_count']:<10}")
    
    print("=" * 60)
    print()
    
    # Group players by team
    teams: Dict[str, List[SimpleSpecificPlayerData]] = defaultdict(list)
    for player in players:
        if player.currentTeamAbbrev:
            teams[player.currentTeamAbbrev].append(player)
    
    # Calculate average attractiveness score for each team
    team_stats = []
    nhle = NhleGithub()
    
    for team, team_players in teams.items():
        avg_score = sum(p.ridgeAttractivenessScore for p in team_players) / len(team_players)
        points_pct = nhle.get_num_wins_for_team(team)
        
        team_stats.append({
            'team': team,
            'avg_score': avg_score,
            'player_count': len(team_players),
            'points_pct': points_pct
        })
    
    # Sort teams by average attractiveness score (descending)
    team_stats.sort(key=lambda x: x['avg_score'], reverse=True)
    
    # Print the team rankings
    print("=" * 80)
    print("NHL PLAYER ATTRACTIVENESS RANKINGS BY TEAM")
    print("=" * 80)
    print(f"{'Rank':<6} {'Team':<10} {'Avg Score':<12} {'Players':<10} {'Points %':<12}")
    print("-" * 80)
    
    for i, stat in enumerate(team_stats, 1):
        points_pct_str = f"{stat['points_pct']*100:.1f}%" if stat['points_pct'] > 0 else "N/A"
        print(f"{i:<6} {stat['team']:<10} {stat['avg_score']:<12.4f} {stat['player_count']:<10} {points_pct_str:<12}")
    
    print("=" * 80)
    print()
    
    # Advanced team ranking weighted by ice time and games played
    def convert_toi_to_minutes(toi_str: str) -> float:
        """Convert time on ice string (MM:SS) to minutes as float"""
        if not toi_str:
            return 0.0
        parts = toi_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes + (seconds / 60.0)
        return 0.0
    
    weighted_team_stats = []
    for team, team_players in teams.items():
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for player in team_players:
            if player.thisSeasonTotals and player.thisSeasonTotals.gamesPlayed and player.thisSeasonTotals.avgToi:
                games_played = player.thisSeasonTotals.gamesPlayed or 0
                avg_toi_minutes = convert_toi_to_minutes(player.thisSeasonTotals.avgToi)
                
                # Weight = games played * average TOI (in minutes)
                weight = games_played * avg_toi_minutes
                
                total_weighted_score += player.ridgeAttractivenessScore * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_avg_score = total_weighted_score / total_weight
            points_pct = nhle.get_num_wins_for_team(team)
            
            weighted_team_stats.append({
                'team': team,
                'weighted_avg_score': weighted_avg_score,
                'points_pct': points_pct,
                'player_count': len(team_players)
            })
    
    # Sort teams by weighted average attractiveness score (descending)
    weighted_team_stats.sort(key=lambda x: x['weighted_avg_score'], reverse=True)
    
    # Print the weighted team rankings
    print("=" * 90)
    print("NHL PLAYER ATTRACTIVENESS RANKINGS BY TEAM (Weighted by Ice Time)")
    print("=" * 90)
    print(f"{'Rank':<6} {'Team':<10} {'Weighted Avg':<15} {'Points %':<12} {'Players':<10}")
    print("-" * 90)
    
    for i, stat in enumerate(weighted_team_stats, 1):
        points_pct_str = f"{stat['points_pct']*100:.1f}%" if stat['points_pct'] > 0 else "N/A"
        print(f"{i:<6} {stat['team']:<10} {stat['weighted_avg_score']:<15.4f} {points_pct_str:<12} {stat['player_count']:<10}")
    
    print("=" * 90)
    print()
    
    # Correlation Analysis: Attractiveness vs Goals Scored
    print("=" * 80)
    print("CORRELATION ANALYSIS: Attractiveness vs Performance Stats")
    print("=" * 80)
    
    # Collect data for players with valid stats
    attractiveness_scores = []
    goals_list = []
    assists_list = []
    points_list = []
    pim_list = []
    
    for player in players:
        if player.thisSeasonTotals and player.thisSeasonTotals.goals is not None:
            attractiveness_scores.append(player.ridgeAttractivenessScore)
            goals_list.append(player.thisSeasonTotals.goals)
            assists_list.append(player.thisSeasonTotals.assists or 0)
            points_list.append(player.thisSeasonTotals.points or 0)
            pim_list.append(player.thisSeasonTotals.pim or 0)
    
    # Convert to numpy arrays for analysis
    attractiveness_array = np.array(attractiveness_scores)
    goals_array = np.array(goals_list)
    assists_array = np.array(assists_list)
    points_array = np.array(points_list)
    pim_array = np.array(pim_list)
    
    # Calculate Pearson correlation coefficients
    goals_corr, goals_pvalue = stats.pearsonr(attractiveness_array, goals_array)
    assists_corr, assists_pvalue = stats.pearsonr(attractiveness_array, assists_array)
    points_corr, points_pvalue = stats.pearsonr(attractiveness_array, points_array)
    pim_corr, pim_pvalue = stats.pearsonr(attractiveness_array, pim_array)
    
    # Calculate Spearman rank correlation (non-parametric alternative)
    goals_spearman, goals_spearman_pvalue = stats.spearmanr(attractiveness_array, goals_array)
    assists_spearman, assists_spearman_pvalue = stats.spearmanr(attractiveness_array, assists_array)
    points_spearman, points_spearman_pvalue = stats.spearmanr(attractiveness_array, points_array)
    pim_spearman, pim_spearman_pvalue = stats.spearmanr(attractiveness_array, pim_array)
    
    print(f"\nSample Size: {len(attractiveness_scores)} players with stats\n")
    
    print("PEARSON CORRELATION (measures linear relationship):")
    print("-" * 80)
    print(f"{'Metric':<20} {'Correlation':<15} {'P-Value':<15} {'Interpretation'}")
    print("-" * 80)
    
    def interpret_correlation(corr, pvalue):
        """Interpret correlation strength and significance"""
        # Significance
        if pvalue < 0.001:
            sig = "highly significant ***"
        elif pvalue < 0.01:
            sig = "very significant **"
        elif pvalue < 0.05:
            sig = "significant *"
        else:
            sig = "not significant"
        
        # Strength and direction
        abs_corr = abs(corr)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if corr > 0 else "negative"
        
        return f"{strength} {direction}, {sig}"
    
    print(f"{'Goals':<20} {goals_corr:<15.4f} {goals_pvalue:<15.6f} {interpret_correlation(goals_corr, goals_pvalue)}")
    print(f"{'Assists':<20} {assists_corr:<15.4f} {assists_pvalue:<15.6f} {interpret_correlation(assists_corr, assists_pvalue)}")
    print(f"{'Points':<20} {points_corr:<15.4f} {points_pvalue:<15.6f} {interpret_correlation(points_corr, points_pvalue)}")
    print(f"{'Penalty Minutes':<20} {pim_corr:<15.4f} {pim_pvalue:<15.6f} {interpret_correlation(pim_corr, pim_pvalue)}")
    
    print("\n\nSPEARMAN RANK CORRELATION (measures monotonic relationship):")
    print("-" * 80)
    print(f"{'Metric':<20} {'Correlation':<15} {'P-Value':<15} {'Interpretation'}")
    print("-" * 80)
    print(f"{'Goals':<20} {goals_spearman:<15.4f} {goals_spearman_pvalue:<15.6f} {interpret_correlation(goals_spearman, goals_spearman_pvalue)}")
    print(f"{'Assists':<20} {assists_spearman:<15.4f} {assists_spearman_pvalue:<15.6f} {interpret_correlation(assists_spearman, assists_spearman_pvalue)}")
    print(f"{'Points':<20} {points_spearman:<15.4f} {points_spearman_pvalue:<15.6f} {interpret_correlation(points_spearman, points_spearman_pvalue)}")
    print(f"{'Penalty Minutes':<20} {pim_spearman:<15.4f} {pim_spearman_pvalue:<15.6f} {interpret_correlation(pim_spearman, pim_spearman_pvalue)}")
    
    print("\n\nKEY FINDINGS:")
    print("-" * 80)
    
    # Determine the strongest correlation
    correlations = {
        'Goals': (goals_corr, goals_pvalue),
        'Assists': (assists_corr, assists_pvalue),
        'Penalty Minutes': (pim_corr, pim_pvalue),
        'Points': (points_corr, points_pvalue)
    }
    
    strongest = max(correlations.items(), key=lambda x: abs(x[1][0]))
    
    if abs(strongest[1][0]) < 0.1 and strongest[1][1] > 0.05:
        print("• There is NO meaningful correlation between attractiveness and performance stats.")
        print("• Player attractiveness appears to be independent of scoring ability.")
    else:
        print(f"• Strongest correlation: {strongest[0]} (r = {strongest[1][0]:.4f}, p = {strongest[1][1]:.6f})")
        if strongest[1][0] > 0:
            print(f"• More attractive players tend to score slightly MORE {strongest[0].lower()}.")
        else:
            print(f"• More attractive players tend to score slightly FEWER {strongest[0].lower()}.")
    
    print("=" * 80)
