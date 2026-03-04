import requests
import json
import random
from pathlib import Path
from nhle_github import NhleGithub, allActiveTeams

# Configuration
HEADSHOTS_DIR = Path("headshots")
HEADSHOTS_DIR.mkdir(exist_ok=True)

# Initialize NHL API
print("Fetching NHL player data...")
nhle = NhleGithub()

# Collect all players from all teams
all_players = []
for team_code in allActiveTeams:
    try:
        simple_players = nhle.get_simplifiedPlayers(team_code)
        all_players.extend(simple_players)
        print(f"  {team_code}: {len(simple_players)} players")
    except Exception as e:
        print(f"  {team_code}: Error - {e}")

print(f"\nTotal players found: {len(all_players)}")

# Sample 100 random players
sample_size = min(100, len(all_players))
sampled_players = random.sample(all_players, sample_size)

print(f"Downloading {len(sampled_players)} random player headshots...\n")

downloaded = 0
failed = 0
skipped = 0

for i, player in enumerate(sampled_players, 1):
    first_name = player.firstName.get('default', player.firstName) if hasattr(player.firstName, 'get') else player.firstName
    last_name = player.lastName.get('default', player.lastName) if hasattr(player.lastName, 'get') else player.lastName
    headshot_url = player.headshot
    player_id = player.id
    
    if not headshot_url or not player_id:
        print(f"[{i}/{len(sampled_players)}] Skipping {first_name} {last_name} - missing data")
        skipped += 1
        continue
    
    # Create filename based on player ID
    filename = HEADSHOTS_DIR / f"{player_id}.png"
    
    # Skip if already downloaded
    if filename.exists():
        print(f"[{i}/{len(sampled_players)}] Already have {first_name} {last_name}")
        skipped += 1
        continue
    
    # Download headshot
    try:
        response = requests.get(headshot_url, timeout=10)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"[{i}/{len(sampled_players)}] ✓ Downloaded {first_name} {last_name}")
        downloaded += 1
    except Exception as e:
        print(f"[{i}/{len(sampled_players)}] ✗ Failed to download {first_name} {last_name}: {e}")
        failed += 1

print(f"\n{'='*60}")
print(f"Download complete!")
print(f"  Downloaded: {downloaded}")
print(f"  Skipped (already exist): {skipped}")
print(f"  Failed: {failed}")
print(f"  Total in headshots folder: {len(list(HEADSHOTS_DIR.glob('*.png')))}")
print(f"{'='*60}")
