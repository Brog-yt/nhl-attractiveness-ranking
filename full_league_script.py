
import joblib
import json
from pathlib import Path
from typing import List
from face_processer import FaceProcesser
from nhle_github import NhleGithub, allActiveTeams
from models import PlayerAttractiveAnalysis, SimplePlayer

# Use the male-only trained model for NHL players (SVR with GridSearchCV optimization)
CACHE_DIR = Path("cached-models")
MODEL_FILE = CACHE_DIR / "beauty_score_model_male.pkl"
SCALER_FILE = CACHE_DIR / "beauty_score_model_male_scaler.pkl"

def main():
    # Step 1: Check if the model and scaler exist
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. "
            "Please run ridge-regression-script.py first to generate the model."
        )
    
    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_FILE}. "
            "Please run ridge-regression-script.py first to generate the scaler."
        )
    
    # Load the trained SVR model and scaler
    print(f"Loading SVR model from {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)
    print("SVR model loaded successfully!")
    
    print(f"Loading scaler from {SCALER_FILE}...")
    scaler = joblib.load(SCALER_FILE)
    print("Scaler loaded successfully!\n")
    
    # Initialize FaceProcesser and NhleGithub
    processor = FaceProcesser()
    nhle = NhleGithub()
    
    # Step 2 & 3: Get all players from all teams
    print(f"Fetching players from all {len(allActiveTeams)} NHL teams...")
    all_players: List[SimplePlayer] = []
    
    for team_code in allActiveTeams:
        try:
            print(f"  Fetching {team_code}...")
            team_players = nhle.get_simplifiedPlayers(team_code)
            # Filter out players without headshots
            team_players = [p for p in team_players if p.headshot and p.headshot.strip()]
            all_players.extend(team_players)
        except Exception as e:
            print(f"  Error fetching {team_code}: {e}")
    
    print(f"\nTotal players fetched: {len(all_players)}\n")
    
    # Step 4: Process each player and predict attractiveness
    player_analyses: List[PlayerAttractiveAnalysis] = []
    processing_errors = []
    
    print("Processing player headshots and predicting attractiveness scores...")
    print(f"Using optimized SVR model (Test MSE: 0.0958)\n")
    for i, player in enumerate(all_players):
        try:
            # Get embedding from headshot URL
            embedding = processor.get_embedding_from_url(player.headshot)
            
            # Scale the embedding and predict attractiveness score
            embedding_scaled = scaler.transform(embedding.reshape(1, -1))
            score = model.predict(embedding_scaled)[0]
            
            # Create PlayerAttractiveAnalysis object (rank will be set after sorting)
            analysis = PlayerAttractiveAnalysis(
                rank=0,  # Placeholder, will be updated after sorting
                player=player,
                ridgeAttractivenessScore=float(score)
            )
            player_analyses.append(analysis)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(all_players)} players")
                
        except Exception as e:
            error_info = {
                "playerId": player.id,
                "firstName": player.firstName.default,
                "lastName": player.lastName.default,
                "headshot": player.headshot,
                "error": str(e),
                "errorType": type(e).__name__
            }
            processing_errors.append(error_info)
            print(f"  Error processing {player.firstName.default} {player.lastName.default}: {e}")
    
    print(f"\nSuccessfully processed {len(player_analyses)} players\n")
    
    # Step 5: Sort players by attractiveness (descending)
    player_analyses.sort(key=lambda x: x.ridgeAttractivenessScore, reverse=True)
    
    # Assign ranks after sorting
    for i, analysis in enumerate(player_analyses, 1):
        analysis.rank = i
    
    # Step 6: Print top 10 most attractive players
    print("="*60)
    print("TOP 10 MOST ATTRACTIVE NHL PLAYERS")
    print("(SVR Model - Typical Error: ±0.31 points)")
    print("="*60)
    for i, analysis in enumerate(player_analyses[:10], 1):
        player = analysis.player
        score = analysis.ridgeAttractivenessScore
        print(f"{i:2d}. {player.firstName.default} {player.lastName.default:20s} - Score: {score:.4f}")
    print("="*60)
    print()
    
    # Step 7: Write full list to JSON file
    output_dir = Path("players")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "attractiveness_analysis.json"
    
    # Convert to JSON-serializable format with simplified names
    output_data = []
    for analysis in player_analyses:
        data = {
            "rank": analysis.rank,
            "player": {
                "id": analysis.player.id,
                "headshot": analysis.player.headshot,
                "firstName": analysis.player.firstName.default,
                "lastName": analysis.player.lastName.default
            },
            "ridgeAttractivenessScore": analysis.ridgeAttractivenessScore
        }
        output_data.append(data)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Full analysis saved to: {output_file}")
    print(f"Total players analyzed: {len(player_analyses)}")
    
    # Write processing errors to JSON file
    if processing_errors:
        errors_file = output_dir / "processing-errors.json"
        with open(errors_file, "w", encoding="utf-8") as f:
            json.dump(processing_errors, f, indent=2, ensure_ascii=False)
        print(f"Processing errors saved to: {errors_file}")
        print(f"Total errors: {len(processing_errors)}")

if __name__ == "__main__":
    main()

