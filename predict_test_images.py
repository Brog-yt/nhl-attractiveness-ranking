"""
Script to predict attractiveness scores for test images (biz.jpg, whit.jpg)
and rank them against existing NHL players from attractive_players_with_stats.json
"""

import json
from pathlib import Path
import joblib
from face_processer import FaceProcesser
from models import SimpleSpecificPlayerData

# SVR Model and Scaler paths
CACHE_DIR = Path("cached-models")
MODEL_FILE = CACHE_DIR / "beauty_score_model_male.pkl"
SCALER_FILE = CACHE_DIR / "beauty_score_model_male_scaler.pkl"

def load_svr_model_and_scaler():
    """Load the optimized SVR model and scaler"""
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        raise FileNotFoundError(
            "SVR model or scaler not found. Please run ridge-regression-script.py first."
        )
    
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler


def load_nhl_players():
    """Load NHL players with stats from attractive_players_with_stats.json"""
    data_file = Path(__file__).parent / "players" / "attractive_players_with_stats.json"
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Cast each dict to SimpleSpecificPlayerData
    players = [
        SimpleSpecificPlayerData(**player_dict) for player_dict in data
    ]
    
    return players


def predict_image_attractiveness(image_path, model, scaler):
    """
    Predict attractiveness score for an image using SVR model
    
    Args:
        image_path: Path to the image file
        model: Trained SVR model
        scaler: StandardScaler for preprocessing
    
    Returns:
        float: Predicted attractiveness score (1-5 scale)
    """
    processor = FaceProcesser()
    
    try:
        # Get embedding from image path
        embedding = processor.get_embedding_from_path(image_path)
        
        # Scale and predict
        embedding_scaled = scaler.transform(embedding.reshape(1, -1))
        score = model.predict(embedding_scaled)[0]
        
        return score
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


if __name__ == "__main__":
    # Load SVR model and scaler
    print("Loading optimized SVR model and scaler...")
    model, scaler = load_svr_model_and_scaler()
    print("SVR model loaded successfully!\n")
    
    # Load NHL players
    print("Loading NHL players from attractive_players_with_stats.json...")
    nhl_players = load_nhl_players()
    print(f"Loaded {len(nhl_players)} NHL players\n")
    
    # Test images
    test_images = [
        ("biz.jpg", "Biz"),
        ("whit.jpg", "Whit")
    ]
    
    # Get attractiveness scores for test images
    test_scores = {}
    print("=" * 80)
    print("PREDICTING ATTRACTIVENESS FOR TEST IMAGES")
    print("=" * 80)
    
    for image_path, name in test_images:
        full_path = Path(__file__).parent / image_path
        
        if not full_path.exists():
            print(f"\n❌ {name} ({image_path}): File not found")
            continue
        
        print(f"\nProcessing {name} ({image_path})...")
        score = predict_image_attractiveness(full_path, model, scaler)
        
        if score is not None:
            test_scores[name] = score
            print(f"✓ Predicted attractiveness score: {score:.4f} / 5.0")
        else:
            print(f"✗ Failed to predict score for {name}")
    
    print("\n")
    
    # Get NHL player scores for ranking
    nhl_scores = []
    for player in nhl_players:
        nhl_scores.append({
            'name': f"{player.player.firstName.default} {player.player.lastName.default}",
            'score': player.ridgeAttractivenessScore,
            'is_test': False
        })
    
    # Add test images to the ranking
    for name, score in test_scores.items():
        nhl_scores.append({
            'name': name,
            'score': score,
            'is_test': True
        })
    
    # Sort by score descending
    nhl_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Find ranks of test images
    print("=" * 80)
    print("TEST IMAGE RANKINGS")
    print("=" * 80)
    
    total_players = len(nhl_scores)
    
    for i, player in enumerate(nhl_scores, 1):
        if player['is_test']:
            percentile = (i / total_players) * 100
            print(f"\n🎯 {player['name']}")
            print(f"   Score: {player['score']:.4f} / 5.0")
            print(f"   Rank: #{i} out of {total_players}")
            print(f"   Percentile: {percentile:.1f}% (Top {100-percentile:.1f}%)")
            
            # Find how many NHL players are more attractive
            more_attractive = i - 1
            less_attractive = total_players - i
            
            if more_attractive > 0:
                print(f"   More attractive than: {less_attractive} NHL players")
            if less_attractive > 0:
                print(f"   Less attractive than: {more_attractive} NHL players")
    
    print("\n" + "=" * 80)
    print("TOP 15 RANKINGS (INCLUDING TEST IMAGES)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Name':<30} {'Score':<12} {'Type'}")
    print("-" * 80)
    
    for i, player in enumerate(nhl_scores[:15], 1):
        player_type = "📸 TEST" if player['is_test'] else "NHL"
        print(f"{i:<6} {player['name']:<30} {player['score']:<12.4f} {player_type}")
    
    print("=" * 80)
