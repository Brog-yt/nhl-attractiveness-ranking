from nhle_github import NhleGithub
from face_processer import FaceProcesser

if __name__ == "__main__":
    # Get all the players on the canucks
    nhle = NhleGithub()
    simplified_players = nhle.get_simplifiedPlayers("VAN")
    
    # With the headshot URL, get the face embedding using FaceProcesser
    processor = FaceProcesser() 
    
    for player in simplified_players:
        try:
            print(f"Processing {player.firstName.default} {player.lastName.default}...")
            embedding = processor.get_embedding_from_url(player.headshot)
            print(f"  First 10 values: {embedding[:10]}")
        except Exception as e:
            print(f"  Error: {e}")