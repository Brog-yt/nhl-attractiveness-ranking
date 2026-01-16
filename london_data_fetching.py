

import pandas as pd
from pathlib import Path


class LondonDataFetching:

    # Read from the london-data/london_faces_ratings.csv file
    # The photo name (ex: X001) will be a column
    # All the scores in that column will be averaged to get the final score for that photo
    # Scale scores from 1-7 range to 1-5 range for consistency with SCUT data
    def process_csv(self) -> dict:
        """
        Read london_faces_ratings.csv and calculate average score for each photo.
        Ignores the first 3 columns (rater_sex, rater_sexpref, rater_age) and processes photo columns only.
        Scales scores from 1-7 to 1-5 range.
        
        Returns:
            dict: Mapping of photo name (ex: "X001") to average score (scaled to 1-5)
        """
        csv_path = Path("london-data/london_faces_ratings.csv")
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Skip the first 3 metadata columns (rater_sex, rater_sexpref, rater_age)
        # Process only photo columns (X001, X002, etc.)
        photo_scores = {}
        
        for column in df.columns[3:]:  # Start from column 4 (index 3)
            if column.startswith('X'):  # Only process columns starting with 'X'
                photo_name = column
                scores = pd.to_numeric(df[column], errors='coerce').dropna()
                if len(scores) > 0:
                    avg_score = scores.mean()
                    # Scale from 1-7 to 1-5: (score - 1) * (5 - 1) / (7 - 1) + 1
                    scaled_score = (avg_score - 1) * 4 / 6 + 1
                    photo_scores[photo_name] = scaled_score
        
        return photo_scores

    # Each photo is stored in london-data/neutral-front/{photo_name}_03.jpg
    # For example, photo X001 is stored in london-data/neutral-front/X001_03.jpg
    # Return a DataFrame with columns ['image', 'score', 'path'] matching KaggleData format
    def get_london_data(self) -> pd.DataFrame:
        """
        Get London dataset as a DataFrame matching the KaggleData format.
        Returns columns: ['image', 'score', 'path'] with scores scaled from 1-7 to 1-5.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['image', 'score', 'path']
                - image: photo name (ex: "X001")
                - score: beauty score (1-5 scale)
                - path: full path to image file
        """
        # Get photo name to score mapping
        photo_scores = self.process_csv()
        
        # Convert to DataFrame format
        data = []
        for photo_name, score in photo_scores.items():
            photo_path = f"london-data/neutral-front/{photo_name}_03.jpg"
            data.append((photo_name, score, photo_path))
        
        df = pd.DataFrame(data, columns=["image", "score", "path"])
        return df
    
    
    


