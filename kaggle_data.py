import kagglehub
from pathlib import Path
import pandas as pd


class KaggleData:
    """Handle Kaggle dataset downloads and data preparation"""
    
    def getSCUTData(self) -> pd.DataFrame:
        """
        Download and process the SCUT-FBP5500-v2 facial beauty dataset.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['image', 'score', 'path']
                - image: filename
                - score: beauty score (float)
                - path: full path to image file
        """
        # Download dataset
        root = Path(kagglehub.dataset_download(
            "pranavchandane/scut-fbp5500-v2-facial-beauty-scores"
        ))
        
        labels_path = root / "labels.txt"
        images_dir = root / "Images" / "Images"
        
        # Parse labels file
        data = []
        with open(labels_path) as f:
            for line in f:
                fname, score = line.strip().split()
                data.append((fname, float(score)))
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=["image", "score"])
        df["path"] = df["image"].apply(lambda x: str(images_dir / x))
        
        return df
