import kagglehub
from pathlib import Path
import pandas as pd


class KaggleData:
    """Handle Kaggle dataset downloads and data preparation"""
    
    def getSCUTData(self, gender: str = None) -> pd.DataFrame:
        """
        Download and process the SCUT-FBP5500-v2 facial beauty dataset.
        
        Args:
            gender: Optional filter - 'male', 'female', or None for all
                   SCUT filenames: AM=Asian Male, CM=Caucasian Male, 
                                   AF=Asian Female, CF=Caucasian Female
        
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
                
                # Apply gender filter if specified
                if gender == 'male' and not (fname.startswith('AM') or fname.startswith('CM')):
                    continue
                elif gender == 'female' and not (fname.startswith('AF') or fname.startswith('CF')):
                    continue
                
                data.append((fname, float(score)))
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=["image", "score"])
        df["path"] = df["image"].apply(lambda x: str(images_dir / x))
        
        return df
