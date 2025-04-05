import kagglehub
import os

def loader():
    """
   Downloads the MBTI dataset from Kaggle via kagglehub.
    Returns the path to the unzipped folder.
    """
    # downloading process 
    path = kagglehub.dataset_download("mazlumi/mbti-personality-type-twitter-dataset")

    print("Dataset downloaded at path:", path)
    return path