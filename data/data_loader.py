import kagglehub

def loader():
    """
   Downloads the MBTI dataset from Kaggle via kagglehub.
    Returns the path to the unzipped folder.
    """
    # downloading process 
    path = kagglehub.dataset_download("praveengovi/emotions-dataset-for-nlp")

    print("Dataset downloaded at path:", path)
    return path