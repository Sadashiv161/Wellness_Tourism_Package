
import pandas as pd
import sklearn
import os


from huggingface_hub import HfApi



# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi



# Define constants for the dataset and output paths


token = os.getenv("HF_TOKEN")

api = HfApi(token=os.getenv("HF_TOKEN"))

# please create your dataset as you create your space
DATASET_PATH = "hf://datasets/sadashivbhatt/Wellness_Tourism_Package/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier and
# Split into X (features) and y (target)
target_col = "ProdTaken"
X = df.drop(columns=["ProdTaken", "CustomerID"])
y = df["ProdTaken"]


#### Create Dummies which behaviour is categorical but in numerical

categorical_features = [
    "TypeofContact",
    "CityTier",                  # 1, 2, 3 are categories
    "Occupation",
    "Gender",
    "ProductPitched",
    "PreferredPropertyStar",     # 3, 4, 5 stars are categories
    "MaritalStatus",
    "Passport",                  # 0/1 (binary category)
    "PitchSatisfactionScore",    # Rating (1–5)
    "OwnCar",                    # 0/1 (binary category)
    "Designation"
]

numerical_features = [
    "Age",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "NumberOfTrips",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename

        repo_id="sadashivbhatt/Wellness_Tourism_Package",

        repo_type="dataset",
    )
