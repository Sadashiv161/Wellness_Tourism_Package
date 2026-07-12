from huggingface_hub import HfApi
import os

token = os.getenv("HF_TOKEN")  # please use your token

api = HfApi(token=token)

api.upload_folder(
    folder_path="deployment",                        # the local folder containing your files
    repo_id="sadashivbhatt/Wellness_Tourism_Package",  # the target repo
    repo_type="space",                                # dataset, model, or space
    path_in_repo="",                                  # optional: subfolder path inside the repo
)
