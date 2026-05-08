from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="model_a/forensic_cnn.pth",
    path_in_repo="forensic_cnn.pth",
    repo_id="parina13/synthsenses-models",
    repo_type="model",
)
print("forensic_cnn.pth uploaded")

api.upload_file(
    path_or_fileobj="model_b/model.pkl",
    path_in_repo="model.pkl",
    repo_id="parina13/synthsenses-models",
    repo_type="model",
)
print("model.pkl uploaded")
