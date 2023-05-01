import tarfile

tar_path = "sci-simplify_backend/sentence-encoder/universal-sentence-encoder_4.tar.gz"
output_path = "sci-simplify_backend/sentence-encoder"

with tarfile.open(tar_path, "r") as tar:
    tar.extractall(output_path)
