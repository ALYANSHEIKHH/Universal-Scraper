import os

def create_folder(path):
    os.makedirs(path, exist_ok=True)

def get_folder_path(label):
    return os.path.join("data", label)