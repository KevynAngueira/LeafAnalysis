# Author: Kevyn Angueira Irizarry
# Created: 2025-03-26
# Last Modified: 2025-03-26



import os

def CreateLeafFolders(leaf_id: str, base_path: str):
    # Normalize and ensure we're under LeafMedia
    base_path = os.path.abspath(base_path)
    if os.path.basename(base_path) != "LeafMedia":
        base_path = os.path.join(base_path, "LeafMedia")

    # Conditions and subfolders
    conditions = ["healthy", "defoliated"]
    subfolders = ["images", "videos", "results"]

    for condition in conditions:
        for subfolder in subfolders:
            path = os.path.join(base_path, condition, leaf_id, subfolder)
            os.makedirs(path, exist_ok=True)

    print(f"Directory structure for leaf ID: {leaf_id}")


leaf_media_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/LeafMedia"
CreateLeafFolders("002", leaf_media_path)
