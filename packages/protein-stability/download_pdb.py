import os

from google.cloud import storage


def download_gcs_bucket_folder(bucket_name, bucket_folder, local_dir):
    """
    Downloads all files and folders from a GCS bucket folder to a local directory.

    :param bucket_name: Name of the GCS bucket
    :param bucket_folder: Path to the folder in the bucket to download
    :param local_dir: Path to the local directory where files will be saved
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=bucket_folder)  # List everything under the folder
    for blob in blobs:
        # Create the local path
        relative_path = os.path.relpath(blob.name, bucket_folder)
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {blob.name} to {local_path}")
        blob.download_to_filename(local_path)


# Define your variables
bucket_name = "proteincrow"  # Name of the bucket
bucket_folder = "datasets/full_stability_dataset_megascale_with_pdb_ids"  # Path to the folder in the bucket
local_dir = "./protein_stability/data"

# Call the function to download
download_gcs_bucket_folder(bucket_name, bucket_folder, local_dir)
