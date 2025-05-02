import os
import random
import shutil

def sample_images(source_folder, destination_folder, n):
    """
    Samples n images from the source folder without replacement and copies them to the destination folder.

    :param source_folder: Path to the folder containing the images.
    :param destination_folder: Path to the folder where sampled images will be copied.
    :param n: Number of images to sample.
    """
    # Ensure source folder exists
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder '{source_folder}' does not exist.")

    # Ensure destination folder exists, create if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get list of all files in the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Ensure there are enough images to sample
    if n > len(all_files):
        raise ValueError(f"Not enough images in the source folder to sample {n} images.")

    # Randomly sample n images without replacement
    sampled_files = random.sample(all_files, n)

    # Copy sampled images to the destination folder
    for file_name in sampled_files:
        src_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(destination_folder, file_name)
        shutil.copy(src_path, dest_path)

    print(f"Successfully sampled {n} images to '{destination_folder}'.")

# Example usage
if __name__ == "__main__":
    source_folder = "./dataset/Non"
    destination_folder = "./dataset/sampled_Non"
    n = 200  # Number of images to sample

    try:
        sample_images(source_folder, destination_folder, n)
    except Exception as e:
        print(f"Error: {e}")