import os
import cv2
import shutil
from glob import glob
from tqdm import tqdm
import albumentations as A
import sys

# Augmentation Pipeline
augmenter = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(p=0.7),
        A.CLAHE(p=0.7),
        A.HueSaturationValue(p=0.7)
    ], p=0.8),

    A.OneOf([
        A.MotionBlur(p=0.5),
        A.MedianBlur(blur_limit=3, p=0.5),
        A.GaussianBlur(p=0.5),
    ], p=0.3),

    A.OneOf([
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.9),
        A.Affine(shear=5, p=0.5)
    ], p=0.8),

    A.RandomShadow(p=0.3),
    A.RandomFog(p=0.2),
    A.CoarseDropout(max_height=32, max_width=32, max_holes=1, p=0.3),
])

def augment_and_save(input_dir, output_dir, augmentations_per_image=3, image_exts=(".jpg", ".png", ".jpeg")):
    """
    Augments images from input_dir and saves them to output_dir with sequential numbering.

    Args:
        input_dir (str): Path to the input directory containing subdirectories for each class.
        output_dir (str): Path to the output directory where augmented images will be saved.
        augmentations_per_image (int): Number of augmented versions to create for each original image.
        image_exts (tuple): Tuple of image file extensions to look for.

    Returns:
        int: The total number of images (originals + augmentations) successfully saved.
        Returns -1 if the input directory is not found or is empty.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return -1 # Indicate error

    os.makedirs(output_dir, exist_ok=True)
    image_paths = [f for ext in image_exts for f in glob(os.path.join(input_dir, "*", f"*{ext}"))]

    if not image_paths:
        print(f"Warning: No images found in {input_dir} with extensions {image_exts}", file=sys.stderr)
        return 0 # No images processed

    # Initialize a counter for sequential numbering within this function call
    image_counter = 1
    num_digits = 6 # Adjust if you expect more than 999,999 total images
    successfully_saved_count = 0

    for img_path in tqdm(image_paths, desc="Augmenting images"):
        original_output_path = None # Initialize for error handling scope
        try:
            label = os.path.basename(os.path.dirname(img_path))
            label_output_dir = os.path.join(output_dir, label)
            os.makedirs(label_output_dir, exist_ok=True)

            # --- Save the original image with sequential numbering ---
            original_output_filename = f"augmented_{image_counter:0{num_digits}d}.jpg"
            original_output_path = os.path.join(label_output_dir, original_output_filename)
            shutil.copy2(img_path, original_output_path)
            successfully_saved_count += 1
            current_original_counter = image_counter # Store counter for this original
            image_counter += 1

            # --- Generate and save augmentations ---
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Failed to read {img_path}. Skipping augmentations for this image.")
                # Note: Original was already copied, we keep it but log the warning.
                # If you want to remove the original on read failure, uncomment below:
                # if os.path.exists(original_output_path):
                #     os.remove(original_output_path)
                #     successfully_saved_count -= 1 # Decrement if removing
                #     image_counter = current_original_counter # Reset counter partially
                continue # Skip augmentation part

            for i in range(augmentations_per_image):
                try:
                    augmented = augmenter(image=image)
                    aug_image = augmented["image"]

                    aug_filename = f"augmented_{image_counter:0{num_digits}d}.jpg"
                    aug_path = os.path.join(label_output_dir, aug_filename)

                    save_success = cv2.imwrite(aug_path, aug_image)
                    if not save_success:
                         print(f"Warning: Failed to write augmented image {aug_path}. Skipping this augmentation.")
                         # Don't increment counter if save failed
                         continue # Move to next augmentation attempt or next image

                    successfully_saved_count += 1
                    image_counter += 1 # Increment counter only after successful save

                except Exception as aug_e:
                    # Handle errors during augmentation or saving of a single augmented image
                    print(f"Error augmenting/saving iteration {i+1} for {img_path}: {aug_e}. Skipping this augmentation.")
                    # Counter doesn't increment here


        except FileNotFoundError:
             print(f"Error: Original image not found during processing (might have been deleted?): {img_path}", file=sys.stderr)
             # Adjust counts if original was counted but couldn't be processed
             if original_output_path and os.path.exists(original_output_path):
                 successfully_saved_count -= 1
                 image_counter = current_original_counter # Reset counter fully if original failed before augmentations started
        except Exception as e:
            print(f"Error processing {img_path}: {e}", file=sys.stderr)
            # Attempt to clean up potentially partially saved files for this image is complex.
            # It's safer to just log the error and continue. The count might be slightly off
            # if an error happens *after* saving the original but *before* all augmentations.
            # The current logic tries to handle read failures more gracefully.

    # Return the total count of successfully saved images
    return successfully_saved_count

# --- Example Usage ---
input_directory = "./dataset"
output_directory = "./augmented"
augmentations_count = 3

try:
    # Call the function and store the returned count
    total_generated = augment_and_save(
        input_dir=input_directory,
        output_dir=output_directory,
        augmentations_per_image=augmentations_count
    )

    # Check the return value for errors before printing success messages
    if total_generated >= 0:
        print("-" * 30)
        print(f"Augmentation process finished.")
        print(f"Output saved to: {os.path.abspath(output_directory)}")
        print(f"Total images successfully generated (originals + augmentations): {total_generated}")
        print("-" * 30)
    else:
        # Error message was already printed inside the function
        print("Augmentation process failed. Please check the error messages above.", file=sys.stderr)


except Exception as e:
    # Catch any unexpected errors during the function call itself
    print(f"An unexpected error occurred during the augmentation process: {e}", file=sys.stderr)