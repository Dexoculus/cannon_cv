import cv2
import os

def video_to_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save each frame as an image file
        frame_filename = os.path.join(output_folder, f"image_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Video has been split into {frame_count} frames and saved in '{output_folder}'.")

if __name__ == "__main__":
    video_path = "./dataset/video/3.mp4"  # Replace with your video file path
    output_folder = "output_frames"  # Replace with your desired output folder
    video_to_frames(video_path, output_folder)