import cv2
import os

def images_to_video(image_folder, output_video, frame_duration=0.15, fps=None):
    # Get the list of PNG files in the folder
    images = [img for img in sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x)))) if img.endswith(".png")]    
    
    if not images:
        print("No PNG files found in the folder.")
        return

    # Read the first image to get the size
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Calculate FPS if not provided (fps = 1 / frame_duration)
    fps = fps or int(1 / frame_duration)
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        print(f"Writing {img_path} to video...")
        img = cv2.imread(img_path)
        video.write(img)

    # Release the video writer
    video.release()

    print(f"Video saved as {output_video}")

# Example usage:
image_folder = 'output/saved_images'
output_video = 'output_video.mp4'
images_to_video(image_folder, output_video)
