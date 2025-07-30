import cv2
import numpy as np

def extract_first_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if not ret:
        raise ValueError("Could not read the first frame from the video.")
    
    return frame

def calculate_psnr(reference_image, test_image):
    # Convert images to float32
    reference_image = reference_image.astype(np.float32)
    test_image = test_image.astype(np.float32)
    
    # Calculate MSE
    mse = np.mean((reference_image - test_image) ** 2)
    
    if mse == 0:
        return float('inf')  # PSNR is infinite if images are identical
    
    # Calculate PSNR
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

# Example usage
if __name__ == "__main__":
    video_path = 'path_to_your_video.mp4'  # Replace with your video file path
    reference_image_path = 'path_to_your_reference_image.png'  # Replace with your reference image path
    
    # Extract the first frame from the video
    first_frame = extract_first_frame(video_path)
    
    # Load the reference image
    reference_image = cv2.imread(reference_image_path)
    
    # Calculate PSNR
    psnr_value = calculate_psnr(reference_image, first_frame)
    
    print(f"PSNR between the first frame and the reference image: {psnr_value:.2f} dB")