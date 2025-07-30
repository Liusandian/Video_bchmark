import json
import numpy as np
import matplotlib.pyplot as plt

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Plot the pose information
def plot_pose(data):
    # Extract pitch, yaw, roll from the data
    pitch = [frame['pitch'] for frame in data]
    yaw = [frame['yaw'] for frame in data]
    roll = [frame['roll'] for frame in data]

    # Calculate statistics
    pitch_mean, pitch_std = np.mean(pitch), np.std(pitch)
    yaw_mean, yaw_std = np.mean(yaw), np.std(yaw)
    roll_mean, roll_std = np.mean(roll), np.std(roll)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(pitch, label='Pitch', color='r')
    plt.plot(yaw, label='Yaw', color='g')
    plt.plot(roll, label='Roll', color='b')

    # Add statistics to the plot
    plt.text(0.05, 0.95, f'Pitch: μ={pitch_mean:.2f}, σ={pitch_std:.2f}\n'
                         f'Yaw: μ={yaw_mean:.2f}, σ={yaw_std:.2f}\n'
                         f'Roll: μ={roll_mean:.2f}, σ={roll_std:.2f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.title('Pose Angles Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()
    
    # Save the plot
    plt.savefig('pose_angles_plot.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    json_file_path = 'path_to_your_json_file.json'  # Replace with your JSON file path
    pose_data = load_json_data(json_file_path)
    plot_pose(pose_data)