import os
import subprocess

# Set the paths to the COLMAP executable, project file, and output directory
colmap_executable = "COLMAP-3.8-windows-cuda\COLMAP.bat"
project_path = "./valve_handle/colmap/sparse/0/"
output_path = "./"

# Create the output directory if it doesnt exist
os.makedirs(output_path, exist_ok=True)

# Run COLMAP command to extract camera poses, point cloud, and used images
camera_poses_path = os.path.join(output_path, "camera_poses.txt")
point_cloud_path = os.path.join(output_path, "point_cloud.ply")
used_images_path = os.path.join(output_path, "used_images.txt")

# Extract camera poses
os.system(f"{colmap_executable} model_converter --input_path {project_path} --output_path {camera_poses_path} --output_type CAM")

# Extract point cloud
os.system(f"{colmap_executable} model_converter --input_path {project_path} --output_path {point_cloud_path} --output_type ply")

# Extract used images
os.system(f"{colmap_executable} model_converter --input_path {project_path} --output_path {used_images_path} --output_type txt")

#print(Extraction completed successfully.)
