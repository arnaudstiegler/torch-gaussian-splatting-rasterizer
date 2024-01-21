import os
import re

def create_ffmpeg_filelist(directory, output_filename="filelist.txt"):
    pattern = re.compile(r'image_iter_(\d+).png')
    files = [f for f in os.listdir(directory) if pattern.match(f)]
    
    # Extracting numbers and sorting
    files.sort(key=lambda x: int(pattern.search(x).group(1)))

    with open(os.path.join(directory, output_filename), 'w') as file:
        for filename in files:
            file.write(f"file '{filename}'\n")

# Usage
directory_path = '/home/arnaud/splat_images/'
create_ffmpeg_filelist(directory_path)
