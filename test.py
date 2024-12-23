import os
from bing_image_downloader import downloader
from PIL import Image

# List of persons
persons = [
    "Robert Downey Jr", "Will Smith", "Tom Cruise", "Miu Shiromine",
    "Leonardo DiCaprio", "Brad Pitt", "Johnny Depp", "Dwayne Johnson"
]

# Download images for each person
for person in persons:
   
    # Find and download images of that person
    downloader.download(person, limit=5, output_dir='crawl_data', 
                        adult_filter_off=True, force_replace=False, timeout=60)
    

    person_path = os.path.join('crawl_data', person)

    # Rename downloaded images
    for idx, filename in enumerate(os.listdir(person_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            new_filename = f"{person.replace(" ","_")}_{idx+1}.jpg"
            os.rename(os.path.join(person_path, filename), os.path.join(person_path, new_filename))
            # Resize the image
            with Image.open(new_file_path) as img:
                img = img.resize((250, 250))
                img.save(new_file_path)