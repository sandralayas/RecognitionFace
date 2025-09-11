import os
import shutil

corpus = r"C:\Users\sandr\Documents\git\insightface\finetuning\validation"

listall = []
count = 0

for photos in os.listdir(corpus):
    try:
        # Check if the item is a file before trying to process it.
        # This prevents errors if there are subdirectories in the corpus.
        if os.path.isfile(os.path.join(corpus, photos)):
            # The '[:-1]' slice was likely intended to remove the last character (the age).
            # This is a bit fragile; a better approach is to use the actual file name without the extension.
            # A common file naming convention is 'name_age.jpg'. Let's assume that.
            name_part = photos.split('.')[0]
            
            # Extract the base name (e.g., 'person1') from 'person1_25A'.
            # This is a bit of an assumption, but based on your original attempt,
            # you were trying to group by the name part before the age.
            # Let's assume the age is a number at the end, and we want to group by the prefix.
            # This is a more robust way to handle it.
            name = ''.join(filter(str.isalpha, name_part))
            
            if name not in listall:
                listall.append(name)
                count += 1
                
                # Create the destination folder path
                dest_folder = os.path.join(corpus, str(count))
                
                # Create the directory. exist_ok=True prevents an error if the directory already exists.
                os.makedirs(dest_folder, exist_ok=True)
                
                # Move the file to the newly created folder
                shutil.move(os.path.join(corpus, photos), os.path.join(dest_folder, photos))
            else:
                # Find the existing folder number for the current name
                folder_index = listall.index(name) + 1
                dest_folder = os.path.join(corpus, str(folder_index))
                
                # Move the file to the existing folder
                shutil.move(os.path.join(corpus, photos), os.path.join(dest_folder, photos))
                
    except Exception as e:
        print(f"Error processing {photos}: {e}")