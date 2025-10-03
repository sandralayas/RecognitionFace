import base64
import os

def encode_images_in_folder(input_folder, output_folder):
    """
    Takes all image files from a folder, encodes them into Base64 strings,
    and saves each string into a separate text file in an output folder.

    Args:
        input_folder (str): The path to the folder containing the images.
        output_folder (str): The path to the folder where the text files will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # List of common image file extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.tiff')

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)
    print(f"Found {len(files)} items in the input folder.")

    processed_count = 0
    for filename in files:
        # Construct the full path to the file
        file_path = os.path.join(input_folder, filename)

        # Check if the file is an image based on its extension
        if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
            try:
                # Read the image file in binary mode
                with open(file_path, "rb") as image_file:
                    # Encode the image data to base64
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

                # Create a new filename for the output text file
                # For example, "my_image.png" will become "my_image.txt"
                base_name = os.path.splitext(filename)[0]
                output_file_name = f"{base_name}.txt"
                output_path = os.path.join(output_folder, output_file_name)

                # Write the encoded string to the new text file
                with open(output_path, "w") as text_file:
                    text_file.write(encoded_string)

                print(f"Successfully converted and saved: {filename} -> {output_file_name}")
                processed_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nCompleted! Processed {processed_count} image(s) and saved them to the '{output_folder}' directory.")


# --- SCRIPT USAGE ---
# Define the input and output folders
input_folder_path = r"C:\Users\sandr\Documents\pix\FaceRec\corpus"  # Change this to your folder name
output_folder_path = r"C:\Users\sandr\Documents\database\base64imagecorpus"

# Run the function
encode_images_in_folder(input_folder_path, output_folder_path)
