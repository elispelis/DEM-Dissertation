import os
from PIL import Image

def crop_images_in_subfolder(folder):
    # Iterate through each subfolder
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        # Output folder for cropped images
        output_folder = os.path.join(subfolder_path, "cropped")
        
        # Create the output folder if it does not exist
        os.makedirs(output_folder, exist_ok=True)


        for filename in os.listdir(subfolder_path):
            if filename.endswith(".png"):  # Process only PNG files
                # Open the image
                im = Image.open(os.path.join(subfolder_path, filename))

                # Calculate cropping coordinates
                width, height = im.size
                left = (width - 500) / 2
                top = (height - 350) / 2
                right = (width + 500) / 2
                bottom = (height + 350) / 2

                # Crop the image
                im_cropped = im.crop((left, top, right, bottom))

                # Save the cropped image to the output folder
                cropped_filename = os.path.join(output_folder, filename)
                im_cropped.save(cropped_filename, format="PNG")

        print("Cropping complete for subfolder:", subfolder)

# Master folder containing nested folders with images
master_folder = folder = rf"V:\GrNN_EDEM-Sims\Rot_drum_400k_data\Export_Data\RNNSR_plots"


# Call the function to crop images in each folder
crop_images_in_subfolder(master_folder)
