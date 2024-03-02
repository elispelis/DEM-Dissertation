from PIL import Image
import os


def create_gif(png_folder_path, output_gif_path, delay=1):
    # Get all PNG files in the folder
    png_files = [file for file in os.listdir(png_folder_path) if file.lower().endswith('.png')]

    # Sort the PNG files by name
    png_files.sort()

    # Create a list to store image objects
    images = []

    # Open and append each PNG file to the list
    for png_file in png_files:
        file_path = os.path.join(png_folder_path, png_file)
        img = Image.open(file_path)
        images.append(img)

    # Save the GIF with the specified delay between frames
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=delay * 1000, loop=0)


if __name__ == "__main__":
    # Set the input folder containing PNG files and the output GIF file
    folder =  r"V:\GrNN_EDEM-Sims\Rot_drum_400k_data\Export_Data\35_12_35_10_plots"
    input_folder = folder
    output_gif = rf"{folder}\output.gif"

    # Set the delay between frames in milliseconds
    delta_t = 0.05

    # Call the function to convert PNG files to GIF with specified delay
    create_gif(input_folder, output_gif, delta_t)

    print(f"GIF created: {output_gif}")
