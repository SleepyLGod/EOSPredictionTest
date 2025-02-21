import os
from PIL import Image

prompt_num = 3

folder_path = f'../images/maps_final/prompt_full_s_{prompt_num}/'
output_path = f'../images/maps_final/{prompt_num}_combined_heatmaps.png'

# get the files with the beginning of the name 'heatmap_layer_' and the ending of the name '.png'
files = [f for f in os.listdir(folder_path) if f.startswith('heatmap_layer_') and f.endswith('.png')]

# initialize the number of layers and heads
num_layers = 15
num_heads = 31

# ceate a dictionary to store the images, with the key being a tuple of (layer, head)
images = {}

for file in files:
    # from the file name, extract the layer and head number
    parts = file.split('_')
    layer = int(parts[2])
    head = int(parts[4].replace('.png', ''))
    
    # check if the layer and head number are within the expected range
    if 0 <= layer <= num_layers and 0 <= head <= num_heads:
        images[(layer, head)] = os.path.join(folder_path, file)
    else:
        print(f"Skipping file {file} as layer or head number is out of range.")

# open a sample image to get the dimensions
sample_image_path = next(iter(images.values()))
with Image.open(sample_image_path) as img:
    img_width, img_height = img.size

# create a new image to combine all the images
total_width = img_width * num_heads
total_height = img_height * num_layers
combined_image = Image.new('RGB', (total_width, total_height))

for layer in range(1, num_layers + 1):
    for head in range(1, num_heads + 1):
        key = (layer, head)
        if key in images:
            img_path = images[key]
            with Image.open(img_path) as img:
                # create the combined image by pasting each image at the right position
                x = (head - 1) * img_width
                y = (layer - 1) * img_height
                combined_image.paste(img, (int(x), int(y)))
        else:
            # if the image is missing, create a blank image
            blank_image = Image.new('RGB', (img_width, img_height))
            x = (head - 1) * img_width
            y = (layer - 1) * img_height
            combined_image.paste(blank_image, (int(x), int(y)))
            print(f"Missing image for layer {layer}, head {head}.")

combined_image.save(output_path)
print(f"Combined image saved to {output_path}")