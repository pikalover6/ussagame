from PIL import Image

# Load the image
image = Image.open("texture_5.jpg")

# Mirror the image horizontally
mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)

# Save the mirrored image
mirrored_image.save("texture_5_mirrored.jpg")

print("Mirrored image saved as texture_5_mirrored.jpg")
