from PIL import Image
import os


def clean_image_data(final_size, image):
    """Function to normalised the size and mode of the image to the input final size and RGB mode.

    Args:
        final_size (int): integer value of the final pixel size of the new image.
        image (image): image object of desired image to normalize.

    Returns:
        image: image object of original image to formated to desired pixel size in RGB mode.
    """
    size = image.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])

    image = image.resize(new_image_size, Image.ANTIALIAS)
    new_image = Image.new("RGB", (final_size, final_size))
    new_image.paste(image, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    
    return new_image


if __name__ == '__main__':
    path = "../images/"
    new_path = "../cleaned_images/"
    dirs = os.listdir(path)
    final_size = 64

    if not os.path.exists(new_path):
        os.mkdir(new_path)
    
    for n, item in enumerate(dirs[:5], 1):
        image = Image.open(path + item)
        new_image = clean_image_data(final_size, image)
        new_image.save(f'{new_path}{n}_resized.jpg')
