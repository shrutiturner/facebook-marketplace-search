from PIL import Image
import os

def resize_image(final_size, image):
    size = image.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])

    image = image.resize(new_image_size, Image.ANTIALIAS)
    new_image = Image.new("RGB", (final_size, final_size))
    new_image.paste(image, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    
    return new_image


def rgb_channels(image, r, g, b):
    pass


if __name__ == '__main__':
    path = "images/"
    dirs = os.listdir(path)
    final_size = 512
    
    for n, item in enumerate(dirs[:5], 1):
        image = Image.open('images/' + item)
        new_image = resize_image(final_size, image)
        new_image.save(f'{n}_resized.jpg')
