from torchvision import transforms
from PIL import Image

def image_processor(image, final_size) -> transforms.ToTensor:
    """Processes a single input image to the desired size.

    Args:
        image (str): Image to ingest
        final_size (int): one side square dimension of desired final image size.

    Returns:
        transforms.ToTensor: Tensor of processed image.
    """
    size = image.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])

    image = image.resize(new_image_size, Image.ANTIALIAS)
    new_image = Image.new("RGB", (final_size, final_size))
    new_image.paste(image, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))

    transform = transforms.PILToTensor() 
    new_image_tensor = transform(new_image)
    new_image_tensor = new_image_tensor.reshape(1, 3, 64, 64)
    
    return new_image_tensor