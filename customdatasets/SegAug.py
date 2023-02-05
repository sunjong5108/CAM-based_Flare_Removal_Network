from torchvision import transforms

def brightness_jitter(input, label):
    brightness_jitter = transforms.ColorJitter(brightness=0.5)
    brightness_jitter_input = brightness_jitter(input)
    
    return brightness_jitter_input, label

def color_jitter(input, label):
    brightness_jitter = transforms.ColorJitter(brightness=0.5, contrast=(0.7, 2.2))
    brightness_jitter_input = brightness_jitter(input)
    
    return brightness_jitter_input, label

def contrast_jitter(input, label):
    contrast_jitter = transforms.ColorJitter(contrast=(0.7, 2.2))
    contrast_jitter_input = contrast_jitter(input)
    
    return contrast_jitter_input, label

def random_crop(input, label, height, width):
    random_crop = transforms.RandomCrop(size=(height, width))
    random_crop_input = random_crop(input)
    random_crop_label = random_crop(label)
    
    return random_crop_input, random_crop_label

def horizontal_filp(input, label):
    horizontal_filp = transforms.RandomHorizontalFlip(p=1)
    horizontal_filp_input = horizontal_filp(input)
    horizontal_filp_label = horizontal_filp(label)
    
    return horizontal_filp_input, horizontal_filp_label

def vertical_filp(input, label):
    vertical_filp = transforms.RandomVerticalFlip(p=1)
    vertical_filp_input = vertical_filp(input)
    vertical_filp_label = vertical_filp(label)
    
    return vertical_filp_input, vertical_filp_label

def random_transforms_apply(mode, input, label):
    if mode == 1:
      brightness_jitter = transforms.ColorJitter(brightness=0.5)
      brightness_jitter_input = brightness_jitter(input)
      return brightness_jitter_input, label
    if mode == 2:
      contrast_jitter = transforms.ColorJitter(contrast=(0.7, 2.2))
      contrast_jitter_input = contrast_jitter(input)
      return contrast_jitter_input, label
    if mode == 3:
      color_jitter = transforms.ColorJitter(brightness=0.5, contrast=(0.7, 2.2))
      color_jitter_input = color_jitter(input)
      return color_jitter_input, label
    if mode == 4:
      horizontal_filp = transforms.RandomHorizontalFlip(p=1)
      horizontal_filp_input = horizontal_filp(input)
      horizontal_filp_label = horizontal_filp(label)
      return horizontal_filp_input, horizontal_filp_label
    if mode == 5:
      brightness_jitter = transforms.ColorJitter(brightness=0.5)
      brightness_jitter_input = brightness_jitter(input)
      horizontal_filp = transforms.RandomHorizontalFlip(p=1)
      horizontal_filp_input = horizontal_filp(brightness_jitter_input)
      horizontal_filp_label = horizontal_filp(label)
      return horizontal_filp_input, horizontal_filp_label
    if mode == 6:
      color_jitter = transforms.ColorJitter(brightness=0.5, contrast=(0.7, 2.2))
      color_jitter_input = color_jitter(input)
      horizontal_filp = transforms.RandomHorizontalFlip(p=1)
      horizontal_filp_input = horizontal_filp(color_jitter_input)
      horizontal_filp_label = horizontal_filp(label)
      return horizontal_filp_input, horizontal_filp_label
    if mode == 7:
      return input, label