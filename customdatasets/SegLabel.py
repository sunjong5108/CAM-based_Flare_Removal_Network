import numpy as np

label_color_map = {"Sky": [128, 128, 128],
                  "Building": [128, 0, 0],
                  "Pole": [192, 192, 128],
                  "Road": [128, 64, 128],
                  "Pavement": [0, 0, 192],
                  "Tree": [128, 128, 0],
                  "SignSymbol": [192, 128, 128],
                  "Fence": [64, 64, 128],
                  "Car": [64, 0, 128],
                  "Pedestrian": [64, 64, 0],
                  "Bicyclist": [0, 128, 192],
                  "Void": [0, 0, 0]}

def rgb_to_mask(mask, color_map):
    class_names = list(color_map.keys())
    result_label = np.zeros((mask.shape[0], mask.shape[1], len(class_names)), dtype=np.uint8)

    for j, name in enumerate(class_names):
        result_label[:, :, j] = np.all(mask.reshape((-1, 3)) == label_color_map[name], axis=1).reshape(result_label.shape[:2])
  
    return result_label.transpose(2, 0, 1)

def mask_to_rgb(masks, pred=True, color_map=label_color_map):
  masks = np.array(masks)
  class_names = list(color_map.keys())

  if pred == True:
    test_pred = np.argmax(masks, axis=0)
    output = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)

    for k, name in enumerate(class_names):
      output[test_pred == k] = color_map[name]
  else:
    output = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)

    for k, name in enumerate(class_names):
      output[masks == k] = color_map[name]

  return output