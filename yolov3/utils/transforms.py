import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from yolov3.model_structure import image_size

# Transform for training 
train_transform = A.Compose( 
    [ 
        # Rescale an image so that maximum side is equal to image_size 
        A.LongestMaxSize(max_size=image_size), 
        # Pad remaining areas with zeros 
        A.PadIfNeeded( 
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
        ), 
        # Random color jittering 
        A.ColorJitter( 
            brightness=0.5, contrast=0.5, 
            saturation=0.5, hue=0.5, p=0.5
        ), 
        # Flip the image horizontally 
        A.HorizontalFlip(p=0.5), 
        # Normalize the image 
        A.Normalize( 
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ), 
        # Convert the image to PyTorch tensor 
        ToTensorV2() 
    ],  
    # Augmentation for bounding boxes 
    bbox_params=A.BboxParams( 
                    format="yolo",  
                    min_visibility=0.4,  
                    label_fields=[] 
                ) 
) 
  
# Transform for testing 
test_transform = A.Compose( 
    [ 
        # Rescale an image so that maximum side is equal to image_size 
        A.LongestMaxSize(max_size=image_size), 
        # Pad remaining areas with zeros 
        A.PadIfNeeded( 
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
        ), 
        # Normalize the image 
        A.Normalize( 
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ), 
        # Convert the image to PyTorch tensor 
        ToTensorV2() 
    ], 
    # Augmentation for bounding boxes  
    bbox_params=A.BboxParams( 
                    format="yolo",  
                    min_visibility=0.4,  
                    label_fields=[] 
                ) 
)

def resize_image(image_arr, bboxes, input_size, transform):
    """
    Code based on function from https://gist.github.com/sheldonsebastian/053540025ca00483f1a500b45252f5ef#file-resize-image-and-bb-py
    Resize the image and adjust the bounding boxes accordingly.
    Input: 
        - image_arr (as numpy array),
        - bboxes (as numpy array), h (height), w (width)
        - Input_size: tuple (h, w) for the desired output size
    Output:
        - resized_image (numpy array)
        - resized_bboxes (numpy array)  
    """
    h, w = input_size
    
    class_labels = ["Pole"]*len(bboxes)
    transformed = transform(image=image_arr, bboxes=bboxes, class_labels=class_labels)
    resized_image = transformed['image']
    resized_bboxes = transformed['bboxes']
    
    return resized_image, resized_bboxes
