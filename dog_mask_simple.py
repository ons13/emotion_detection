import cv2 
import numpy as np


def apply_mask(face : np.array , mask : np.array)-> np.array:
    """add the mask to the provided face , and return the face with mask."""
    mask_h , mask_w , _ = mask.shape
    face_h , face_w , _ = face.shape
   
    #resize the mask to fit the face
    factor = min(face_h / mask_h , face_w / mask_w)
    new_mask_w = nt(factor * mask_w)
    new_mask_h = int(factor * mask_h)
    new_mask_shape = (new_mask_w , new_mask_h)
    #add mask to face ensure is centred 
    resized_mask = cv2.resize(mask , new_mask_shape)
    cv2.imwrite('outputs/resized_dog.png' , resized_mask)

def main():
    face = cv2.imread('assets/me.jpg')
    mask = cv2.imread('assets/dog_mask.png')
    face_with_mask = apply_mask(face, mask)


if __name__ == '__main__':  
    main()
