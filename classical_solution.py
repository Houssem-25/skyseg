import cv2
import numpy as np
from sklearn.cluster import KMeans
import cv2

def segment_sky(image, k=5):
    # Read the image
    w, h, _ = image.shape
    # Flatten the image into a 1D array
    flattened_image = image.reshape(-1, 3)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(flattened_image)

    # Get the cluster labels and cluster centers
    labels = kmeans.labels_

    # Reshape the labels back to the original image shape
    segmented_image = labels.reshape(image.shape[:2])
    sky_cluster = np.rint(segmented_image[:4,h//4:h*3//4].mean())

    # Create a binary mask for the sky cluster
    sky_mask = np.uint8(segmented_image == sky_cluster)

    # Apply morphological operations for post-processing (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to the original image to extract the sky segment
    segmented_sky = cv2.bitwise_and(image, image, mask=sky_mask)

    return segmented_sky, sky_mask


if __name__ == "__main__":
    # Load the image
    image = cv2.imread('demo.jpeg')

    # Segment the sky
    segmented_sky, sky_mask = segment_sky(image)

    # Display the original image and the segmented sky
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Sky', segmented_sky)
    cv2.waitKey(0)
    cv2.destroyAllWindows()