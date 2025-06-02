# Image Quality Enhancement Features

import cv2

def illumination_normalization(image, method="clahe"):
    """
    Normalizes the illumination of an image.

    Args:
        image: Input image (NumPy array).
        method: "clahe" (default), "gamma", etc.

    Returns:
        Normalized image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized_image = clahe.apply(gray)
    elif method == "gamma":
        gamma = 1.5  # Adjust as needed (gamma > 1 brightens, < 1 darkens)
        normalized_image = cv2.pow(gray / 255.0, gamma) * 255.0
        normalized_image = normalized_image.astype("uint8")
    else:
        normalized_image = gray  # Default: no normalization

    return normalized_image

# Example usage:
image = cv2.imread(r"models/corpus/Abhiram2.PNG")
if image is not None:
    normalized_face = illumination_normalization(image, method="clahe")
    cv2.imshow("Original", image)
    cv2.imshow("Normalized", normalized_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite("output1.png", normalized_face)
else:
    print("Error: Could not load image.")