# Image Quality Enhancement Features

import cv2
import numpy as np

def find_which_preprocess(image):
    '''to find which pre-prpcessing method is to be applied on the given image'''   
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    average_intensity = np.mean(gray_image)

    high_brightness_threshold = 200
    low_brightness_threshold = 70


    if average_intensity > high_brightness_threshold or average_intensity < low_brightness_threshold:
        return 'uneven brightness'

    return 'normal'
    
def preprocessing(image, method='low_resolution_sharpness'):
    processed_image = image.copy()
    
    if method == "brightness":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        gamma = 1.0
        if mean_brightness < 128:
            gamma = 1 - (mean_brightness / 255.0) * 0.5 # Brighter for darker images
        elif mean_brightness > 128:
            gamma = 1 + ((mean_brightness - 128) / 127.0) * 0.5 # Darker for brighter images

        # Apply gamma correction to the full BGR image for better results
        processed_image = np.array(255 * (image / 255.0)**gamma, dtype='uint8')
        
    elif method == "low_resolution_sharpness":
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        processed_image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    return processed_image

def imgShow(image1,image2):
    cv2.namedWindow('My Resizable Window',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('My Resizable Window', 800, 600)
    cv2.imshow('My Resizable Window',image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.namedWindow('My Resizable Window',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('My Resizable Window', 800, 600)
    cv2.imshow('My Resizable Window',image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"------------------------------------------------------------------------------------------------------"
def generated_preprocessing(image, method='low_resolution_sharpness'):

    processed_image = image.copy() # Start with a copy of the original image

    if method == "normalization_grey":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_image = clahe.apply(gray)
    
    elif method == "combained":
        processed_image = preprocessing(image,'low_resolution_sharpness')
        processed_image = preprocessing(image,'gray')
    
    elif method == "normalization_colour":
        # Note: If you want to apply CLAHE to each color channel, you'd do:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged_lab = cv2.merge([cl, a_channel, b_channel])
        processed_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    elif method == "brightness":
        # Works on grayscale, but can be adapted for color too
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        gamma = 1.0
        if mean_brightness < 128:
            gamma = 1 - (mean_brightness / 255.0) * 0.5 # Brighter for darker images
        elif mean_brightness > 128:
            gamma = 1 + ((mean_brightness - 128) / 127.0) * 0.5 # Darker for brighter images

        # Apply gamma correction to the full BGR image for better results
        processed_image = np.array(255 * (image / 255.0)**gamma, dtype='uint8')

    elif method == "gray":
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif method == "yellowing_browning":
        # Convert to Lab color space for robust color correction
        # L = lightness, a = green-red, b = blue-yellow
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)

        # For yellowing/browning, we typically need to reduce the 'b' component
        # (move towards blue) and sometimes adjust 'a' (move away from red/green).
        # These are empirical adjustments, may need tuning.
        b_corrected = b.astype(np.float32) - 10 # Decrease yellow/brown (shift towards blue)
        a_corrected = a.astype(np.float32) # You might slightly adjust 'a' if needed

        # Clip values to valid range (0-255 for uint8)
        b_corrected = np.clip(b_corrected, 0, 255).astype(np.uint8)
        a_corrected = np.clip(a_corrected, 0, 255).astype(np.uint8) # Or original a

        merged_lab = cv2.merge([L, a_corrected, b_corrected])
        processed_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    elif method == "Gray_World":
        # You could also try a simple white balance algorithm (e.g., Gray World)
        # This often works well for general color casts.
        R, G, B = cv2.split(image)
        avg_R, avg_G, avg_B = np.mean(R), np.mean(G), np.mean(B)
        avg_gray = (avg_R + avg_G + avg_B) / 3
        R_gain = avg_gray / (avg_R + 1e-6)
        G_gain = avg_gray / (avg_G + 1e-6)
        B_gain = avg_gray / (avg_B + 1e-6)
        R_corrected = np.clip(R * R_gain, 0, 255).astype(np.uint8)
        G_corrected = np.clip(G * G_gain, 0, 255).astype(np.uint8)
        B_corrected = np.clip(B * B_gain, 0, 255).astype(np.uint8)
        processed_image = cv2.merge([B_corrected, G_corrected, R_corrected]) # OpenCV expects BGR

    elif method == "fading_desaturation":
        # Convert to HSV or HSL to adjust saturation directly
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Increase saturation (e.g., by 20-30%)
        s_increase = s.astype(np.float32) * 1.3 # Factor > 1 to increase saturation
        s_increase = np.clip(s_increase, 0, 255).astype(np.uint8)

        merged_hsv = cv2.merge([h, s_increase, v])
        processed_image = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)

        # Also apply a slight contrast enhancement
        clahe_for_fading = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        # Convert to Lab, apply CLAHE to L channel, then convert back
        lab_for_fading = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        l_for_fading, a_for_fading, b_for_fading = cv2.split(lab_for_fading)
        cl_for_fading = clahe_for_fading.apply(l_for_fading)
        merged_lab_for_fading = cv2.merge([cl_for_fading, a_for_fading, b_for_fading])
        processed_image = cv2.cvtColor(merged_lab_for_fading, cv2.COLOR_LAB2BGR)


    elif method == "low_resolution_sharpness":
        # Apply a sharpening filter. A common approach is to use unsharp masking.
        # This involves blurring the image and subtracting the blurred version from the original.
        # Gaussian blur for the 'mask'
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        # Add weighted original and blurred to sharpen. Alpha > 1 for sharpening.
        processed_image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        # You might need to clip values if they go out of 0-255 range due to addWeighted
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

        # For very low resolution, super-resolution techniques (deep learning)
        # would be needed, which are beyond basic cv2.

    elif method == "poor_lighting":
        # Combine brightness and contrast enhancement
        # First, convert to Lab to work on lightness channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel for contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)) # Slightly higher clipLimit for poor lighting
        cl = clahe.apply(L)

        # Apply a global brightness adjustment (e.g., simple linear scaling or gamma)
        # Let's use gamma correction on the L channel for brightness, similar to 'brightness' method
        mean_lightness = np.mean(cl) # Use CLAHE-adjusted lightness for mean
        gamma_poor_light = 1.0
        if mean_lightness < 128:
            gamma_poor_light = 1 - (mean_lightness / 255.0) * 0.7 # More aggressive brightening for very dark images
        
        # Apply gamma to the CLAHE-adjusted lightness channel
        cl_gamma = np.array(255 * (cl / 255.0)**gamma_poor_light, dtype='uint8')
        
        merged_lab = cv2.merge([cl_gamma, a, b])
        processed_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    elif method == "reflections_glare":
        # Reducing reflections/glare is challenging with simple methods as it often involves
        # separating specular and diffuse components or requires multiple images (e.g., polarization).
        # A basic attempt might be localized contrast reduction or highlight suppression,
        # but results can be inconsistent.
        # Here's a very basic attempt at highlight reduction by darkening brightest spots:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Threshold to find very bright areas (potential glare)
        _, glare_mask = cv2.threshold(v, 240, 255, cv2.THRESH_BINARY) # Adjust 240 as needed

        # Apply a localized darkening or blurring to glare spots
        # This is a rudimentary approach and may not look natural
        glare_mask_inv = cv2.bitwise_not(glare_mask)
        darkened_v = cv2.subtract(v, 30, mask=glare_mask) # Darken glare areas by 30 (adjust value)
        # Combine original V where no glare, and darkened V where glare
        v_no_glare = cv2.bitwise_and(v, v, mask=glare_mask_inv)
        v_final = cv2.add(v_no_glare, darkened_v)

        merged_hsv = cv2.merge([h, s, v_final])
        processed_image = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)

        print("Warning: 'reflections_glare' method is rudimentary and may not yield optimal results for all types of glare.")

    else:
        print(f"Warning: Unknown preprocessing method '{method}'. No processing applied.")
        processed_image = image # Return original image if method is unknown

    return processed_image

def old_preprocessing(image,method='normalization'):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    if method == "normalization":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized_image = clahe.apply(gray)
    elif method == "brightness":
        mean_brightness = np.mean(gray)
        gamma = (100 / (mean_brightness + 1e-6))  # Adjust as needed (gamma > 1 brightens, < 1 darkens)
        normalized_image = cv2.pow(gray / 255.0, gamma) * 255.0
        normalized_image = normalized_image.astype("uint8")
    elif method == "gray":
        normalized_image = gray  # Default: no normalization
    else:pass
        
    return normalized_image
    
# Example usage:

# image=cv2.imread("corpus/Anees3.jpg")

# if image is not None:
#     normalized_face = illumination_normalization(image)
#     # imgShow(image,normalized_face)
    
#     cv2.imwrite("output.png", normalized_face)
# else:
#     print("Error: Could not load image.")