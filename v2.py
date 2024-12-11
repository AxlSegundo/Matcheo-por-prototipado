import os
import shutil
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN

def init_samples(samples, dataset, sample_number):
    # Reset samples folder to avoid sample size conflicts
    if os.path.exists(samples):
        shutil.rmtree(samples)  
    os.makedirs(samples)

    for class_folder in os.listdir(dataset):
        class_folder_path = os.path.join(dataset, class_folder)

        # Recreate folders in samples
        sample_class_folder = os.path.join(samples, class_folder)
        os.makedirs(sample_class_folder, exist_ok=True)

        files = os.listdir(class_folder_path)
        files = [f for f in files if os.path.isfile(os.path.join(class_folder_path, f))]
        selected_files = files[:sample_number]
        for file_name in selected_files:
            src_path = os.path.join(class_folder_path, file_name)
            dst_path = os.path.join(sample_class_folder, file_name)
            shutil.copy(src_path, dst_path)

def preprocess_image(image, resize_shape=None):
    # Resize (if specified)
    if resize_shape:
        image = cv.resize(image, resize_shape, interpolation=cv.INTER_AREA)
    
    # Step 1: Blurring
    image = cv.GaussianBlur(image, (3, 3), 0)
    
    # Step 2: Normalization (scale pixel values to 0–1)
    image = image / 255.0  # Convert to float and normalize
    
    # Step 3: Histogram Equalization
    # Equalization requires scaling back to 0-255 and converting to uint8
    image = (image * 255).astype(np.uint8)  # Scale back
    image = cv.equalizeHist(image)
    
    # Step 4: Thresholding
    _, image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    
    return image

def eval_image(samples, testing_images, likelihood_threshold, template_size, proximity_threshold, output_dir):
    # Load templates into a dictionary by class
    class_templates = {}
    for class_folder in os.listdir(samples):
        class_folder_path = os.path.join(samples, class_folder)
        if os.path.isdir(class_folder_path):
            class_templates[class_folder] = [
                preprocess_image(
                    cv.imread(os.path.join(class_folder_path, file), cv.IMREAD_GRAYSCALE),
                    (template_size, template_size)
                )
                for file in os.listdir(class_folder_path) 
                if os.path.isfile(os.path.join(class_folder_path, file))
            ]

    # Load and preprocess testing images
    images = [
        (file, cv.resize(cv.imread(os.path.join(testing_images, file)),(900,900),interpolation=cv.INTER_AREA))##Add resize to improve performance
        for file in os.listdir(testing_images) 
        if os.path.isfile(os.path.join(testing_images, file))
    ]

    # Create output directory for marked images
    os.makedirs(output_dir, exist_ok=True)

    # Match templates against test images
    for image_idx, (file_name, image) in enumerate(images):
        print(f"Evaluating test image {image_idx + 1}/{len(images)}...")
        grayscale_image = preprocess_image(cv.cvtColor(image, cv.COLOR_BGR2GRAY),(900,900))##Add resize to improve performance

        all_matches = []
        for class_name, templates in class_templates.items():
            for template in templates:
                matches = search_template(grayscale_image, template, class_name, likelihood_threshold)
                all_matches.extend(matches)
                #Old method to draw all regions individually
                '''for x, y, tag in matches:
                    # Draw rectangles around matches
                    template_height, template_width = template.shape
                    top_left = (x, y)
                    bottom_right = (x + template_width, y + template_height)
                    color = COLORS[tag]
                    cv.rectangle(image, top_left, bottom_right, color, 2)

                #if matches:
                #    print(f"Matches found for {class_name} in {file_name}: {len(matches)} matches")'''

        zones = merge_matches(all_matches, proximity_threshold)

        # Draw zones on the image
        draw_zones(image, zones)

        # Save the marked image
        output_path = os.path.join(output_dir, file_name)
        cv.imwrite(output_path, image)
        print(f"Marked image saved to {output_path}")


def search_template(image, template, tag, likelihood_threshold):
    template_height, template_width = template.shape
    image_height, image_width = image.shape
    marks = []

    # Sliding window search
    for y in range(image_height - template_height + 1):
        for x in range(image_width - template_width + 1):
            # Extract region of interest
            image_region = image[y:y+template_height, x:x+template_width]
            
            # Use NCC as error metric, originally used absolute error but had horrible accuracy
            ncc = normalized_correlation_coefficient(template, image_region)
            
            # Check likelihood threshold
            if ncc >= likelihood_threshold:
                marks.append((x, y, tag))  # Mark the match
                
    return marks



#used for testing as is faster than custom method
def search_template_opencv(image, template, tag, likelihood_threshold):
    """
    Search for a template in an image using OpenCV's built-in function and return matching regions.

    Parameters:
    - image: The test image (grayscale).
    - template: The template image (grayscale).
    - tag: The class or label of the template.
    - likelihood_threshold: The similarity threshold for matches.

    Returns:
    - List of tuples (x, y, tag) where matches were found.
    """
    # Perform template matching using normalized correlation coefficient
    result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    
    # Threshold to find good matches
    match_locations = np.where(result >= likelihood_threshold)
    
    # Return match coordinates with the tag
    matches = [(x, y, tag) for y, x in zip(*match_locations)]
    return matches

def merge_matches(matches, proximity_threshold):
    zones = []
    tags = set([match[2] for match in matches])  # Exrtract unique tags

    for tag in tags:
        # Filter matches by tag
        tag_matches = [(x, y) for x, y, t in matches if t == tag]
        if not tag_matches:
            continue
        
        # Cluster matches using DBSCAN
        clustering = DBSCAN(eps=proximity_threshold, min_samples=1).fit(tag_matches)
        clusters = clustering.labels_

        # Create bounding boxes for each found cluster
        for cluster_id in set(clusters):
            cluster_points = np.array([tag_matches[i] for i in range(len(clusters)) if clusters[i] == cluster_id])
            x_min, y_min = cluster_points.min(axis=0)
            x_max, y_max = cluster_points.max(axis=0)
            zones.append({"tag": tag, "bounding_box": (x_min, y_min, x_max, y_max)})

    return zones

def draw_zones(image, zones):
    for zone in zones:
        color = COLORS[zone["tag"]]
        x_min, y_min, x_max, y_max = zone["bounding_box"]
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2) 
        #cv.putText(image, zone["tag"], (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) #Descriptive text, really noisy for lots of matches

def normalized_correlation_coefficient(template, image_region):
    # Ensure the template and region are the same size
    assert template.shape == image_region.shape, "Template and region must have the same dimensions."
    
    # Compute means
    template_mean = np.mean(template)
    region_mean = np.mean(image_region)
    
    # Subtract means
    template_normalized = template - template_mean
    region_normalized = image_region - region_mean
    
    # Compute numerator
    numerator = np.sum(template_normalized * region_normalized)
    
    # Compute denominator
    template_std = np.sqrt(np.sum(template_normalized ** 2))
    region_std = np.sqrt(np.sum(region_normalized ** 2))
    denominator = template_std * region_std
    
    # Avoid division by zero
    if denominator == 0:
        return 0
    
    # Compute NCC
    ncc = numerator / denominator
    return ncc



if __name__ == "__main__":
    SAMPLE_NUMBER = 1 #Number of templates taken from DB
    LIKELIHOOD_THRESHOLD = 0.3  # Adjust as needed, best performance around 0.4-0.5 for opencv, around 0.3 for the other one
    TEMPLATE_SIZE = 64  # Size to which templates are resized, should not move unless switching template data
    OUTPUT_DIR = "marked_images" #Output path
    PROXIMITY_THRESHOLD = 10 #Max ndistance between zones for clustering
    COLORS = {
        "AnnualCrop": (3, 252, 11),
        "Forest": (14, 51, 15),
        "HerbaceousVegetation": (120, 255, 237),
        "Highway": (0, 0, 0),
        "Industrial": (69, 69, 69),
        "Pasture": (126, 67, 148),
        "PermanentCrop": (143, 51, 70),
        "Residential": (255, 0, 53),
        "River": (21, 0, 255),
        "SeaLake": (217, 109, 9),
    } #Dyuers coloures :))))

    # Directories
    samples = "Prototypes"
    dataset = "EuroSAT_RGB"
    test = "testing_images"

    init_samples(samples, dataset, SAMPLE_NUMBER)
    eval_image(samples, test, LIKELIHOOD_THRESHOLD, TEMPLATE_SIZE, PROXIMITY_THRESHOLD, OUTPUT_DIR)

