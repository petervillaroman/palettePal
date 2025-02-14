from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
from PIL.ExifTags import TAGS

# def get_dominant_colors(image_path, k=6):
#     """Extract the k most dominant colors and their counts from an image using K-Means clustering."""
#     image = Image.open(image_path).convert("RGB")
#     image_np = np.array(image)
#     pixels = image_np.reshape(-1, 3)

#     kmeans = KMeans(n_clusters=k, n_init=10)
#     kmeans.fit(pixels)

#     # Get the cluster centers (dominant colors) and the pixel counts for each cluster
#     dominant_colors = kmeans.cluster_centers_.astype(int)
#     labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
#     # Sort by descending frequency
#     sorted_indices = np.argsort(-counts)
#     sorted_colors = dominant_colors[sorted_indices]
#     sorted_counts = counts[sorted_indices]

#     return sorted_colors, sorted_counts

def get_dominant_colors(image_path, k=6, resize_dim=(200, 200)):
    """Extract the k most dominant colors and their counts from an image using K-Means clustering.
       Optionally downscale the image to `resize_dim` for faster processing."""
    image = Image.open(image_path).convert("RGB")
    
    # Create a copy to downscale for color analysis without modifying the original image
    small_image = image.copy()
    small_image.thumbnail(resize_dim, Image.LANCZOS)
    
    image_np = np.array(small_image)
    pixels = image_np.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    # Get the cluster centers (dominant colors) and the pixel counts for each cluster
    dominant_colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    # Sort by descending frequency
    sorted_indices = np.argsort(-counts)
    sorted_colors = dominant_colors[sorted_indices]
    sorted_counts = counts[sorted_indices]

    return sorted_colors, sorted_counts


def get_exif_data(image_path):
    """Extract and format selected EXIF data from an image."""
    image = Image.open(image_path)
    exif_data = image._getexif()
    if not exif_data:
        print("No EXIF data found in the image.")
        return None
    
    exif_dict = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    aperture = exif_dict.get("FNumber", "Unknown")
    iso = exif_dict.get("ISOSpeedRatings", "Unknown")
    shutter_speed = exif_dict.get("ExposureTime", "Unknown")
    camera = exif_dict.get("Model", "Unknown")
    lens = exif_dict.get("LensModel", "Unknown")
    
    # Format shutter speed if it's a tuple
    if isinstance(shutter_speed, tuple):
        shutter_speed = f"{shutter_speed[0]}/{shutter_speed[1]} sec"
    
    return {
        "Aperture": f"f/{aperture}" if aperture != "Unknown" else "Unknown",
        "ISO": iso,
        "Shutter Speed": shutter_speed,
        "Camera": camera,
        "Lens": lens
    }

def create_combined_image(image_path, k=6):
    """Creates a new image with a white border that contains:
       - The original image at the top.
       - A separation gap.
       - Color swatches (with block widths proportional to each color's frequency).
       - EXIF data text (if available) displayed below the swatches.
    """
    # Load original image
    original = Image.open(image_path)
    width, height = original.size

    # Define border and extra space sizes
    border_width = int(width * 0.1)         # Left/right margin (10% of image width)
    spacing_between = 200                   # Extra space (in pixels) between the image and swatches
    swatch_area_height = int(height * 0.12)   # Height for the color swatches
    exif_area_height = int(height * 0.2)      # Height for EXIF text

    new_width = width + 2 * border_width
    new_height = height + spacing_between + swatch_area_height + exif_area_height + 2 * border_width

    # Create new blank image with a white background
    new_image = Image.new("RGB", (new_width, new_height), "white")
    
    # Paste the original image at the top with a border
    new_image.paste(original, (border_width, border_width))
    
    # Get dominant colors and their counts
    colors, counts = get_dominant_colors(image_path, k)
    total = np.sum(counts)
    
    # Draw color swatches proportionally to their frequency
    draw = ImageDraw.Draw(new_image)
    swatch_y_start = border_width + height + spacing_between  # Below the original image plus extra spacing
    swatch_area_width = width  # The swatches span the full width of the original image area

    cumulative_ratio = 0.0
    for color, count in zip(colors, counts):
        x0 = border_width + int(cumulative_ratio * swatch_area_width)
        cumulative_ratio += count / total
        x1 = border_width + int(cumulative_ratio * swatch_area_width)
        draw.rectangle([x0, swatch_y_start, x1, swatch_y_start + swatch_area_height],
                       fill=tuple(color), outline="black")
    
    # Retrieve EXIF data (if available)
    exif_data = get_exif_data(image_path)
    
    # Draw the EXIF data text below the color swatches
    if exif_data:
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        text_x = border_width
        text_y = swatch_y_start + swatch_area_height + int(border_width * 0.5)
        line_spacing = 25
        for key, value in exif_data.items():
            draw.text((text_x, text_y), f"{key}: {value}", fill="black", font=font)
            text_y += line_spacing

    # Save and display the final image
    new_image.save("final_output.jpg")
    new_image.show()

# Path to your image file
# query user for image path

image_path = input("Enter the path to your image file: ")

# Create and display the combined image with the desired modifications
create_combined_image(image_path, k=6)
