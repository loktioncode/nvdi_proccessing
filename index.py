import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_smooth_colormap():
    """Create a custom colormap for NDVI visualization"""
    colors = ['brown', 'yellow', 'green']
    n_bins = 256  # more bins = smoother transition
    return LinearSegmentedColormap.from_list("custom", colors, N=n_bins)


def process_ndvi(image_path, output_path):
    """
    Process NoIR camera image with blue filter for NDVI analysis

    Parameters:
    image_path (str): Path to input image
    output_path (str): Path to save processed image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Split into BGR channels
    b, g, r = cv2.split(img)

    # Calculate NDVI
    # With NoIR + blue filter setup, we use blue channel as NIR
    # and red channel as visible light
    nir = b.astype(float)
    vis = r.astype(float)

    # Calculate reflectance (NIR)
    reflectance = cv2.normalize(nir, None, 0, 255, cv2.NORM_MINMAX)

    # Calculate NDVI
    ndvi = (nir - vis) / (nir + vis + 1e-7)

    # Normalize NDVI to 0-1 range
    ndvi_normalized = cv2.normalize(ndvi, None, 0, 1, cv2.NORM_MINMAX)

    # Create smooth colormap
    colormap = create_smooth_colormap()

    # Create figure with three subplots
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Reflectance (grayscale)
    plt.subplot(132)
    plt.imshow(reflectance, cmap='gray')
    plt.title('NIR Reflectance')
    plt.colorbar(label='Reflectance Value')
    plt.axis('off')

    # NDVI visualization with smooth colormap
    plt.subplot(133)
    ndvi_plot = plt.imshow(ndvi_normalized, cmap=colormap)
    plt.title('NDVI Analysis')
    plt.colorbar(ndvi_plot, label='NDVI Value')
    plt.axis('off')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save individual images
    base_name = output_path.rsplit('.', 1)[0]

    # Save reflectance image
    plt.figure(figsize=(8, 6))
    plt.imshow(reflectance, cmap='gray')
    plt.colorbar(label='Reflectance Value')
    plt.axis('off')
    plt.title('NIR Reflectance')
    plt.savefig(f"{base_name}_reflectance.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # Save NDVI colormap image
    plt.figure(figsize=(8, 6))
    plt.imshow(ndvi_normalized, cmap=colormap)
    plt.colorbar(label='NDVI Value')
    plt.axis('off')
    plt.title('NDVI Analysis')
    plt.savefig(f"{base_name}_ndvi.jpg", dpi=300, bbox_inches='tight')
    plt.close()

     # Calculate statistics
    stats = {
        'average_ndvi': np.mean(ndvi),
        'max_ndvi': np.max(ndvi),
        'min_ndvi': np.min(ndvi),
        'average_reflectance': np.mean(reflectance),
        # Use ndvi_normalized instead of raw ndvi for vegetation coverage
        'vegetation_coverage': np.mean(ndvi_normalized > 0.6)  # Changed from ndvi > 0.2
    }

    return stats


# Example usage
if __name__ == "__main__":
    try:
        stats = process_ndvi('test.jpg', 'ndvi_analysis.jpg')
        print("\nNDVI Analysis Complete!")
        print(f"Average NDVI: {stats['average_ndvi']:.3f}")
        print(f"Max NDVI: {stats['max_ndvi']:.3f}")
        print(f"Min NDVI: {stats['min_ndvi']:.3f}")
        print(f"Average NIR Reflectance: {stats['average_reflectance']:.1f}")
        print(f"Approximate Vegetation Coverage: {stats['vegetation_coverage'] * 100:.1f}%")
    except Exception as e:
        print(f"Error processing image: {str(e)}")