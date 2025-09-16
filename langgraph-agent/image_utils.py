"""
Image processing utilities for medical imaging analysis.
"""

import logging
import base64
import io
from typing import Union, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

logger = logging.getLogger(__name__)


class MedicalImageProcessor:
    """Utility class for processing medical images."""
    
    @staticmethod
    def load_image(image_input: Union[str, Image.Image]) -> Image.Image:
        """
        Load image from various input formats.
        
        Args:
            image_input: Image path, base64 string, PIL Image, or data URL
            
        Returns:
            PIL Image object
        """
        if isinstance(image_input, Image.Image):
            return image_input
        
        if isinstance(image_input, str):
            if image_input.startswith('data:image'):
                # Handle data URL
                image_data = image_input.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                return Image.open(io.BytesIO(image_bytes))
            elif image_input.startswith('/') or image_input.startswith('.'):
                # Handle file path
                return Image.open(image_input)
            else:
                # Assume base64 string
                image_bytes = base64.b64decode(image_input)
                return Image.open(io.BytesIO(image_bytes))
        
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    @staticmethod
    def preprocess_for_analysis(image: Image.Image, 
                              target_size: Tuple[int, int] = (512, 512),
                              enhance_contrast: bool = True,
                              normalize: bool = True) -> Image.Image:
        """
        Preprocess medical image for analysis.
        
        Args:
            image: Input PIL Image
            target_size: Target size for resizing
            enhance_contrast: Whether to enhance contrast
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Enhance contrast if requested
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # Slight contrast enhancement
        
        # Normalize if requested
        if normalize:
            # Convert to numpy array for normalization
            img_array = np.array(image)
            
            # Normalize to 0-1 range
            img_array = img_array.astype(np.float32) / 255.0
            
            # Apply histogram equalization for better contrast
            img_array = np.clip(img_array, 0, 1)
            
            # Convert back to PIL Image
            img_array = (img_array * 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        return image
    
    @staticmethod
    def extract_roi(image: Image.Image, 
                   center: Tuple[int, int] = None,
                   size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """
        Extract region of interest from medical image.
        
        Args:
            image: Input PIL Image
            center: Center point for ROI extraction (default: image center)
            size: Size of ROI to extract
            
        Returns:
            Extracted ROI as PIL Image
        """
        if center is None:
            center = (image.width // 2, image.height // 2)
        
        # Calculate ROI bounds
        half_width = size[0] // 2
        half_height = size[1] // 2
        
        left = max(0, center[0] - half_width)
        top = max(0, center[1] - half_height)
        right = min(image.width, center[0] + half_width)
        bottom = min(image.height, center[1] + half_height)
        
        # Extract ROI
        roi = image.crop((left, top, right, bottom))
        
        # Resize to target size if needed
        if roi.size != size:
            roi = roi.resize(size, Image.Resampling.LANCZOS)
        
        return roi
    
    @staticmethod
    def create_image_pyramid(image: Image.Image, 
                           scales: list = [1.0, 0.75, 0.5]) -> list:
        """
        Create image pyramid at different scales.
        
        Args:
            image: Input PIL Image
            scales: List of scale factors
            
        Returns:
            List of PIL Images at different scales
        """
        pyramid = []
        
        for scale in scales:
            if scale == 1.0:
                pyramid.append(image)
            else:
                new_size = (int(image.width * scale), int(image.height * scale))
                scaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
                pyramid.append(scaled_image)
        
        return pyramid
    
    @staticmethod
    def apply_medical_filters(image: Image.Image) -> dict:
        """
        Apply various medical imaging filters for analysis.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Dictionary of filtered images
        """
        filtered_images = {}
        
        # Original image
        filtered_images['original'] = image
        
        # Edge detection
        filtered_images['edges'] = image.filter(ImageFilter.FIND_EDGES)
        
        # Sharpen
        filtered_images['sharpened'] = image.filter(ImageFilter.SHARPEN)
        
        # Smooth
        filtered_images['smoothed'] = image.filter(ImageFilter.SMOOTH)
        
        # High contrast
        enhancer = ImageEnhance.Contrast(image)
        filtered_images['high_contrast'] = enhancer.enhance(2.0)
        
        # Low contrast
        filtered_images['low_contrast'] = enhancer.enhance(0.5)
        
        return filtered_images
    
    @staticmethod
    def convert_to_base64(image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        image_bytes = buffer.getvalue()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/{format.lower()};base64,{base64_string}"
    
    @staticmethod
    def validate_medical_image(image: Image.Image) -> Tuple[bool, str]:
        """
        Validate if image is suitable for medical analysis.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check image size
        if image.width < 64 or image.height < 64:
            return False, "Image too small for analysis"
        
        if image.width > 4096 or image.height > 4096:
            return False, "Image too large for analysis"
        
        # Check image mode
        if image.mode not in ['RGB', 'L', 'RGBA']:
            return False, f"Unsupported image mode: {image.mode}"
        
        # Check if image is not completely black or white
        img_array = np.array(image.convert('L'))
        if img_array.min() == img_array.max():
            return False, "Image appears to be uniform (all black or white)"
        
        return True, "Image is valid for analysis"


def create_sample_medical_image(width: int = 512, height: int = 512) -> Image.Image:
    """
    Create a sample medical image for testing.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Sample PIL Image
    """
    # Create a simple test pattern
    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Add some structure to make it look more like a medical image
    center_x, center_y = width // 2, height // 2
    
    # Create a circular region with different intensity
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 < (min(width, height) // 4)**2
    img_array[mask] = img_array[mask] // 2
    
    return Image.fromarray(img_array)
