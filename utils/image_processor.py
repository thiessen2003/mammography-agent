"""
Image processing utility for medical images.
"""

import base64
import io
from typing import Union, Optional
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Utility class for processing medical images."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    MAX_SIZE = (2048, 2048)  # Maximum image dimensions
    
    @classmethod
    def load_image(cls, image_path: Union[str, Path]) -> Image.Image:
        """Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if image_path.suffix.lower() not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        try:
            image = Image.open(image_path)
            logger.info(f"Loaded image: {image_path} ({image.size}, {image.mode})")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    @classmethod
    def encode_to_base64(cls, image: Image.Image, format: str = "PNG") -> str:
        """Encode PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            format: Output format (PNG, JPEG, etc.)
            
        Returns:
            Base64 encoded string
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            buffer.seek(0)
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logger.debug(f"Encoded image to base64 ({len(encoded)} characters)")
            return encoded
        except Exception as e:
            raise ValueError(f"Failed to encode image to base64: {e}")
    
    @classmethod
    def decode_from_base64(cls, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image.
        
        Args:
            base64_string: Base64 encoded image data
            
        Returns:
            PIL Image object
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            logger.debug(f"Decoded base64 image ({image.size}, {image.mode})")
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")
    
    @classmethod
    def resize_image(cls, image: Image.Image, max_size: tuple = None) -> Image.Image:
        """Resize image while maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            max_size: Maximum size tuple (width, height)
            
        Returns:
            Resized PIL Image object
        """
        if max_size is None:
            max_size = cls.MAX_SIZE
        
        if image.size[0] <= max_size[0] and image.size[1] <= max_size[1]:
            return image
        
        # Calculate new size maintaining aspect ratio
        ratio = min(max_size[0] / image.size[0], max_size[1] / image.size[1])
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"Resized image from {image.size} to {resized.size}")
        return resized
    
    @classmethod
    def convert_to_rgb(cls, image: Image.Image) -> Image.Image:
        """Convert image to RGB mode if needed.
        
        Args:
            image: PIL Image object
            
        Returns:
            RGB mode PIL Image object
        """
        if image.mode == 'RGB':
            return image
        
        if image.mode in ('RGBA', 'LA'):
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image)
            return background
        
        # Convert other modes to RGB
        rgb_image = image.convert('RGB')
        logger.info(f"Converted image from {image.mode} to RGB")
        return rgb_image
    
    @classmethod
    def process_medical_image(cls, image_path: Union[str, Path], 
                            resize: bool = True, 
                            convert_rgb: bool = True) -> str:
        """Process a medical image for analysis.
        
        Args:
            image_path: Path to the image file
            resize: Whether to resize the image
            convert_rgb: Whether to convert to RGB
            
        Returns:
            Base64 encoded image string ready for analysis
        """
        try:
            # Load image
            image = cls.load_image(image_path)
            
            # Convert to RGB if needed
            if convert_rgb:
                image = cls.convert_to_rgb(image)
            
            # Resize if needed
            if resize:
                image = cls.resize_image(image)
            
            # Encode to base64
            base64_string = cls.encode_to_base64(image)
            
            logger.info(f"Successfully processed medical image: {image_path}")
            return base64_string
            
        except Exception as e:
            logger.error(f"Failed to process medical image {image_path}: {e}")
            raise
    
    @classmethod
    def validate_image(cls, image_path: Union[str, Path]) -> bool:
        """Validate that an image file is suitable for analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            image = cls.load_image(image_path)
            
            # Check minimum size
            if image.size[0] < 100 or image.size[1] < 100:
                logger.warning(f"Image too small: {image.size}")
                return False
            
            # Check maximum size
            if image.size[0] > 4096 or image.size[1] > 4096:
                logger.warning(f"Image too large: {image.size}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed for {image_path}: {e}")
            return False
