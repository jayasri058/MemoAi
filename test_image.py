"""
Test script to verify image processing functionality
"""
from PIL import Image
import numpy as np

# Create a simple test image
width, height = 300, 200
image = Image.new('RGB', (width, height), color='red')
image.save('test_image.jpg')

print("Created test image: test_image.jpg")