
import os
import google.generativeai as genai
from typing import Optional, List
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class GeminiService:
    def __init__(self, api_key: str):
        if not api_key:
            print("Warning: No Gemini API Key provided")
            self.model = None
            return
            
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("Gemini Service initialized with gemini-2.5-flash")
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            self.model = None
            
        # Initialize fallback models if available
        self.blip_processor = None
        self.blip_model = None
        if HAS_TRANSFORMERS:
            try:
                print("Initializing Hugging Face BLIP model for fallback...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            except Exception as e:
                print(f"Failed to load HF fallback models: {e}")

    def describe_image(self, image_path: str) -> str:
        """
        Generates a description for an image using Gemini Vision.
        Returns a string description.
        """
        if not self.model:
            return "Gemini service not available."
            
        if not os.path.exists(image_path):
            return "Image file not found."

        try:
            # Upload the file to Gemini
            myfile = genai.upload_file(image_path)
            
            # fast/low-cost model for description
            result = self.model.generate_content(
                [myfile, "\n\n", "Describe this image in detail for memory storage. Include objects, text, colors, and context."]
            )
            return result.text
        except Exception as e:
            print(f"Gemini analysis failed: {e}. Attempting local fallback...")
            return self.fallback_describe_image(image_path)

    def fallback_describe_image(self, image_path: str) -> str:
        """
        Fallback image description using local BLIP model (Hugging Face)
        """
        if not self.blip_model or not self.blip_processor:
            return "Local image analysis fallback not available."
            
        try:
            raw_image = Image.open(image_path).convert('RGB')
            # unconditional image captioning
            inputs = self.blip_processor(raw_image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_new_tokens=50)
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return f"{description} (Analyzed using local AI)"
        except Exception as e:
            print(f"Local fallback analysis failed: {e}")
            return "Failed to analyze image with local fallback."

    def generate_tags(self, text: str) -> list[str]:
        """
        Generate tags using Gemini (optional, better than regex)
        """
        if not self.model:
            return []
            
        try:
            prompt = f"Analyze the following text and return a list of 3-5 relevant tags (lowercase, single words) separated by commas. Text: {text}"
            result = self.model.generate_content(prompt)
            tags_text = result.text.strip()
            return [t.strip().lower() for t in tags_text.split(',')]
        except Exception as e:
            print(f"Error generating tags: {e}")
            return []
