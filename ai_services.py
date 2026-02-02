
import os
import google.generativeai as genai
from typing import Optional

class GeminiService:
    def __init__(self, api_key: str):
        if not api_key:
            print("Warning: No Gemini API Key provided")
            self.model = None
            return
            
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("Gemini Service initialized")
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            self.model = None

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
            print(f"Error describing image: {e}")
            return "Failed to analyze image."

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
