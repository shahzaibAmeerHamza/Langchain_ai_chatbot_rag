# gemini_client.py
import os
import requests

class GeminiClient:
    """
    Wrapper for Google Gemini API (free-tier compatible).
    Supports v1 endpoints and handles simple text generation calls.
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")

        # Use the correct free endpoint and model (v1)
        # You can switch to "gemini-1.5-pro" for higher quality if you want later
        self.api_url = os.getenv(
            "GEMINI_API_URL",
            "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
        )

        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY. Please set it in your environment.")

    def generate(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 512) -> str:
        """
        Sends a text prompt to the Gemini API and returns the response text.
        """

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens
            }
        }

        response = requests.post(self.api_url, headers=headers, params=params, json=payload, timeout=30)

        if response.status_code != 200:
            raise RuntimeError(f"Gemini API error: {response.status_code}, {response.text}")

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            # Return raw response if parsing fails (for debugging)
            return str(data)
