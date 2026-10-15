
import os
import re


def generate_rule_based(sequence: str) -> str:
    """Generate a simple, predictable output from recognized sign labels."""

    if not sequence or not sequence.strip():
        return "No signs recognized yet."

    tokens = [
        token.strip().upper()
        for token in re.split(r"\s+", sequence.strip())
        if token.strip()
    ]

    # If all labels are single letters, display the spelling.
    if all(len(token) == 1 and token.isalpha() for token in tokens):
        return "Spelled: " + "".join(tokens)

    # Convert labels such as THANK_YOU into readable words.
    words = [token.replace("_", " ").lower() for token in tokens]
    text = " ".join(words)

    # Simple examples for your current vocabulary.
    templates = {
        "hello": "Hello.",
        "bye": "Goodbye.",
        "yes": "Yes.",
        "no": "No.",
        "please": "Please.",
        "sorry": "Sorry.",
        "thank you": "Thank you.",
        "help you": "Help you.",
        "i help you": "I can help you.",
    }

    if text in templates:
        return templates[text]

    # Keep unknown sequences conservative.
    return "Recognized signs: " + text.capitalize() + "."


def generate_with_gemini(sequence: str) -> str:
    """Generate a cautious English rendering using Gemini."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Install Gemini support using: "
            "python -m pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash"
    )

    prompt = f"""
You convert recognized sign labels into cautious English text.

The input comes from an ASL alphabet or isolated-word recognition
prototype. The labels may be incomplete or incorrectly recognized.

Rules:
- Preserve the meaning supported by the input.
- Do not invent missing words, events, or context.
- Expand underscores into spaces, such as THANK_YOU.
- If the input is ambiguous, use a conservative literal rendering.
- Do not claim that the result is a verified ASL translation.
- Return only the resulting text.

Recognized labels: {sequence}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    result = getattr(response, "text", None)

    if not result or not result.strip():
        raise RuntimeError("Gemini returned an empty response.")

    return result.strip()
