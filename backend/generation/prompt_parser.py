import os
import json
from groq import Groq

# Create Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# All valid emotion and genre tags the music model understands
VALID_EMOTIONS = ["angry", "exciting", "fear", "funny", "happy", "lazy", "magnificent", "quiet", "romantic", "sad", "warm"]
VALID_GENRES   = ["classical", "country", "jazz", "pop", "rock", "traditional"]

# Synonym map: words users might type that map to our valid tags
EMOTION_SYNONYMS = {
    "angry":       ["angry", "anger", "aggressive", "intense", "furious", "mad", "fierce", "rage", "violent"],
    "exciting":    ["exciting", "excited", "thrilling", "energetic", "hype", "pumped", "dynamic", "upbeat", "lively"],
    "fear":        ["fear", "scary", "horror", "dark", "eerie", "spooky", "tense", "suspense", "creepy", "ominous"],
    "funny":       ["funny", "humorous", "playful", "silly", "quirky", "whimsical", "comic", "comedic", "fun", "lighthearted"],
    "happy":       ["happy", "happiness", "joyful", "cheerful", "bright", "positive", "uplifting", "celebratory", "glad", "joyous"],
    "lazy":        ["lazy", "chill", "relaxed", "mellow", "slow", "laid back", "laid-back", "lofi", "lo-fi", "easy"],
    "magnificent": ["magnificent", "epic", "grand", "majestic", "powerful", "triumphant", "glorious", "heroic", "soaring", "cinematic"],
    "quiet":       ["quiet", "calm", "peaceful", "gentle", "soft", "ambient", "serene", "tranquil", "meditation", "soothing"],
    "romantic":    ["romantic", "romance", "love", "passionate", "tender", "intimate", "dreamy", "longing", "sensual", "sweet"],
    "sad":         ["sad", "sadness", "melancholy", "melancholic", "emotional", "heartbreak", "heartbroken", "sorrowful", "tearful", "nostalgic", "lonely"],
    "warm":        ["warm", "cozy", "comforting", "homey", "hopeful", "optimistic"],
}

GENRE_SYNONYMS = {
    "classical":   ["classical", "orchestra", "orchestral", "symphonic", "symphony", "piano", "baroque", "opera", "chamber", "concert", "beethoven", "mozart", "bach"],
    "country":     ["country", "western", "bluegrass", "folk", "americana", "southern", "cowboy", "rustic", "texas", "nashville"],
    "jazz":        ["jazz", "blues", "swing", "bebop", "improvisation", "saxophone", "trumpet", "smooth", "new orleans"],
    "pop":         ["pop", "popular", "mainstream", "radio", "catchy", "modern", "contemporary", "dance"],
    "rock":        ["rock", "guitar", "electric", "band", "metal", "punk", "indie", "alternative", "hard rock"],
    "traditional": ["traditional", "folk", "ethnic", "cultural", "world", "heritage", "acoustic", "celtic", "irish", "spiritual"],
}


def parse_simple(text: str) -> tuple[str | None, str | None]:
    text_lower = text.lower()
    emotion = None
    genre   = None

    for tag, synonyms in EMOTION_SYNONYMS.items():
        if any(syn in text_lower for syn in synonyms):
            emotion = tag
            break

    for tag, synonyms in GENRE_SYNONYMS.items():
        if any(syn in text_lower for syn in synonyms):
            genre = tag
            break

    return emotion, genre


def parse_with_groq(text: str) -> tuple[str | None, str | None]:
    try:
        prompt = f"""
        You are a music prompt parser. Extract the emotion and genre from this user prompt.

        Valid emotions: {', '.join(VALID_EMOTIONS)}
        Valid genres: {', '.join(VALID_GENRES)}

        Rules:
        - Map synonyms to the closest valid tag
        - If no clear emotion detected, use null
        - If no clear genre detected, use null
        - Respond ONLY with valid JSON, no extra text
        - Format: {{"emotion": "tag_or_null", "genre": "tag_or_null"}}

        User prompt: "{text}"
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)
        emotion = result.get("emotion")
        genre   = result.get("genre")

        if emotion not in VALID_EMOTIONS:
            emotion = None
        if genre not in VALID_GENRES:
            genre = None

        return emotion, genre

    except Exception as e:
        print(f"Groq parse failed ({e}), falling back to keyword matching...")
        return parse_simple(text)


def parse_prompt(text: str, use_groq: bool = False) -> tuple[str | None, str | None]:
    return parse_with_groq(text) if use_groq else parse_simple(text)


def format_result(text: str, emotion: str | None, genre: str | None) -> str:
    return (
        f'Prompt: "{text}"\n'
        f"  Emotion: {emotion or '(none detected)'}\n"
        f"  Genre:   {genre   or '(none detected)'}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a music prompt into emotion and genre tags")
    parser.add_argument("prompt", type=str, help="Natural language music prompt")
    parser.add_argument("--groq", action="store_true", help="Use Groq API for smarter parsing")
    args = parser.parse_args()

    emotion, genre = parse_prompt(args.prompt, use_groq=args.groq)
    print(format_result(args.prompt, emotion, genre))