# guard.py
import re

def extract_keywords(question: str):
    # keep meaningful words; you can improve later
    words = re.findall(r"[a-zA-Z]{3,}", question.lower())
    stop = {"what","why","how","the","and","about","there","any","discuss","discussion","video","this","that","with","from"}
    return [w for w in words if w not in stop]

def should_answer_yes_no(question: str, transcript_text: str) -> tuple[bool, str]:
    """
    If the question is a yes/no 'is X discussed' type:
    - we require that at least one keyword appears in transcript.
    """
    q = question.lower()
    if not any(x in q for x in ["is there", "is this", "does it", "discuss", "discussion", "mentioned"]):
        return True, ""  # not a yes/no style check, proceed

    kws = extract_keywords(question)
    t = transcript_text.lower()
    if kws and not any(k in t for k in kws):
        return False, f"I couldn't find these keywords in the transcript: {kws[:8]}"
    return True, ""