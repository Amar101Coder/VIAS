import re


DIFFICULT_WORDS = {
    "utilize": "use",
    "implement": "apply",
    "commence": "start",
    "terminate": "end",
    "numerous": "many",
}


def simplify_text(text):
    simplified = text
    for hard, easy in DIFFICULT_WORDS.items():
        simplified = re.sub(rf"\b{hard}\b", easy, simplified, flags=re.IGNORECASE)
    return simplified


def highlight_difficult_words(text):
    highlighted = text
    for hard, easy in DIFFICULT_WORDS.items():
        highlighted = re.sub(
            rf"\b({hard})\b",
            rf'<span class="highlight" title="Meaning: {easy}">\1</span>',
            highlighted,
            flags=re.IGNORECASE
        )
    return highlighted
