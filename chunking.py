
import re

def clean_text(text):

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def chunk_text(text, chunk_size=500, overlap=75):

    text = clean_text(text)

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        # Prefer ending at a word boundary
        if end < len(text):
            space = text.rfind(" ", start, end)

            if space > start:
                end = space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


# Example webpage content
text = """
Acme AI builds autonomous AI agents for enterprise workflows.

Our products include research agents, coding agents,
and customer support agents.

We help businesses automate repetitive tasks
and improve productivity.
"""

cleaned = clean_text(text)

chunks = chunk_text(
    cleaned,
    chunk_size=100,
    overlap=20
)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
