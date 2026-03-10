import re
from pathlib import Path

import polars as pl


# Pre-compiled regex for sentence splitting (compiled once per worker at import).
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


# Worker-local globals, set by _init_file_worker via Pool initializer.
_worker_input_dir: Path | None = None
_worker_output_dir: Path | None = None

def segment_article(text: str) -> list[str]:
    """Segment article text into 100-word passages with sentence alignment.

    Algorithm:
      1. Split text into sentences (regex on .!? followed by whitespace)
      2. Greedily pack sentences into segments while word_count < 100
      3. Sentences > 100 words: mechanical split at 100-word boundaries;
         final fragment < 100 words uses backward slide (last 100 words
         of the sentence) to preserve contextual contiguity
      4. Normal segments < 100 words: pad to exactly 100 words with words
         from the article's first segment (mirrors DPR padding strategy)

    Args:
        text: Full article text.

    Returns:
        List of passage strings, each exactly 100 words.
    """
    # Sentence split using pre-compiled regex.
    # Imperfect for abbreviations ("U.S. Army" splits after "U.S.") but
    # harmless: short fragments get merged into the same segment.
    sentences = _SENTENCE_SPLIT.split(text)

    # --- Greedy sentence accumulation ---
    # Pack sentences into segments, keeping word count strictly < 100.
    # When the next sentence would push the count to >= 100, close the
    # current segment and start a new one with that sentence.
    segments = []
    current = []
    current_wc = 0

    for sent in sentences:
        # count(' ') + 1: word count without allocating a list.
        # Equivalent to len(split(' ')) but O(1) memory (no list created).
        sent_wc = sent.count(' ') + 1

        if sent_wc >= 100:
            # --- Long sentence handler ---
            # Close any accumulated segment first
            if current:
                segments.append(' '.join(current))
                current = []
                current_wc = 0

            # Mechanical split at 100-word boundaries
            words = sent.split(' ')
            while len(words) >= 100:
                segments.append(' '.join(words[:100]))
                words = words[100:]

            # Final fragment < 100 words: backward slide.
            # Take the last 100 words of the original sentence to preserve
            # contextual contiguity (accepts overlap with previous chunk).
            if words:
                all_words = sent.split(' ')
                segments.append(' '.join(all_words[-100:]))
        elif current and current_wc + sent_wc >= 100:
            # Adding this sentence would reach/exceed 100 — close segment
            segments.append(' '.join(current))
            current = [sent]
            current_wc = sent_wc
        else:
            current.append(sent)
            current_wc += sent_wc

    # Flush remaining sentences as the last segment
    if current:
        segments.append(' '.join(current))

    if not segments:
        return []

    # --- Pad to exactly 100 words ---
    # Source: words from the first segment (same strategy as DPR, which pads
    # the last chunk from the article's beginning). Gives every passage an
    # explicit anchor to the article's identity.
    #
    # Circular repetition: if first_words has fewer words than padding_needed,
    # we cycle through them repeatedly. This guarantees exactly 100 words
    # even for very short articles (e.g., 30-word article needs 70 padding
    # words — cycles through the 30 words ~2.3 times).
    first_words = segments[0].split(' ')
    n_first = len(first_words)
    padded = []

    for seg in segments:
        n = seg.count(' ') + 1
        if n < 100:
            padding_needed = 100 - n
            # Modular indexing: word at position i comes from first_words[i % n_first]
            padding_words = [first_words[i % n_first] for i in range(padding_needed)]
            padded.append(seg + ' ' + ' '.join(padding_words))
        else:
            padded.append(seg)

    return padded


def _init_file_worker(input_dir: str, output_dir: str) -> None:
    """Pool initializer: store I/O paths in worker-local globals."""
    global _worker_input_dir, _worker_output_dir
    _worker_input_dir = Path(input_dir)
    _worker_output_dir = Path(output_dir)


def file_segment_worker(frag_idx: int) -> int:
    """Process one input fragment, write output fragment.

    Reads input_dir/frag_{idx}.tsv (columns: title, text),
    applies segment_article to each article, writes flat passages
    to output_dir/frag_{idx}.tsv (columns: text, title).

    Resumability: if output file already exists, skip processing.

    Args:
        frag_idx: fragment index (0-99).

    Returns:
        Number of passages produced (-1 if skipped due to resumability).
    """
    input_path = _worker_input_dir / f"frag_{frag_idx}.tsv"
    output_path = _worker_output_dir / f"frag_{frag_idx}.tsv"

    # Resumability: skip if output already exists
    if output_path.exists():
        return -1

    df = pl.read_csv(input_path, separator="\t")

    titles = df["title"].to_list()
    texts = df["text"].to_list()

    # Build flat passage rows: one (text, title) per passage
    out_texts = []
    out_titles = []

    for title, text in zip(titles, texts):
        passages = segment_article(text)
        for passage in passages:
            out_texts.append(passage)
            out_titles.append(title)

    result_df = pl.DataFrame({
        "text": out_texts,
        "title": out_titles,
    })
    result_df.write_csv(output_path, separator="\t")

    return len(out_texts)

