import os
import string
import warnings
from core.spacy_utils.load_nlp_model import init_nlp, SPLIT_BY_CONNECTOR_FILE
from core.utils import *
from core.utils.models import _3_1_SPLIT_BY_NLP

warnings.filterwarnings("ignore", category=FutureWarning)

# Minimum words per segment to avoid translation quality issues
MIN_WORDS_PER_SEGMENT = 4


def count_words(text, language=None):
    """
    Count words in text, handling both space-separated and CJK languages.
    """
    if not text or not isinstance(text, str):
        return 0

    text = text.strip()

    # Check if text contains CJK characters
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or
                   '\u3040' <= c <= '\u309f' or
                   '\u30a0' <= c <= '\u30ff' or
                   '\uac00' <= c <= '\ud7af')

    if cjk_chars > len(text) * 0.3:  # Mostly CJK
        return sum(1 for c in text if c.isalnum())
    else:
        return len(text.split())


def is_incomplete_segment(text):
    """
    Check if a segment is incomplete (should be merged with neighbor).

    Incomplete segments:
    - Start with comma, period, or other punctuation
    - Start with lowercase letter (continuation of previous sentence)
    - Too short (handled separately by word count)
    """
    if not text:
        return True

    text = text.strip()
    if not text:
        return True

    first_char = text[0]

    # Starts with punctuation (comma, period, etc.)
    if first_char in ',.:;，。：；、':
        return True

    # Starts with lowercase (for Latin languages - indicates continuation)
    if first_char.islower() and first_char.isalpha():
        return True

    return False


def ends_with_incomplete(text, nlp=None):
    """
    Check if a segment ends with a preposition/conjunction (incomplete sentence).

    Uses spacy POS tagging for dynamic language-independent detection.
    Falls back to basic heuristics if nlp not available.

    Incomplete POS tags:
    - ADP: adposition (preposition) - "on", "в", "для"
    - CCONJ: coordinating conjunction - "and", "и", "but"
    - SCONJ: subordinating conjunction - "if", "если", "because"
    - DET: determiner/article - "the", "a", "этот"
    """
    if not text:
        return False

    text = text.strip()
    if not text:
        return False

    # Use spacy for dynamic POS-based detection
    if nlp is not None:
        doc = nlp(text)
        # Get last non-punctuation token
        last_token = None
        for token in reversed(doc):
            if not token.is_punct and not token.is_space:
                last_token = token
                break

        if last_token:
            # POS tags indicating incomplete sentence
            incomplete_pos = {'ADP', 'CCONJ', 'SCONJ', 'DET'}
            if last_token.pos_ in incomplete_pos:
                return True

            # Also check for relative pronouns (dependency-based)
            if last_token.dep_ in {'mark', 'cc', 'prep', 'det'}:
                return True

        return False

    # Fallback: basic punctuation check (no nlp available)
    words = text.split()
    if not words:
        return False

    last_word = words[-1].lower().rstrip('.,;:!?，。；：')
    # Minimal fallback list for common cases
    basic_incomplete = {'and', 'or', 'but', 'the', 'a', 'an', 'to', 'for', 'with',
                        'и', 'а', 'но', 'на', 'в', 'для', 'с', 'к'}
    return last_word in basic_incomplete


def merge_short_segments(sentences, min_words=MIN_WORDS_PER_SEGMENT, joiner=' ', nlp=None):
    """
    Merge segments that are too short or incomplete for good translation quality.

    Merges segments that:
    - Have fewer than min_words
    - Start with punctuation (comma, period)
    - Start with lowercase letter (sentence continuation)
    - End with preposition/conjunction (detected via spacy POS tagging)

    Args:
        sentences: List of sentence strings
        min_words: Minimum word count per segment
        joiner: Character to join merged segments
        nlp: spacy nlp model for POS-based detection (optional)

    Returns:
        List of merged sentences
    """
    if not sentences:
        return sentences

    result = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        word_count = count_words(sent)
        starts_incomplete = is_incomplete_segment(sent)
        ends_incomplete = ends_with_incomplete(sent, nlp)

        # Merge if too short OR starts incomplete OR ends incomplete
        needs_merge = word_count < min_words or starts_incomplete or ends_incomplete

        if needs_merge:
            if starts_incomplete and word_count >= min_words:
                rprint(f"[yellow]📝 Incomplete start ('{sent[0]}'): '{sent[:40]}...'[/yellow]")
            if ends_incomplete and word_count >= min_words:
                last_word = sent.split()[-1].rstrip('.,;:!?') if sent.split() else ''
                rprint(f"[yellow]📝 Incomplete ending ('{last_word}'): '...{sent[-40:]}'[/yellow]")

            # Accumulate segment
            if current:
                current = current + joiner + sent
            else:
                current = sent
        else:
            # Normal segment
            if current:
                # We have accumulated incomplete segments
                current_words = count_words(current)
                current_starts_incomplete = is_incomplete_segment(current)
                current_ends_incomplete = ends_with_incomplete(current, nlp)

                if current_words < min_words or current_starts_incomplete or current_ends_incomplete:
                    # Still problematic - merge with this segment
                    sent = current + joiner + sent
                    rprint(f"[yellow]📝 Merged: '{current[:30]}...' → with next[/yellow]")
                else:
                    # Accumulated segment is now good
                    result.append(current)
                current = ""
            result.append(sent)

    # Handle remaining accumulated text
    if current:
        if result:
            # Merge with last segment
            rprint(f"[yellow]📝 Merged trailing: '{current[:30]}...' → with previous[/yellow]")
            result[-1] = result[-1] + joiner + current
        else:
            # Only problematic segments - keep as is
            result.append(current)

    return result

def split_long_sentence(doc):
    tokens = [token.text for token in doc]
    n = len(tokens)
    
    # dynamic programming array, dp[i] represents the optimal split scheme from the start to the ith token
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    # record optimal split points
    prev = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(max(0, i - 100), i):  # limit search range to avoid overly long sentences
            if i - j >= 30:  # ensure sentence length is at least 30
                token = doc[i-1]
                if j == 0 or (token.is_sent_end or token.pos_ in ['VERB', 'AUX'] or token.dep_ == 'ROOT'):
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
                        prev[i] = j
    
    # rebuild sentences based on optimal split points
    sentences = []
    i = n
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language # consider force english case
    joiner = get_joiner(language)
    while i > 0:
        j = prev[i]
        sentences.append(joiner.join(tokens[j:i]).strip())
        i = j
    
    return sentences[::-1]  # reverse list to keep original order

def split_extremely_long_sentence(doc):
    tokens = [token.text for token in doc]
    n = len(tokens)
    
    num_parts = (n + 59) // 60  # round up
    
    part_length = n // num_parts
    
    sentences = []
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language # consider force english case
    joiner = get_joiner(language)
    for i in range(num_parts):
        start = i * part_length
        end = start + part_length if i < num_parts - 1 else n
        sentence = joiner.join(tokens[start:end])
        sentences.append(sentence)
    
    return sentences


def split_long_by_root_main(nlp):
    with open(SPLIT_BY_CONNECTOR_FILE, "r", encoding="utf-8") as input_file:
        sentences = input_file.readlines()

    all_split_sentences = []
    for sentence in sentences:
        doc = nlp(sentence.strip())
        if len(doc) > 60:
            split_sentences = split_long_sentence(doc)
            if any(len(nlp(sent)) > 60 for sent in split_sentences):
                split_sentences = [subsent for sent in split_sentences for subsent in split_extremely_long_sentence(nlp(sent))]
            all_split_sentences.extend(split_sentences)
            rprint(f"[yellow]✂️  Splitting long sentences by root: {sentence[:30]}...[/yellow]")
        else:
            all_split_sentences.append(sentence.strip())

    punctuation = string.punctuation + "'" + '"'  # include all punctuation and apostrophe ' and "

    # Filter out empty and punctuation-only lines, merge with previous
    filtered_sentences = []
    for i, sentence in enumerate(all_split_sentences):
        stripped_sentence = sentence.strip()
        if not stripped_sentence or all(char in punctuation for char in stripped_sentence):
            rprint(f"[yellow]⚠️  Warning: Empty or punctuation-only line detected at index {i}[/yellow]")
            if filtered_sentences:
                filtered_sentences[-1] += sentence
            continue
        filtered_sentences.append(stripped_sentence)

    # Merge short segments to improve translation quality
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)

    original_count = len(filtered_sentences)
    filtered_sentences = merge_short_segments(filtered_sentences, min_words=MIN_WORDS_PER_SEGMENT, joiner=joiner, nlp=nlp)
    if len(filtered_sentences) < original_count:
        rprint(f"[green]✓ Merged {original_count - len(filtered_sentences)} incomplete segments (short/incomplete start/end)[/green]")

    with open(_3_1_SPLIT_BY_NLP, "w", encoding="utf-8") as output_file:
        for sentence in filtered_sentences:
            output_file.write(sentence + "\n")

    # delete the original file
    os.remove(SPLIT_BY_CONNECTOR_FILE)   

    rprint(f"[green]💾 Long sentences split by root saved to →  {_3_1_SPLIT_BY_NLP}[/green]")

if __name__ == "__main__":
    nlp = init_nlp()
    split_long_by_root_main(nlp)
    # raw = "平口さんの盛り上げごまが初めて売れました本当に嬉しいです本当にやっぱり見た瞬間いいって言ってくれるそういうコマを作るのがやっぱりいいですよねその2ヶ月後チコさんが何やらそわそわしていましたなんか気持ち悪いやってきたのは平口さんの駒の評判を聞きつけた愛知県の収集家ですこの男性師匠大沢さんの駒も持っているといいますちょっと褒めすぎかなでも確実にファンは広がっているようです自信がない部分をすごく感じてたのでこれで自信を持って進んでくれるなっていう本当に始まったばっかりこれからいろいろ挑戦していってくれるといいなと思って今月平口さんはある場所を訪れましたこれまで数々のタイトル戦でコマを提供してきた老舗5番手平口さんのコマを扱いたいと言いますいいですねぇ困ってだんだん成長しますので大切に使ってそういう長く良い駒になる駒ですね商談が終わった後店主があるものを取り出しましたこの前の名人戦で使った駒があるんですけど去年、名人銭で使われた盛り上げごま低く盛り上げて品良くするというのは難しい素晴らしいですね平口さんが目指す高みですこういった感じで作れればまだまだですけどただ、多分、咲く。"
    # nlp = init_nlp()
    # doc = nlp(raw.strip())
    # for sent in split_still_long_sentence(doc):
    #     print(sent, '\n==========')
