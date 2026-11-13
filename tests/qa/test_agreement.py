import pytest

from docparse.qa.agreement import token_agreement, tokens

DE = "Die Größe der Straße beträgt über 100 Meter. Äpfel und Öl sind günstig, weiß man."


def test_german_text_against_itself_is_perfect() -> None:
    assert token_agreement(DE, DE) == (1.0, 1.0)


def test_umlauts_and_eszett_survive_tokenisation() -> None:
    toks = tokens(DE)
    # umlauts are kept; case-folding maps ß to ss on BOTH sides, so Straße == Strasse
    for word in ("grösse", "strasse", "über", "äpfel", "öl", "günstig", "weiss", "beträgt"):
        assert word in toks
    assert token_agreement("Die Straße ist weiß", "Die Strasse ist weiss") == (1.0, 1.0)


def test_markdown_and_html_are_stripped() -> None:
    md = "# Title\n\n**bold** <b>word</b> | cell |\n![x](a.png)"
    assert tokens(md) == ["title", "bold", "word", "cell"]


def test_single_chars_are_ignored() -> None:
    assert tokens("a b cc dd") == ["cc", "dd"]


def test_hyphenated_line_break_is_joined() -> None:
    layer = "Die Geschwindig-\nkeit des Lichts ist konstant"
    assert token_agreement("Die Geschwindigkeit des Lichts ist konstant", layer) == (1.0, 1.0)


def test_rapidfuzz_pairs_leftover_fragments() -> None:
    # same words, but one OCR'd with a one-letter difference: counts as a match via the tie-break
    layer = "internationalization of documents"
    vlm = "internationalisation of documents"
    recall, precision = token_agreement(vlm, layer)
    assert recall == precision == 1.0


def test_thirty_percent_missing_words_lowers_recall_only() -> None:
    words = [f"wort{i}x" for i in range(100)]
    layer = " ".join(words)
    vlm = " ".join(words[:70])
    recall, precision = token_agreement(vlm, layer)
    assert recall == pytest.approx(0.70) and precision == 1.0


def test_hallucinated_words_lower_precision_only() -> None:
    words = [f"wort{i}x" for i in range(100)]
    vlm = " ".join(words + [f"fake{i}y" for i in range(50)])
    recall, precision = token_agreement(vlm, " ".join(words))
    assert recall == 1.0 and precision == pytest.approx(100 / 150)


def test_empty_sides() -> None:
    assert token_agreement("", "") == (1.0, 1.0)
    assert token_agreement("", "some words here") == (0.0, 1.0)
    assert token_agreement("some words here", "") == (1.0, 0.0)


def test_letters_only_ignores_math_and_glyph_clusters() -> None:
    from docparse.qa.agreement import strip_math, token_agreement

    vlm = r"We show that $x \leq \infty$ holds, and $$\mathbb{R}^n$$ is complete."
    layer = "We show that ˆxη ≤ ∞ holds, and Rn is complete."
    assert strip_math(vlm).count("$") == 0
    r_all, _ = token_agreement(vlm, layer)
    r_let, p_let = token_agreement(strip_math(vlm), layer, 3, letters_only=True)
    assert r_let > r_all
    assert p_let == 1.0


def test_letters_only_keeps_umlauts_and_hyphen_breaks() -> None:
    from docparse.qa.agreement import token_agreement

    assert token_agreement("Straße über Größe", "Straße über Größe", 3, True) == (1.0, 1.0)
    assert token_agreement("Entwick-\nlung läuft", "Entwicklung läuft", 3, True) == (1.0, 1.0)
