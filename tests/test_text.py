from arag.text import is_arabic, normalize, numbers_in, strip_html, tokenize


def test_arabic_normalization():
    assert normalize("أحمد إلى آخر") == normalize("احمد الي اخر")
    assert normalize("المتابعةُ ١٢") == "المتابعه 12"


def test_tokenize_strips_article_and_stopwords():
    assert tokenize("كم عدد الصالات الرياضية في الدوحة؟") == tokenize("صالات رياضية دوحة")
    assert tokenize("How many gyms in Doha?") == ["gym", "doha"]


def test_numbers_are_canonical():
    assert numbers_in("السكان ١٦٥٬٤٣٢ في 2023") == {"165432", "2023"}
    assert numbers_in("18,250.50 GWh") == {"18250.5"}


def test_helpers():
    assert strip_html("<p>A &amp; B</p>") == "A & B"
    assert is_arabic("كم عدد السكان؟") and not is_arabic("How many people?")
