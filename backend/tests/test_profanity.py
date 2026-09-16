from app.pipeline.profanity import PRESETS, WORD_GROUPS, find_matches, mask_text

ALL = [g["id"] for g in WORD_GROUPS]


def groups_of(text, groups=ALL, custom=()):
    return [(m.group_id, m.text) for m in find_matches(text, groups, custom)]


def test_f_word_variants():
    text = "What the fuck? You fucking idiot, motherfucker. Fuckin' A. That's f***ed up."
    found = groups_of(text, ["f_word"])
    assert [t for _, t in found] == ["fuck", "fucking", "motherfucker", "Fuckin'", "f***ed"]


def test_goddamn_beats_damn_and_phrases():
    found = groups_of("God damn it. Goddamn. Damn.", ALL)
    assert found == [("goddamn", "God damn"), ("goddamn", "Goddamn"), ("damn", "Damn")]


def test_allowlist_and_boundaries():
    assert groups_of("Hello, the shell class passed. Shiitake mushrooms in the cockpit of Niger.") == []
    assert groups_of("Charles Dickens wrote about assassins.") == []


def test_quotes_and_apostrophes():
    found = groups_of("He said ‘shit’ twice")
    assert found == [("s_word", "shit")]


def test_custom_words_and_wildcards():
    found = groups_of("Frickin' heck, what the frick", [], ["frick*", "heck"])
    assert [g for g, _ in found] == ["custom:frick*", "custom:heck", "custom:frick*"]


def test_lords_name_is_contextual_group():
    found = groups_of("Oh my God. Jesus Christ! Merry Christmas.", ["lords_name"])
    assert [t for _, t in found] == ["Oh my God", "Jesus Christ"]


def test_mask_keeps_layout():
    text = "Oh shit,\nget down"
    m = find_matches(text, ["s_word"])[0]
    assert mask_text(text, [(m.start, m.end)]) == "Oh ****,\nget down"


def test_presets_reference_real_groups():
    ids = {g["id"] for g in WORD_GROUPS}
    for p in PRESETS:
        assert set(p["groups"]) <= ids
