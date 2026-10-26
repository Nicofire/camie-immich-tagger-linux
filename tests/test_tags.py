from camie_tagger.tags import build_taglist, has_camie_tags, normalize_tag, tags_in_namespace


def test_tag_normalization_and_deduplication():
    prediction = {
        "character": [("a/b", 0.9), ("a_b", 0.8)],
        "rating": [("rating_general", 0.9)],
        "year": [("2024", 0.9)],
        "unknown": [("unused", 0.9)],
    }
    assert build_taglist(prediction) == ["character/a_b", "rating/general"]
    assert normalize_tag("a/b") == "a_b"


def test_empty_tag_names_are_not_written():
    assert build_taglist({"character": [("", 0.9), ("   ", 0.9)]}) == []
    assert build_taglist({"rating": [("rating_", 0.9)]}) == []


def test_namespace_matching_does_not_match_similar_prefixes():
    tags = ["character/alice", "characters/bob", "general/blue_hair"]
    assert has_camie_tags(tags)
    assert not has_camie_tags(["characters/bob"])
    assert tags_in_namespace(tags, "character") == ["character/alice"]