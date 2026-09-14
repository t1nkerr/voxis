from voxis.speaker_context import clean_speaker_context


def test_link_destinations_removed_but_visible_names_preserved():
    source = 'Guest [Peter Yang](https://example.com/PeterYang). <a href="https://x.com/peteryang">Peter Yang</a>'
    assert clean_speaker_context(source) == "Guest Peter Yang. Peter Yang"


def test_bare_urls_and_nested_parentheses():
    assert clean_speaker_context("Peter: https://example.com/a_(PeterYang)\nHost: Alice") == "Peter: \nHost: Alice"
    assert clean_speaker_context("[Alice](https://example.com/a_(b))") == "Alice"


def test_reference_links_and_autolinks():
    text = "[Peter Yang][person]\n\n[person]: https://example.com/PeterYang\n<https://example.com/OtherName>"
    assert clean_speaker_context(text) == "Peter Yang"


def test_plain_names_are_not_reformatted():
    assert clean_speaker_context("PeterYang and 李丰, host Peter.") == "PeterYang and 李丰, host Peter."
