import pytest

pytest.importorskip("bm25s")

from rdp import Registry, Store
from research.corpus.prepare import (
    LEAD,
    chunk,
    first_sentence,
    is_topic,
    register,
    split_sections,
    write_corpus,
)
from research.index.build import build_indexes
from research.tools import Corpus, FlatBM25, Structured

PROSE = "This sentence is long enough to count as a paragraph of real article prose here."

ALIEN = {
    "id": "1", "url": "https://en.wikipedia.org/wiki/Alien_(film)", "title": "Alien (film)",
    "text": (
        "Alien is a 1979 science fiction horror film directed by Ridley Scott. " + PROSE + "\n\n"
        "Plot\n\nThe crew of the Nostromo answers a distress signal on a moon. " + PROSE + "\n\n"
        "Production\n\nDan O'Bannon wrote the screenplay with Ronald Shusett. " + PROSE + "\n\n"
        "References\n\nSome reference list entry that is long enough to be prose text here."
    ),
}
ABBEY = {
    "id": "2", "url": "https://en.wikipedia.org/wiki/Abbey_Road", "title": "Abbey Road",
    "text": "Abbey Road is the eleventh studio album by the English rock band the Beatles. "
            + PROSE + "\n\nRecording\n\nSessions took place at EMI Studios in London. " + PROSE,
}
OFF_TOPIC = {"id": "3", "url": "u", "title": "Granite",
             "text": "Granite is a coarse-grained igneous rock. " + PROSE}
LIST_PAGE = {"id": "4", "url": "u", "title": "List of films",
             "text": "This is a list of films that is a film list. " + PROSE}


def test_topic_filter():
    assert is_topic(ALIEN["text"]) and is_topic(ABBEY["text"])
    assert not is_topic(OFF_TOPIC["text"])


def test_topic_filter_ignores_capitalized_title_words():
    assert not is_topic("Melisandre is a character in the A Song of Ice and Fire novels.")
    assert is_topic("Seinfeld is an American TV series created by Larry David.")


def test_first_sentence_skips_initials_and_abbreviations():
    text = ("Melisandre is a character in novels by George R. R. Martin. She is a priestess.")
    assert first_sentence(text) == "Melisandre is a character in novels by George R. R. Martin."
    assert first_sentence("Sammy Davis Jr. was a singer. He danced.") == \
        "Sammy Davis Jr. was a singer."


def test_list_headings_detected_before_lists():
    text = ("Alien is a 1979 film. " + PROSE + "\n\nPlot\n\nThe crew lands. " + PROSE
            + "\n\nCast\n\nTom Skerritt as Dallas\nSigourney Weaver as Ripley")
    secs = split_sections(text)
    assert [h for h, _ in secs] == [LEAD, "Plot", "Cast"]
    assert "Sigourney Weaver" in secs[2][1] and "Sigourney" not in secs[1][1]


def test_split_sections_finds_headings_and_drops_references():
    secs = split_sections(ALIEN["text"])
    assert [h for h, _ in secs] == [LEAD, "Plot", "Production"]
    assert secs[2][1].startswith("Dan O'Bannon")


def test_split_sections_keeps_list_items_inside_their_section():
    text = ("X is an album. " + PROSE + "\n\nTrack listing\n\nSong one\nSong two\n"
            "Sideways\nSong three")
    secs = split_sections(text)
    assert [h for h, _ in secs] == [LEAD, "Track listing"]
    assert "Sideways" in secs[1][1]


def test_chunk_sizes():
    long_para = " ".join(["word"] * 500)
    chunks = chunk("short para here\n" + long_para)
    assert all(len(c.split()) <= 220 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 503


@pytest.fixture
def corpus(tmp_path):
    data = tmp_path / "data"
    store = Store(tmp_path / "store")
    paths = write_corpus([ALIEN, OFF_TOPIC, LIST_PAGE, ABBEY], data / "corpus", max_docs=10)
    versions = register(paths, store, "test")
    build_indexes(data, store)
    return Corpus(data), store, versions


def test_corpus_filters_and_lineage(corpus):
    c, store, versions = corpus
    assert sorted(d["title"] for d in c.docs.values()) == ["Abbey Road", "Alien (film)"]
    chain = [v for v, _, _ in Registry(store).lineage("bm25_passages:v1")]
    assert chain == ["bm25_passages:v1", versions["passages"], versions["sections"],
                     versions["docs"]]


def test_flat_search(corpus):
    a = FlatBM25(corpus[0])
    out = a.call("search", {"query": "Nostromo distress signal"})
    assert out.startswith("[1] Alien (film) — Plot")
    assert a.call("search", '{"query": "the of"}') == "No results."   # stopwords only
    assert a.call("search", '{"query": "zzzqqq"}') == "No results."   # unknown term


def test_structured_navigation(corpus):
    c = Structured(corpus[0])
    assert c.call("search_docs", {"query": "Beatles album"}).startswith("- Abbey Road:")
    outline = c.call("open_doc", {"title": "alien (film)"})
    assert "- Plot" in outline and "- Production" in outline and "References" not in outline
    assert "O'Bannon" in c.call("read_section", {"title": "Alien (film)",
                                                 "section": "production"})
    assert "Did you mean" in c.call("open_doc", {"title": "Alien film"})
    assert "Sections:" in c.call("read_section", {"title": "Alien (film)", "section": "Cast"})


def test_tool_errors_are_returned_not_raised(corpus):
    c = Structured(corpus[0])
    assert c.call("search", {"query": "x"}).startswith("Error: unknown tool")
    assert c.call("open_doc", "{bad json").startswith("Error: arguments")
    assert c.call("open_doc", {"name": "x"}).startswith("Error: bad arguments")
