"""The food pack's spoken words (Part 2 decision 8)."""

from engine.contract.spoken import LANGUAGES
from packs.food import rules
from packs.food.pack import FoodPack

# Every status an order can have. Built from rules.py, so a new status there
# fails this test until it gets a phrase too.
ALL_STATUSES = set(rules.FORWARD_PATH) | {rules.CANCELLED}


def test_every_order_status_has_a_phrase_and_no_extra_ones():
    assert FoodPack().spoken_words().values("status") == ALL_STATUSES


def test_every_status_phrase_exists_in_every_language():
    words = FoodPack().spoken_words()
    for status in ALL_STATUSES:
        for language in LANGUAGES:
            assert words.phrase("status", status, language)


def test_a_status_phrase_reads_as_expected():
    words = FoodPack().spoken_words()
    assert words.phrase("status", "preparing", "hinglish") == "Kitchen mein aapka khana ban raha hai."
