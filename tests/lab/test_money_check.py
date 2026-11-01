"""The money check (Part 2, decisions 8 and 10), without the network: its
situations, its verdicts on good and bad replies, and its report."""

from dataclasses import replace
from datetime import datetime

import pytest

from lab.week1.flash_lite_check import STYLE_PROMPT, Reply, situation_from
from lab.week1.money_check import (
    SLOT,
    TRIES,
    MoneySituation,
    load_money_situations,
    report,
    slot_verdict,
    verdict,
)
from packs.food.money import format_rupees
from packs.food.pack import FoodPack

CASES = {case.situation.id: case for case in load_money_situations()}
WORDS = FoodPack().spoken_words()


def test_eight_situations_each_with_its_amount_language_and_wording():
    assert list(CASES) == list(range(1, 9))
    assert all(case.say or case.situation.id == 8 for case in CASES.values())  # only 8 needs no amount
    assert {case.language for case in CASES.values()} == {"hinglish", "english"}
    assert TRIES >= 3  # one good reply at temperature 1.0 proves little


def test_the_talker_sees_the_slot_never_the_amount():
    for case in CASES.values():
        notes = " ".join(case.seen.notes)
        assert SLOT in notes and format_rupees(case.amount_paise) not in notes


def test_the_prompt_shows_none_of_the_situations_amounts():
    """The check must measure the slot rule, not the copying of an example."""
    prompt = STYLE_PROMPT.read_text(encoding="utf-8")
    assert SLOT in prompt
    for case in CASES.values():
        assert format_rupees(case.amount_paise) not in prompt


def test_a_situation_needs_its_fields_and_a_note_with_the_amount(tmp_path):
    path = tmp_path / "money.yaml"
    path.write_text('- {id: 1, good: g, turns: [{caller: "hi"}], notes: [], say: [], never: []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="needs amount_paise, language, say and never"):
        load_money_situations(path)
    path.write_text('- {id: 1, good: g, amount_paise: 6250, language: hinglish, turns: [{caller: "hi"}],'
                    ' notes: ["done: refunded ₹62.5"], say: [], never: []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no note says ₹62.50"):  # written unlike the backend
        load_money_situations(path)
    with pytest.raises(ValueError, match="needs exactly"):
        situation_from({"id": 1, "good": "g", "turns": [{"caller": "hi"}], "notes": [], "typo": 1})


@pytest.mark.parametrize(("case", "wrote", "problem", "spoken"), [
    (1, "Ho gaya. [amount] aapke UPI par wapas bhej diye gaye hain.", None,
     "Ho gaya. 62 rupaye 50 paise aapke UPI par wapas bhej diye gaye hain."),
    (2, "Done, [amount] went back to your UPI.", None, "Done, 62 rupees 50 paise went back to your UPI."),
    (6, "[amount] aapke UPI par aa gaye.", None, "50 paise aapke UPI par aa gaye."),
    (7, "[amount] aapke card par wapas aayenge.", None, "₹450 aapke card par wapas aayenge."),
    (4, "Mujhe maaf kijiye, 200 rupaye nahi, sirf [amount] bache hain.", None,  # the caller said 200
     "Mujhe maaf kijiye, 200 rupaye nahi, sirf 120 rupaye 50 paise bache hain."),
    (8, "Ek second, main check kar raha hoon.", None, "Ek second, main check kar raha hoon."),
    # What must never reach the caller:
    (1, "Ho gaya. 62 rupaye 50 paise wapas aaye.", "typed a number itself: 62, 50",
     "Ho gaya. 62 rupaye 50 paise wapas aaye."),
    (6, "5 paise wapas aaye.", "typed a number itself: 5", "5 paise wapas aaye."),  # the 1 Nov mistake
    (1, "Ho gaya. [refund] wapas aaye.", "made up a slot [refund]", "Ho gaya. [refund] wapas aaye."),
    (1, "Ho gaya, refund bhej diya gaya.", "left the amount out", "Ho gaya, refund bhej diya gaya."),
    (8, "₹99.50 ka refund check kar raha hoon.", "says a decimal amount",  # echoing the caller's decimal
     "₹99.50 ka refund check kar raha hoon."),
])
def test_each_reply_gets_the_right_verdict(case, wrote, problem, spoken):
    assert slot_verdict(wrote, CASES[case], WORDS) == (problem, spoken)


def test_the_filled_words_follow_the_calls_language():
    english = CASES[2]
    hinglish = replace(english, language="hinglish", never=())
    assert slot_verdict("[amount]", english, WORDS)[1] == "62 rupees 50 paise"
    assert slot_verdict("[amount]", hinglish, WORDS)[1] == "62 rupaye 50 paise"
    assert slot_verdict("[amount]", replace(english, never=("rupees",)), WORDS)[0] == 'says "rupees"'


def test_words_match_whole_not_inside_other_numbers():
    # "0 rupaye" (a never) is not inside "450 rupaye"...
    assert verdict("Aapke 450 rupaye 50 paise card par aayenge.", CASES[6]) is None
    # ...and "5 paise" (a say) is not inside "05 paise".
    five = MoneySituation(CASES[5].situation, 124905, "english", say=("5 paise",), never=())
    assert verdict("You paid 1249 rupees 05 paise.", five) == 'doesn\'t say the amount as "5 paise"'
    assert verdict("You paid 1249 rupees 5 paise.", five) is None


def test_the_report_counts_passes_and_counts_an_error_as_a_fail():
    one = CASES[1].seen
    replies = [Reply(one, "Ho gaya, [amount] aa gaye.", 0.5),
               Reply(one, "Ho gaya, 62 rupaye aa gaye.", 0.5),
               Reply(one, "", 0.5, error="ServerError: 503")]
    text, passed = report(list(CASES.values()), replies, datetime(2026, 11, 1, 15, 0), WORDS)
    assert passed == 1 and "**Passed:** 1 of 3" in text
    assert "| Ho gaya, 62 rupaye 50 paise aa gaye. | ✅ |" in text  # what the voice gets is shown
    assert "❌ typed a number itself: 62" in text and "❌ error: ServerError: 503" in text
