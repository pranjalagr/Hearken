"""The Talker's amount slots and number check (Part 2 decisions 8 and 10)."""

import pytest

from agents.agent_c.talker.slots import UnknownSlot, fill_slots, stray_numbers, used_slots
from packs.food.money import spoken_amount
from packs.food.pack import FoodPack

WORDS = FoodPack().spoken_words()
SLOTS = {"amount": spoken_amount(6250)}  # ₹62.50


@pytest.mark.parametrize(("language", "said"), [
    ("hinglish", "Ho gaya, 62 rupaye 50 paise aapke UPI par bhej diye gaye hain."),
    ("english", "Ho gaya, 62 rupees 50 paise aapke UPI par bhej diye gaye hain."),
])
def test_a_slot_is_filled_with_the_exact_words_in_the_replys_language(language, said):
    reply = "Ho gaya, [amount] aapke UPI par bhej diye gaye hain."
    assert fill_slots(reply, SLOTS, WORDS, language) == said


def test_slots_say_every_kind_of_amount_like_the_yes_questions_do():
    for paise, said in [(45000, "₹450"), (24905, "249 rupaye 5 paise"), (50, "50 paise")]:
        assert fill_slots("[amount]", {"amount": spoken_amount(paise)}, WORDS, "hinglish") == said


def test_two_slots_and_a_repeated_slot_are_all_filled():
    slots = {"refund": spoken_amount(6250), "total": spoken_amount(45000)}
    reply = "[refund] of [total] went back; [refund] in all."
    assert fill_slots(reply, slots, WORDS, "english") == "62 rupees 50 paise of ₹450 went back; 62 rupees 50 paise in all."
    assert used_slots(reply) == {"refund", "total"}


@pytest.mark.parametrize("made_up", ["[refund]", "[Amount]", "[amount ]", "[62.50]"])
def test_a_slot_no_note_gave_is_never_spoken(made_up):
    """Anything in square brackets is a slot; brackets never belong in speech."""
    with pytest.raises(UnknownSlot):
        fill_slots(f"Ho gaya, {made_up} wapas aa gaye.", SLOTS, WORDS, "hinglish")


def test_a_reply_without_slots_is_left_as_it_is():
    assert fill_slots("Ek second, check kar raha hoon.", SLOTS, WORDS, "hinglish") == "Ek second, check kar raha hoon."


# ---- decision 8's number check ---------------------------------------------------

NOTES = ["done: refunded [amount] to upi for order 4512"]


def test_numbers_from_the_notes_or_the_caller_are_allowed():
    caller = "Mujhe 200 rupaye wapas chahiye."
    assert stray_numbers("Order 4512 par 200 rupaye nahi, [amount] wapas aaye.", [*NOTES, caller]) == []


def test_a_number_the_talker_typed_itself_is_caught():
    # The amount is behind the slot, so 62 and 50 appear nowhere it was given.
    assert stray_numbers("Ho gaya, 62 rupaye 50 paise wapas aaye.", NOTES) == ["62", "50"]
    assert stray_numbers("5 paise wapas aaye.", ["done: refunded [amount]"]) == ["5"]  # the 1 Nov "5 paise"


def test_a_slots_own_name_is_not_read_as_a_number():
    assert stray_numbers("[amount2] aur [amount] wapas aaye.", NOTES) == []


def test_a_leading_zero_is_a_different_number():
    """"05 paise" could be read "zero five"; it doesn't count as the 5 in the notes."""
    assert stray_numbers("1249 rupees 05 paise", ["total paid: 1249 rupees 5 paise"]) == ["05"]
