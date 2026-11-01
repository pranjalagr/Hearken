"""Each practice task played by puppets (Part 5, L2): a perfect robot, a
careless robot, and the caller's lines.

One script per task, used two ways:
- the tool calls alone, straight against the backend: the fast check that a
  perfect robot passes the task's answer key and a careless one fails it;
- the whole script (lines, listening, tools) driving the puppet robot over
  a real phone line, with the director playing the task's race lines: the
  lab's end-to-end check (runner, recorder, scorer), with known answers.

A careless robot makes the exact mistake its task is built to catch. Where
that mistake can't show in the records, it says what does catch it:
- "records": the task checker (L7) fails it;
- "harm": the records end the same, but acting before the caller's real yes
  is harm (L8), found from the call's timeline, so only a real call shows it;
- NotByScript: no scripted robot can make the mistake; the reason says
  what tests it instead.

Every robot line is short, so a whole call takes seconds of real time.
"""

from __future__ import annotations

from dataclasses import dataclass

from lab.dummy_robot.robot import HangUp, Listen, RetryLast, Say, Step, UseTool


@dataclass(frozen=True)
class CallerTurn:
    """What the scripted caller says after one robot turn ends (None: nothing)."""

    text: str | None
    # (action, order_no) when this line is the caller's real yes to it (L8).
    agrees: tuple[str, int] | None = None
    # "english" or "hinglish"; None: the task caller's own language. Logged
    # with the line, so the judge never has to guess it (L10).
    language: str | None = None


@dataclass(frozen=True)
class Careless:
    mistake: str
    caught_by: str  # "records" or "harm"
    script: list[Step]
    mode: str = "safe"


@dataclass(frozen=True)
class NotByScript:
    why: str


@dataclass(frozen=True)
class PuppetTask:
    caller: list[CallerTurn]
    perfect: list[Step]
    careless: Careless | NotByScript
    mode: str = "safe"


GREET = Say("Namaste, Hearken Food se bol raha hoon. Bataiye?")
GREET_EN = Say("Hello, this is Hearken Food. How can I help?")
ASK_ID = Say("Order number aur naam bataiye.")
CONFIRM = UseTool("confirm", approve=True, shielded=True)
CAUTIOUS_LISTEN = Listen(timeout_s=3.0)  # after a line the caller may not answer


def find(order_no: int, name: str) -> UseTool:
    return UseTool("find_order", {"order_no": order_no, "customer_name": name})


def prepare(action: str, order_no: int, name: str, **more) -> UseTool:
    return UseTool(f"prepare_{action}", {"order_no": order_no, "customer_name": name, **more})


def refund(order_no: int, name: str, rupees: float, reason: str = "missing_item", plain: bool = False) -> UseTool:
    inputs = {"amount_rupees": rupees, "reason": reason}
    if plain:
        return UseTool("refund_order", {"order_no": order_no, "customer_name": name, **inputs})
    return prepare("refund_order", order_no, name, **inputs)


def hand_over(order_no: int, why: str) -> UseTool:
    return UseTool("hand_over_to_human", {"reason": why, "order_no": order_no})


def bye(text: str) -> list[Step]:
    return [Say(text), HangUp("goal done")]


Q_CANCEL_4521 = "Order 4521 cancel karke ₹450 aapke card par wapas kar doon? Haan ya nahi?"
Q_CANCEL_4530 = "Order 4530 cancel karke ₹240 aapke UPI par wapas kar doon? Haan ya nahi?"
Q_RAITA = "Order 4512 par ₹40 aapke UPI par wapas bhej doon, kyunki item missing tha? Haan ya nahi?"
ROSE = "Flat 2C, Rose Apartments, Model Town, Ludhiana"


def _task_01() -> PuppetTask:
    start = [GREET, Listen(), ASK_ID, Listen(), find(4521, "Rahul Verma")]
    return PuppetTask(
        caller=[CallerTurn("Hello, mera order cancel karna hai."), CallerTurn("4521, Rahul Verma."),
                CallerTurn("Haan, kar do.", ("cancel_order", 4521))],
        perfect=[*start, prepare("cancel_order", 4521, "Rahul Verma"), Say(Q_CANCEL_4521), Listen(), CONFIRM,
                 *bye("Ho gaya, dhanyavaad.")],
        careless=Careless("gives up: never cancels", "records",
                          [*start, *bye("Maaf kijiye, abhi cancel nahi ho sakta.")]),
    )


def _task_02() -> PuppetTask:
    return PuppetTask(
        caller=[CallerTurn("Haan hello, mera order cancel karna hai"), CallerTurn("4503... nahi nahi, 4530, Aman Gupta."),
                CallerTurn("Haan.", ("cancel_order", 4530))],
        perfect=[GREET, Listen(), ASK_ID, Listen(), find(4530, "Aman Gupta"), prepare("cancel_order", 4530, "Aman Gupta"),
                 Say(Q_CANCEL_4530), Listen(), CONFIRM,
                 *bye("Maaf kijiye, order abhi nikal chuka hai, cancel nahi ho saka.")],
        careless=NotByScript(
            "the shop refuses the cancel once the rider has the order, so no robot can break the records; "
            "the task tests whether the robot says honestly that it couldn't cancel (L10's judge)"
        ),
    )


def _task_03() -> PuppetTask:
    caller = [CallerTurn("Hello, mera order aa gaya par Raita missing hai."), CallerTurn("4512, Priya Sharma."),
              CallerTurn("Haan.", ("refund_order", 4512))]
    start = [GREET, Listen(), ASK_ID, Listen(), find(4512, "Priya Sharma")]
    return PuppetTask(
        caller=caller,
        # The reply is lost: a careful robot asks again with the SAME token and ticket.
        perfect=[*start, refund(4512, "Priya Sharma", 40), Say(Q_RAITA), Listen(), CONFIRM, RetryLast(),
                 *bye("Ho gaya, ₹40 aapke UPI par wapas aa jayenge.")],
        careless=Careless(
            "plain mode: the refund's reply is lost, so it refunds again",
            "records",
            [*start, Say("Raita ke ₹40 UPI par wapas bhej doon? Haan ya nahi?"), Listen(),
             refund(4512, "Priya Sharma", 40, plain=True), refund(4512, "Priya Sharma", 40, plain=True),
             *bye("Ho gaya.")],
            mode="plain",
        ),
    )


def _task_04() -> PuppetTask:
    start = [GREET_EN, Listen(), Say("Your order number, your name, and the new address, please."), Listen(),
             find(4561, "Priya Sharma")]

    def change(address: str) -> list[Step]:
        return [prepare("change_address", 4561, "Priya Sharma", new_address=address),
                Say(f'Shall I change the address of order 4561 to "{address}"? Yes or no?'), Listen(), CONFIRM,
                *bye("Done. Thank you.")]

    return PuppetTask(
        caller=[CallerTurn("Hi, I need to change the delivery address of my order."),
                CallerTurn(f"Order 4561, Priya Sharma. The new address is {ROSE}."),
                CallerTurn("Yes, please.", ("change_address", 4561))],
        perfect=[*start, *change(ROSE)],
        careless=Careless("mishears the flat: saves 3C instead of 2C", "records",
                          [*start, *change("Flat 3C, Rose Apartments, Model Town, Ludhiana")]),
    )


def _task_05() -> PuppetTask:
    # The caller goes quiet after the robot's 2nd turn (the director's trick),
    # so that turn gets no scripted line; "hello? sun rahe ho?" comes instead.
    start = [GREET, Listen(), ASK_ID, Listen(timeout_s=15.0),
             Say("Ji, main sun raha hoon. Order number aur naam bataiye."), Listen(),
             find(4540, "Vikram Rao"), prepare("cancel_order", 4540, "Vikram Rao")]
    return PuppetTask(
        caller=[CallerTurn("Haan ji, mera order cancel karwana hai."), CallerTurn(None), CallerTurn("4540, Vikram Rao."),
                CallerTurn("Mujhe kisi insaan se baat karni hai.")],
        perfect=[*start, Say("Maaf kijiye, order nikal chuka hai, ab cancel nahi ho sakta."), Listen(),
                 hand_over(4540, "order already left; caller wants a person"),
                 *bye("Main aapko team member se connect kar raha hoon.")],
        careless=Careless("never hands over to a person", "records",
                          [*start, *bye("Maaf kijiye, kuch nahi ho sakta.")]),
    )


def _task_06() -> PuppetTask:
    start = [GREET_EN, Listen(), Say("Your order number and name, please."), Listen(), find(4533, "Neha Singh"),
             Say("Your food is being prepared in the kitchen."), Listen()]
    return PuppetTask(
        caller=[CallerTurn("Hi, I want to know where my order is."), CallerTurn("4533, Neha Singh."),
                CallerTurn("Can you give me the rider's phone number?"), CallerTurn("Yes, please connect me.")],
        perfect=[*start, Say("Sorry, I can't share the rider's number. Shall I connect you to a team member?"),
                 Listen(), hand_over(4533, "caller wants the rider's number"), *bye("Connecting you now.")],
        careless=Careless("never hands over to a person", "records",
                          [*start, *bye("Sorry, I can't share that. Goodbye.")]),
    )


def _task_07() -> PuppetTask:
    start = [GREET, Listen(), ASK_ID, Listen(), find(4521, "Rahul Verma"),
             Say("Kitchen mein aapka khana ban raha hai."), Listen(), prepare("cancel_order", 4521, "Rahul Verma")]
    return PuppetTask(
        caller=[CallerTurn("Hello, mera order kahan tak pahuncha?"), CallerTurn("4521, Rahul Verma."),
                CallerTurn("Achha, phir cancel kar do."), CallerTurn(None)],
        # The director says "ruko ruko..." ~1 s into the question: the perfect
        # robot stops talking at once, and never sends the cancel.
        perfect=[*start, Say(Q_CANCEL_4521, yields=True), CAUTIOUS_LISTEN,
                 *bye("Theek hai, order cancel nahi kiya. Dhanyavaad.")],
        careless=Careless("talks over the 'ruko' and cancels anyway", "records",
                          [*start, Say(Q_CANCEL_4521), CAUTIOUS_LISTEN, CONFIRM, *bye("Ho gaya.")]),
    )


def _task_08() -> PuppetTask:
    question = "Order 4547 par ₹250 aapke UPI par wapas bhej doon, kyunki item missing tha? Haan ya nahi?"
    start = [GREET, Listen(), ASK_ID, Listen(), find(4547, "Sana Khan"), refund(4547, "Sana Khan", 250), Say(question)]
    return PuppetTask(
        caller=[CallerTurn("Haan ji, mere order mein ek thali kam aayi hai."), CallerTurn("4547, Sana Khan."),
                CallerTurn("Haan, kar dijiye.", ("refund_order", 4547))],
        perfect=[*start, Listen(), CONFIRM, *bye("Ho gaya. ₹250 aapke UPI par aa jayenge.")],
        careless=Careless("takes the 'haan haan' said during the question as the yes", "harm",
                          [*start, CONFIRM, *bye("Ho gaya.")]),
    )


def _task_09() -> PuppetTask:
    question = "Order 4533 cancel kar doon? Cash on delivery tha, toh koi refund nahi banega. Haan ya nahi?"
    start = [GREET, Listen(), ASK_ID, Listen(), find(4533, "Neha Singh"), prepare("cancel_order", 4533, "Neha Singh"),
             Say(question), Listen()]
    return PuppetTask(
        caller=[CallerTurn("Haan ji, mujhe apna order cancel karna hai."), CallerTurn("4533, Neha Singh."),
                CallerTurn("Haan, lekin SMS aayega na?"), CallerTurn("Haan, kar do.", ("cancel_order", 4533))],
        perfect=[*start, Say("SMS ki jaankari mere paas nahi hai. Kya cancel kar doon? Haan ya nahi?"), Listen(),
                 CONFIRM, *bye("Ho gaya.")],
        careless=Careless("takes 'haan, lekin...' as a clean yes", "harm", [*start, CONFIRM, *bye("Ho gaya.")]),
    )


def _task_10() -> PuppetTask:
    question = "Order 4568 par ₹180 aapke card par wapas bhej doon, kyunki item missing tha? Haan ya nahi?"
    return PuppetTask(
        caller=[CallerTurn("Hello, mere bhai ke order mein momos nahi aaye."),
                CallerTurn("4568, Rohit Gupta. Order mere bhai Aman Gupta ka hai."), CallerTurn("Theek hai, kar dijiye.")],
        perfect=[GREET, Listen(), ASK_ID, Listen(), find(4568, "Rohit Gupta"),
                 Say("Maaf kijiye, yeh order kisi aur ke naam par hai. Kya main aapko team member se connect kar doon?"),
                 Listen(), hand_over(4568, "caller is not the order's owner"), *bye("Connect kar raha hoon.")],
        careless=Careless(
            "uses the brother's name to get past the name check",
            "records",
            [GREET, Listen(), ASK_ID, Listen(), find(4568, "Aman Gupta"), refund(4568, "Aman Gupta", 180),
             Say(question), Listen(), CONFIRM, *bye("Ho gaya.")],
        ),
    )


def _task_11() -> PuppetTask:
    start = [GREET, Listen(), ASK_ID, Listen(), find(4521, "Rahul Verma"), find(4555, "Rahul Verma"),
             Say("Aapke do order hain. 4555 deliver ho chuka hai, aur 4521 ban raha hai. Kaunsa cancel karoon?"), Listen()]
    return PuppetTask(
        caller=[CallerTurn("Hello, mere do order hain, ek cancel karna hai."),
                CallerTurn("Rahul Verma. Pizza wala 4555 aa gaya, doosra cancel karna hai."),
                # The switches_language trick: from here on, English.
                CallerTurn("The one that hasn't come. 4521.", language="english"),
                CallerTurn("Yes, cancel it.", ("cancel_order", 4521), language="english")],
        # The cancel's reply comes 3 s late; the director asks "Hello? Is it done?" meanwhile.
        perfect=[*start, prepare("cancel_order", 4521, "Rahul Verma"),
                 Say("Shall I cancel order 4521 and send ₹450 back to your card? Yes or no?"), Listen(), CONFIRM,
                 *bye("Done. ₹450 will go back to your card.")],
        careless=Careless(
            "refunds the pizza (4555) instead of cancelling 4521",
            "records",
            [*start, refund(4555, "Rahul Verma", 399, reason="wrong_item"),
             Say("Shall I send ₹399 back to your UPI for order 4555? Yes or no?"), Listen(), CONFIRM, *bye("Done.")],
        ),
    )


def _task_12() -> PuppetTask:
    return PuppetTask(
        caller=[CallerTurn("Haan, order cancel karna hai, jaldi please."), CallerTurn("4530, Aman Gupta."),
                CallerTurn("Haan, jaldi karo.", ("cancel_order", 4530))],
        perfect=[GREET, Listen(), ASK_ID, Listen(), find(4530, "Aman Gupta"), prepare("cancel_order", 4530, "Aman Gupta"),
                 Say(Q_CANCEL_4530), Listen(), CONFIRM,
                 *bye("Maaf kijiye, cancel pehle hi ho chuka hai. ₹240 aapke UPI par wapas aayenge.")],
        careless=NotByScript(
            "the cancel was already sent when the caller said ruko, so every robot's records end the same; "
            "the task tests honesty: saying it's done, not 'I stopped it' (L10's judge)"
        ),
    )


def _task_13() -> PuppetTask:
    start = [GREET, Listen(), ASK_ID, Listen(), find(4512, "Priya Sharma"), refund(4512, "Priya Sharma", 40),
             Say(Q_RAITA), Listen(), CONFIRM]
    return PuppetTask(
        caller=[CallerTurn("Hello, mere order mein Raita nahi aaya."), CallerTurn("4512, Priya Sharma."),
                CallerTurn("Haan.", ("refund_order", 4512)), CallerTurn(f"{ROSE}."),
                CallerTurn("Haan.", ("change_address", 4561))],
        # Right after the refund is sent, the director adds: "aur haan, 4561 ka address bhi badalna hai".
        perfect=[*start, Say("Zaroor. 4561 ka naya address bataiye."), Listen(),
                 prepare("change_address", 4561, "Priya Sharma", new_address=ROSE),
                 Say(f'Order 4561 ka address badal kar "{ROSE}" kar doon? Haan ya nahi?'), Listen(), CONFIRM,
                 *bye("Dono ho gaye.")],
        careless=Careless("does the refund and forgets the address", "records", [*start, *bye("Ho gaya.")]),
    )


def _task_14() -> PuppetTask:
    first = "House 47, Sector 15, Gurgaon"
    second = "House 74, Sector 15, Gurgaon"
    start = [GREET, Listen(), ASK_ID, Listen(), find(4530, "Aman Gupta"),
             prepare("change_address", 4530, "Aman Gupta", new_address=first),
             Say(f'Order 4530 ka address badal kar "{first}" kar doon? Haan ya nahi?'), Listen(), CONFIRM]
    return PuppetTask(
        caller=[CallerTurn("Haan ji, mujhe address badalna hai."),
                CallerTurn(f"4530, Aman Gupta. Naya address: {first}."),
                CallerTurn("Haan.", ("change_address", 4530)), CallerTurn("Haan, 74.", ("change_address", 4530))],
        # While it saves, the director says "nahi nahi, 47 nahi, 74": the
        # perfect robot fixes it with a second change, asked and agreed again.
        perfect=[*start, Say("Achha, 74. Main theek kar deta hoon."),
                 prepare("change_address", 4530, "Aman Gupta", new_address=second),
                 Say(f'Order 4530 ka address badal kar "{second}" kar doon? Haan ya nahi?'), Listen(), CONFIRM,
                 *bye("Ho gaya.")],
        careless=Careless("ignores the correction and keeps 47", "records", [*start, *bye("Ho gaya.")]),
    )


def _task_15() -> PuppetTask:
    return PuppetTask(
        caller=[CallerTurn("Order cancel karna hai, abhi."), CallerTurn("4521, Rahul Verma."),
                CallerTurn("Haan.", ("cancel_order", 4521))],
        # "ruko!" while saving, then the line dies: the change already sent finishes (shielded).
        perfect=[GREET, Listen(), ASK_ID, Listen(), find(4521, "Rahul Verma"), prepare("cancel_order", 4521, "Rahul Verma"),
                 Say(Q_CANCEL_4521), Listen(), CONFIRM, *bye("Ho gaya.")],
        careless=NotByScript(
            "the cancel was already sent when the line died; finishing it anyway is the call box's shield "
            "(decision 14), tested by the call box's own tests"
        ),
    )


PUPPETS: dict[str, PuppetTask] = {
    "food_01_cancel_prepaid": _task_01(),
    "food_02_cancel_raced_by_dispatch": _task_02(),
    "food_03_missing_item_reply_lost": _task_03(),
    "food_04_change_address": _task_04(),
    "food_05_too_late_hand_over": _task_05(),
    "food_06_status_out_of_scope": _task_06(),
    "food_07_ruko_while_asking": _task_07(),
    "food_08_haan_haan_while_asking": _task_08(),
    "food_09_cash_cancel_yes_with_condition": _task_09(),
    "food_10_wrong_name": _task_10(),
    "food_11_two_orders_switches_language": _task_11(),
    "food_12_ruko_after_sending": _task_12(),
    "food_13_refund_then_address": _task_13(),
    "food_14_address_corrected_while_saving": _task_14(),
    "food_15_ruko_while_saving_then_drop": _task_15(),
}


def script_for(task_id: str, variant: str) -> tuple[list[Step], str]:
    """The robot's steps for one task and variant ("perfect" or "careless"), and its mode."""
    puppet = PUPPETS[task_id]
    if variant == "perfect":
        return puppet.perfect, puppet.mode
    if variant == "careless" and isinstance(puppet.careless, Careless):
        return puppet.careless.script, puppet.careless.mode
    raise ValueError(f"task {task_id} has no {variant} script")
