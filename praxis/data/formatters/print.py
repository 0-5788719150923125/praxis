"""Synthetic `print` format: the model leads with a question and predicts the answer.

Each example is `prefix + question`, where the prefix is drawn from a wildly
varied pool so the same question/answer core is asked in radically different
contexts. Answers are short and structured; the engagement reward fires on
mention (recall), not exact match. The ~25 formats span math, geometry, physics,
science, geography, language, prose, logic, and everyday/social reasoning - the
"kinds" of choices we ask the model to anticipate.
"""

import random
from typing import Callable, Dict, List

from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, sample_developer_prompt

# Varied context snippets. The prefix colors the context the question is asked
# in; it is never the thing being answered. Kept short and cross-domain.
PREFIX_POOL: List[str] = [
    "The rain had not let up for three days, and the river was rising.",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "In 1492 a small fleet set out west across an ocean no one had charted.",
    "She set the kettle on the stove and waited for the whistle.",
    "The mitochondria is often called the powerhouse of the cell.",
    "Markets fell sharply today amid concerns over rising interest rates.",
    '"Are you coming to the festival?" he asked, lacing up his boots.',
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
    "To be, or not to be, that is the question.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Gravity bends the path of light around very massive objects.",
    "git commit -m 'fix the off-by-one in the parser'",
    "The old map showed a road that no longer existed.",
    "Thunder rolled across the valley as the storm drew near.",
    "A balanced diet includes proteins, carbohydrates, and healthy fats.",
    "",  # sometimes no prefix: the model simply leads with its question
]


def _arithmetic() -> Dict[str, str]:
    op = random.choice(["+", "-", "×"])
    if op == "+":
        a, b = random.randint(1, 50), random.randint(1, 50)
        ans = a + b
    elif op == "-":
        a = random.randint(10, 100)
        b = random.randint(1, a)
        ans = a - b
    else:
        a, b = random.randint(2, 12), random.randint(2, 12)
        ans = a * b
    return {"question": f"What is {a} {op} {b}?", "answer": str(ans)}


def _counting() -> Dict[str, str]:
    n = random.randint(3, 10)
    item = random.choice(["apples", "books", "cats", "marbles", "coins"])
    return {"question": f"If I have {n} {item}, how many do I have?", "answer": str(n)}


def _comparison() -> Dict[str, str]:
    a, b = random.randint(1, 100), random.randint(1, 100)
    while a == b:
        b = random.randint(1, 100)
    if random.random() < 0.5:
        return {"question": f"Which is larger: {a} or {b}?", "answer": str(max(a, b))}
    return {"question": f"Which is smaller: {a} or {b}?", "answer": str(min(a, b))}


def _sequence() -> Dict[str, str]:
    start, step = random.randint(1, 9), random.randint(2, 5)
    terms = [start + step * i for i in range(3)]
    nxt = start + step * 3
    seq = ", ".join(map(str, terms))
    return {"question": f"What number comes next: {seq}, ...?", "answer": str(nxt)}


def _rectangle_area() -> Dict[str, str]:
    w, h = random.randint(2, 20), random.randint(2, 20)
    return {
        "question": f"A rectangle is {w} by {h}. What is its area?",
        "answer": str(w * h),
    }


def _polygon_sides() -> Dict[str, str]:
    poly = random.choice(
        [
            ("triangle", 3),
            ("square", 4),
            ("pentagon", 5),
            ("hexagon", 6),
            ("octagon", 8),
        ]
    )
    return {
        "question": f"How many sides does a {poly[0]} have?",
        "answer": str(poly[1]),
    }


def _spatial() -> Dict[str, str]:
    turns = {
        ("north", "right"): "east",
        ("north", "left"): "west",
        ("east", "right"): "south",
        ("south", "left"): "east",
        ("west", "right"): "north",
    }
    (facing, turn), result = random.choice(list(turns.items()))
    return {
        "question": f"If you face {facing} and turn {turn}, which way do you face?",
        "answer": result,
    }


def _speed() -> Dict[str, str]:
    v, t = random.randint(2, 20), random.randint(2, 10)
    d = v * t
    return {
        "question": f"An object travels {d} meters in {t} seconds. What is its speed in m/s?",
        "answer": str(v),
    }


def _physics_fact() -> Dict[str, str]:
    facts = [
        ("What force pulls objects toward Earth?", "gravity"),
        ("What is the speed of light approximately, in km per second?", "300000"),
        ("What state of matter is water vapor?", "gas"),
        ("What do we call stored energy due to position?", "potential"),
    ]
    q, a = random.choice(facts)
    return {"question": q, "answer": a}


def _chemistry() -> Dict[str, str]:
    elems = [
        ("oxygen", "O"),
        ("hydrogen", "H"),
        ("carbon", "C"),
        ("sodium", "Na"),
        ("iron", "Fe"),
    ]
    name, sym = random.choice(elems)
    return {"question": f"What is the chemical symbol for {name}?", "answer": sym}


def _astronomy() -> Dict[str, str]:
    facts = [
        ("Which planet is closest to the Sun?", "Mercury"),
        ("Which planet is known as the red planet?", "Mars"),
        ("What is at the center of our solar system?", "Sun"),
        ("What is Earth's only natural satellite?", "Moon"),
    ]
    q, a = random.choice(facts)
    return {"question": q, "answer": a}


def _biology() -> Dict[str, str]:
    animals = [("spider", "8"), ("insect", "6"), ("dog", "4"), ("bird", "2")]
    name, legs = random.choice(animals)
    return {"question": f"How many legs does a {name} have?", "answer": legs}


def _capital() -> Dict[str, str]:
    pairs = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Egypt", "Cairo"),
        ("Brazil", "Brasilia"),
        ("Canada", "Ottawa"),
    ]
    country, city = random.choice(pairs)
    return {"question": f"What is the capital of {country}?", "answer": city}


def _continent() -> Dict[str, str]:
    pairs = [
        ("Egypt", "Africa"),
        ("Japan", "Asia"),
        ("France", "Europe"),
        ("Brazil", "South America"),
        ("Australia", "Oceania"),
    ]
    country, cont = random.choice(pairs)
    return {"question": f"On which continent is {country}?", "answer": cont}


def _opposite() -> Dict[str, str]:
    pairs = [
        ("hot", "cold"),
        ("up", "down"),
        ("fast", "slow"),
        ("happy", "sad"),
        ("big", "small"),
    ]
    word, opp = random.choice(pairs)
    return {"question": f"What is the opposite of '{word}'?", "answer": opp}


def _plural() -> Dict[str, str]:
    pairs = [
        ("child", "children"),
        ("mouse", "mice"),
        ("foot", "feet"),
        ("cat", "cats"),
    ]
    word, plural = random.choice(pairs)
    return {"question": f"What is the plural of '{word}'?", "answer": plural}


def _past_tense() -> Dict[str, str]:
    pairs = [
        ("go", "went"),
        ("run", "ran"),
        ("eat", "ate"),
        ("swim", "swam"),
        ("walk", "walked"),
    ]
    verb, past = random.choice(pairs)
    return {"question": f"What is the past tense of '{verb}'?", "answer": past}


def _synonym() -> Dict[str, str]:
    pairs = [
        ("big", "large"),
        ("happy", "glad"),
        ("smart", "clever"),
        ("quick", "fast"),
    ]
    word, syn = random.choice(pairs)
    return {"question": f"Give a synonym for '{word}'.", "answer": syn}


def _rhyme() -> Dict[str, str]:
    pairs = [("cat", "hat"), ("light", "night"), ("blue", "true"), ("star", "far")]
    word, rhyme = random.choice(pairs)
    return {"question": f"What is a word that rhymes with '{word}'?", "answer": rhyme}


def _color_mix() -> Dict[str, str]:
    mixes = [
        ("blue and yellow", "green"),
        ("red and blue", "purple"),
        ("red and yellow", "orange"),
        ("black and white", "gray"),
    ]
    pair, result = random.choice(mixes)
    return {"question": f"What color do you get mixing {pair}?", "answer": result}


def _calendar() -> Dict[str, str]:
    months = [("January", "31"), ("April", "30"), ("December", "31"), ("June", "30")]
    month, days = random.choice(months)
    return {"question": f"How many days are in {month}?", "answer": days}


def _hours() -> Dict[str, str]:
    n = random.randint(1, 7)
    return {"question": f"How many hours are in {n} days?", "answer": str(n * 24)}


def _units_length() -> Dict[str, str]:
    n = random.randint(1, 20)
    return {
        "question": f"How many centimeters are in {n} meters?",
        "answer": str(n * 100),
    }


def _temperature() -> Dict[str, str]:
    facts = [
        ("Water freezes at what temperature in Celsius?", "0"),
        ("Water boils at what temperature in Celsius?", "100"),
    ]
    q, a = random.choice(facts)
    return {"question": q, "answer": a}


def _social() -> Dict[str, str]:
    pairs = [
        ("If someone says 'thank you', what is a polite reply?", "you're welcome"),
        ("What do you say when you meet someone for the first time?", "hello"),
        ("What do you say before going to sleep?", "good night"),
    ]
    q, a = random.choice(pairs)
    return {"question": q, "answer": a}


# The ~25 broad-spectrum "kinds" of question we ask the model to anticipate.
PRINT_FORMATS: List[Callable[[], Dict[str, str]]] = [
    _arithmetic,
    _counting,
    _comparison,
    _sequence,
    _rectangle_area,
    _polygon_sides,
    _spatial,
    _speed,
    _physics_fact,
    _chemistry,
    _astronomy,
    _biology,
    _capital,
    _continent,
    _opposite,
    _plural,
    _past_tense,
    _synonym,
    _rhyme,
    _color_mix,
    _calendar,
    _hours,
    _units_length,
    _temperature,
    _social,
]


def format_print(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict:
    """Build one synthetic `print` example: varied prefix, then the model leads
    with a question and its predicted short answer.

    The prefix lives in a leading context turn (untrained color); the assistant
    turn carries the question + predicted answer (the trained target, and the
    `A_hat` span the engagement reward later scores against the user response).
    """
    fmt = random.choice(PRINT_FORMATS)
    core = fmt()
    question, answer = core["question"], core["answer"]
    prefix = random.choice(PREFIX_POOL)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "developer",
            "content": sample_developer_prompt("engage_conversation"),
        },
    ]
    if prefix:
        messages.append({"role": "user", "content": prefix})
    messages.append({"role": "assistant", "content": f"{question}\n{answer}"})

    return {
        "messages": messages,
        "metadata": {
            "format": "print",
            "question": question,
            "ground_truth": answer,
            "category": fmt.__name__.lstrip("_"),
        },
    }


def generate_print_samples(n_per_format: int = 4):
    """Materialize default data: for each of the 25 formats, ``n_per_format``
    random (prefix-modulated) instances. Returns a list of dicts with category,
    prefix, question, and answer - a tangible corpus to inspect or score the
    engagement reward against without a model.
    """
    samples = []
    for fmt in PRINT_FORMATS:
        category = fmt.__name__.lstrip("_")
        for _ in range(n_per_format):
            core = fmt()
            samples.append(
                {
                    "category": category,
                    "prefix": random.choice(PREFIX_POOL),
                    "question": core["question"],
                    "answer": core["answer"],
                }
            )
    return samples
