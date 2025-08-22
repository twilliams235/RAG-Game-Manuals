from query import query
from langchain_ollama import OllamaLLM

# eval_utils.py
import re

YES = {"yes", "y", "true"}
NO = {"no", "n", "false"}

_NUM_WORDS = {
    "zero": "0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _normalize_numbers(text: str) -> str:
    t = _norm(text)
    for w, d in _NUM_WORDS.items():
        t = re.sub(rf"\b{w}\b", d, t)
    return t

def _first_token(s: str) -> str:
    s = _norm(s)
    return s.split()[0] if s else ""

def matches_expected(expected: str, actual: str) -> bool:
    e = _normalize_numbers(expected)
    a = _normalize_numbers(actual)

    # yes/no expectations
    if e in YES | NO:
        a0 = _first_token(a)
        return (e in YES and a0 in YES) or (e in NO and a0 in NO)

    # if expected contains a number, compare the first number
    e_nums = re.findall(r"\d+", e)
    a_nums = re.findall(r"\d+", a)
    if e_nums:
        return bool(a_nums) and a_nums[0] == e_nums[0]

    # fallback: substring match
    return e in a



def test_catan_rules():
    assert query_and_validate(
        question="What roll activates the robber in Catan?",
        expected_response="7",
    )

# Provide negative example
def test_neg_catan_rules():
    assert not query_and_validate(
        question="How many VP are roads worth in catan?",
        expected_response="2",
    )


def test_codename_rules():
    assert query_and_validate(
        question="How many words does the starting team guess in codenames?",
        expected_response="9",
    )

def test_neg_codename_rules():
    assert not query_and_validate(
        question="Are letters and numbers valid clues if they refer to meanings?",
        expected_response="No",
    )



def query_and_validate(question: str, expected_response: str):
    actual = query(question)
    ok = matches_expected(expected_response, actual)
    print(f"Expected: {expected_response}\nActual: {actual}\nMatch: {ok}")
    return ok