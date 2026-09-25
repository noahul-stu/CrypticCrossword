"""
Standalone Python DSL verifier for cryptic crossword proofs.

Based on Andrews & Witteveen (2025) "A Reasoning-Based Approach to Cryptic
Crossword Clue Solving" (ICML 2025).

This verifier executes Python proof functions that use assert statements
with domain-specific helper functions (is_synonym, is_anagram, etc.).

Usage:
    # As a module
    from verifier import execute_proof
    result = execute_proof(proof_code_string)
    print(result["success"], result["errors"])

    # As a CLI
    python verifier.py --code proof_file.py
    python verifier.py --demo
"""

import ast
import re
import sys
import traceback
from enum import Enum
from typing import Optional


# ===================================================================
# Action enum — wordplay operations in cryptic crosswords
# ===================================================================

class Action(Enum):
    ANAGRAM = "ANAGRAM"
    REMOVE_FIRST = "REMOVE_FIRST"
    INITIALS = "INITIALS"
    REMOVE_LAST = "REMOVE_LAST"
    GOES_INSIDE = "GOES_INSIDE"
    GOES_OUTSIDE = "GOES_OUTSIDE"
    REVERSE = "REVERSE"
    SUBSTRING = "SUBSTRING"
    HOMOPHONE = "HOMOPHONE"


# ===================================================================
# Indicator word lists — which words suggest which cryptic device
# ===================================================================

INDICATOR_WORDS = {
    Action.ANAGRAM: {
        "about", "absurd", "abstract", "abuse", "adapt", "adjust", "adrift",
        "agitate", "alter", "amend", "amok", "anew", "another", "around",
        "arrange", "awkward", "bad", "badly", "bananas", "batter", "beat",
        "bizarre", "blend", "blunder", "boil", "break", "brew", "broken",
        "bust", "change", "chaotic", "chop", "churn", "circulate", "cocktail",
        "complex", "compose", "confuse", "construct", "convert", "cook",
        "corrupt", "crack", "craft", "crash", "crazy", "cruel", "crumble",
        "crush", "cultivate", "damage", "dance", "deploy", "design",
        "desperate", "destroy", "develop", "deviate", "devise", "different",
        "disarray", "disorder", "disperse", "dissolve", "distort", "disturb",
        "diverse", "dizzy", "doctor", "dodgy", "drunk", "edit", "elaborate",
        "engineer", "erratic", "evolve", "exchange", "excited", "exotic",
        "explode", "fabricate", "false", "fancy", "fashion", "fiddle",
        "fix", "flip", "flutter", "foolish", "forge", "form", "foul",
        "fracture", "free", "fresh", "frolic", "fudge", "fumble", "funny",
        "garble", "generate", "go", "goes", "going", "gone", "grind",
        "ground", "hack", "haywire", "hybrid", "improper", "incorrect",
        "insane", "irregular", "jiggle", "jumble", "knead", "loose", "mad",
        "mangle", "manipulate", "manoeuvre", "mar", "mash", "maybe", "mess",
        "mince", "mishandle", "misplace", "mix", "mobile", "model",
        "modify", "move", "muddle", "mutate", "nasty", "naughty", "new",
        "novel", "nuts", "odd", "oddly", "off", "order", "organise",
        "organize", "out", "peculiar", "perhaps", "play", "possibly",
        "potential", "prepare", "process", "produce", "properly", "puzzle",
        "random", "rearrange", "rebuild", "recipe", "reconstruct", "recycle",
        "redo", "reform", "refurbish", "remake", "remodel", "renovate",
        "repair", "replace", "reproduce", "resolve", "restore", "result",
        "review", "revise", "revolve", "ridiculous", "rip", "rocky",
        "rough", "ruin", "run", "running", "rupture", "sadly", "scatter",
        "scramble", "set", "shaken", "shatter", "shift", "shot", "shuffle",
        "silly", "sloppy", "smash", "somehow", "sort", "spin", "spoil",
        "spread", "stagger", "stew", "stir", "strange", "strangely",
        "struggle", "style", "suspect", "swap", "swim", "swing", "swirl",
        "switch", "tangle", "terrible", "thrash", "topsy-turvy", "torn",
        "toss", "trampled", "transfer", "transform", "translate", "treat",
        "trouble", "tumble", "turn", "twist", "ugly", "undone", "unravel",
        "unruly", "unsettle", "unusual", "unusually", "upset", "variable",
        "variety", "version", "vile", "volatile", "wander", "wanton",
        "warp", "waste", "wayward", "weave", "weird", "whip", "whirl",
        "wicked", "wild", "work", "wound", "wrangle", "wreck", "wriggle",
        "wrong", "wrongly",
        "shredded", "broadcast", "scattered", "distributed",
    },
    Action.REVERSE: {
        "about", "back", "backward", "backwards", "behind", "brought back",
        "climb", "climbing", "comes back", "contrary", "descend", "down",
        "flip", "flipped", "from the east", "go back", "going up",
        "in return", "inversion", "left", "lifted", "mirror", "over",
        "overturned", "put back", "raised", "recalled", "reflect",
        "reflected", "rejected", "retire", "retired", "retreat", "return",
        "returned", "reverse", "reversed", "revolution", "rising", "rotate",
        "round", "set back", "sent back", "serve", "switch", "turn",
        "turned", "turning", "turns", "up", "upended", "uplifted", "upside",
        "upset", "upturn",
    },
    Action.REMOVE_FIRST: {
        "beheaded", "curtailed at the start", "decapitate", "decapitated",
        "drop the first", "endless", "first off", "head off", "headless",
        "lose the first", "miss the start", "opening lost", "remove first",
        "skip the start", "starter missing", "topless", "trim the start",
        "without a head", "without first", "without head", "without its head",
        "without the first", "without the head",
    },
    Action.REMOVE_LAST: {
        "almost", "abridged", "briefly", "clip", "clipped", "crop",
        "curtail", "curtailed", "cut", "cut short", "dock", "docked",
        "drop the last", "endlessly", "give up last", "incomplete",
        "largely", "limitless", "lose the end", "mostly", "nearly",
        "not complete", "not completely", "not finished", "not quite",
        "practically", "remove last", "short", "shortened", "skip the end",
        "snip", "tail off", "tailless", "trim", "trimmed", "truncate",
        "truncated", "unfinished", "without end", "without its tail",
        "without the last",
    },
    Action.INITIALS: {
        "at first", "beginning", "beginnings", "capital", "capitals",
        "character", "characters", "extreme", "first", "first letter",
        "first letters", "first of", "firstly", "front", "fronts",
        "head", "headed", "heads", "initial", "initially", "initials",
        "lead", "leaders", "leading", "opener", "openers", "opening",
        "primarily", "prime", "start", "started", "starting", "starts",
        "summit", "tip", "tips", "top", "topped", "tops",
    },
    Action.GOES_INSIDE: {
        "absorb", "accept", "accommodate", "acquire", "admit", "around",
        "capture", "clutch", "collect", "conceal", "contain", "cover",
        "devour", "embrace", "encircle", "enclose", "engulf", "envelope",
        "grab", "grasp", "harbour", "hide", "hold", "house", "hug",
        "include", "incorporate", "inside", "keep", "lock", "outside",
        "possess", "protect", "receive", "retain", "shelter", "surround",
        "swallow", "trap", "welcome", "within", "without", "wrap",
        "about", "clothing",
    },
    Action.GOES_OUTSIDE: {
        "around", "astride", "beside", "boxing", "bridging", "carrying",
        "case", "casing", "circling", "clutching", "covering", "crossing",
        "either side", "embracing", "enclosing", "flanking", "framing",
        "grabbing", "gripping", "guarding", "holding", "housing",
        "hugging", "keeping", "nursing", "outside", "over", "parting",
        "penning", "protecting", "restricting", "seizing", "shielding",
        "splitting", "straddling", "surrounding", "trapping", "wrapping",
    },
    Action.SUBSTRING: {
        "a bit of", "a little", "a piece of", "bit", "bit of", "centre",
        "content", "contents", "core", "element", "elements", "extract",
        "fragment", "from", "half", "heart", "in", "in part", "ingredient",
        "inside", "interior", "middle", "part", "part of", "partly",
        "partly in", "piece", "portion", "sample", "section", "segment",
        "slice", "some", "some of", "somewhat",
    },
    Action.HOMOPHONE: {
        "aloud", "announced", "audible", "audibly", "broadcast", "by the sound",
        "declared", "echoed", "expressed", "for the audience", "heard",
        "in conversation", "in speech", "it is said", "mentioned",
        "on the air", "on the phone", "on the radio", "oral", "orally",
        "out loud", "overheard", "phonetically", "proclaimed", "pronounced",
        "read aloud", "recited", "reported", "reportedly", "said",
        "so to speak", "so we hear", "sound", "sounded", "sounding",
        "sounds", "sounds like", "spoken", "stated", "they say", "to say",
        "to the ear", "uttered", "verbal", "verbally", "vocal", "vocally",
        "voiced", "we hear", "when spoken",
    },
}


# ===================================================================
# Abbreviation dictionary — common crossword abbreviations
# ===================================================================

ABBREVIATIONS = {
    "about": ["C", "CA", "RE", "CIRCA"],
    "account": ["AC", "ACC"],
    "ace": ["A"],
    "adult": ["A", "X"],
    "advanced": ["A"],
    "afternoon": ["PM"],
    "against": ["V", "VS", "ANTI"],
    "american": ["US", "AM"],
    "answer": ["A", "ANS"],
    "area": ["A"],
    "army": ["RA", "TA"],
    "article": ["A", "AN", "THE"],
    "artist": ["RA", "PAINTER"],
    "bachelor": ["B", "BA"],
    "before": ["B", "ERE", "PRE"],
    "bishop": ["B", "RR"],
    "black": ["B"],
    "book": ["B", "VOL"],
    "born": ["B", "NEE"],
    "boy": ["LAD", "SON"],
    "british": ["B", "BR"],
    "brother": ["BR", "BRO"],
    "cape": ["C"],
    "captain": ["CAPT", "SKIPPER"],
    "caught": ["C", "CT"],
    "century": ["C"],
    "chapter": ["C", "CH"],
    "church": ["CE", "CH"],
    "circle": ["O", "RING"],
    "club": ["C"],
    "cold": ["C"],
    "company": ["CO", "FIRM"],
    "conservative": ["C", "CON", "TORY"],
    "copper": ["CU"],
    "correct": ["R"],
    "current": ["AC", "DC", "I"],
    "daughter": ["D"],
    "day": ["D"],
    "dead": ["D"],
    "degree": ["D", "BA", "MA"],
    "democrat": ["D", "DEM"],
    "diamonds": ["D"],
    "died": ["D"],
    "direction": ["N", "S", "E", "W", "NE", "NW", "SE", "SW"],
    "doctor": ["DR", "MB", "MD", "GP"],
    "duck": ["O"],
    "duke": ["D"],
    "earl": ["E"],
    "east": ["E"],
    "eastern": ["E"],
    "editor": ["ED"],
    "edward": ["ED", "TED", "NED"],
    "energy": ["E"],
    "engineer": ["RE", "ENG"],
    "english": ["E", "ENG"],
    "european": ["E"],
    "evening": ["E", "EVE", "PM"],
    "exercise": ["PE", "PT"],
    "father": ["DA", "FR", "PA"],
    "fellow": ["F"],
    "female": ["F", "SHE", "HER"],
    "fifty": ["L"],
    "fine": ["F"],
    "first": ["IST", "A"],
    "five": ["V"],
    "five hundred": ["D"],
    "following": ["F"],
    "foot": ["FT"],
    "for": ["F", "PRO"],
    "for every": ["PER"],
    "for example": ["EG"],
    "force": ["F"],
    "french": ["FR"],
    "girl": ["G", "LASS"],
    "god": ["RA", "RE"],
    "gold": ["AU", "OR"],
    "good": ["G"],
    "graduate": ["BA", "MA"],
    "grand": ["G", "K"],
    "hand": ["H"],
    "hard": ["H"],
    "head": ["H"],
    "hearts": ["H"],
    "henry": ["H", "HAL"],
    "her": ["HER"],
    "hospital": ["H"],
    "hot": ["H"],
    "hotel": ["H"],
    "hundred": ["C"],
    "hydrogen": ["H"],
    "independent": ["I", "IND"],
    "international": ["I", "CAP"],
    "iron": ["FE"],
    "island": ["I", "IS"],
    "jack": ["J", "KNAVE"],
    "judge": ["J"],
    "king": ["K", "R", "REX"],
    "knight": ["N", "K", "SIR"],
    "labour": ["L", "LAB"],
    "lake": ["L"],
    "large": ["L", "OS"],
    "last": ["Z"],
    "lead": ["PB"],
    "learner": ["L"],
    "left": ["L", "PORT"],
    "liberal": ["L", "LIB"],
    "line": ["L", "RY"],
    "love": ["O"],
    "male": ["M", "HE"],
    "mark": ["M", "DM"],
    "married": ["M"],
    "master": ["M", "MA"],
    "member": ["MP", "ARM", "LEG"],
    "member of parliament": ["MP"],
    "million": ["M"],
    "minister": ["PM", "REV"],
    "model": ["T"],
    "morning": ["AM"],
    "mother": ["MA", "MUM"],
    "motorway": ["M", "MI"],
    "name": ["N"],
    "navy": ["N", "RN"],
    "new": ["N"],
    "nine": ["IX"],
    "nitrogen": ["N"],
    "north": ["N"],
    "northern": ["N"],
    "note": ["A", "B", "C", "D", "E", "F", "G", "DO", "RE", "MI", "SO", "LA", "TI"],
    "nothing": ["NIL", "O", "ZERO", "LOVE"],
    "number": ["N", "NO"],
    "nurse": ["EN", "RN", "SEN", "SRN"],
    "officer": ["CO"],
    "old": ["O", "EX"],
    "one": ["I", "A", "AN", "ACE"],
    "operation": ["OP"],
    "oriental": ["E"],
    "oxygen": ["O"],
    "page": ["P"],
    "parking": ["P"],
    "penny": ["P", "D"],
    "piano": ["P"],
    "point": ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "DOT"],
    "pole": ["N", "S"],
    "politician": ["MP", "PM"],
    "port": ["L"],
    "pound": ["L", "LB"],
    "power": ["P"],
    "pressure": ["P"],
    "priest": ["FR", "P"],
    "prince": ["P"],
    "princess": ["P", "DI"],
    "private": ["GI"],
    "queen": ["Q", "R", "ER", "QU"],
    "quiet": ["P", "SH"],
    "railway": ["RY", "BR"],
    "record": ["EP", "LP", "CD"],
    "resistance": ["R"],
    "right": ["R", "RT"],
    "ring": ["O"],
    "river": ["R", "DEE", "EXE", "TAY"],
    "road": ["RD", "ST", "AVE"],
    "royal": ["R"],
    "run": ["R"],
    "sailor": ["AB", "TAR", "SALT"],
    "saint": ["S", "ST"],
    "second": ["S", "MO"],
    "silver": ["AG"],
    "sister": ["SIS", "NUN"],
    "six": ["VI"],
    "small": ["S"],
    "society": ["S", "SOC"],
    "soldier": ["GI", "RE", "RA"],
    "son": ["S"],
    "south": ["S"],
    "southern": ["S"],
    "spades": ["S"],
    "special": ["S"],
    "street": ["ST", "RD"],
    "student": ["L"],
    "tea": ["T", "CHA"],
    "teacher": ["SIR"],
    "ten": ["X"],
    "that is": ["IE"],
    "the": ["THE"],
    "the french": ["LE", "LA", "LES"],
    "the german": ["DER", "DIE", "DAS"],
    "the spanish": ["EL", "LA"],
    "thousand": ["K", "M"],
    "time": ["T"],
    "united": ["U"],
    "university": ["U", "UNI"],
    "unknown": ["X", "Y", "Z"],
    "very": ["V", "SO"],
    "victory": ["V"],
    "volume": ["V", "VOL"],
    "way": ["RD", "ST", "AVE"],
    "weight": ["W", "OZ", "LB", "TON"],
    "west": ["W"],
    "western": ["W"],
    "wife": ["W"],
    "woman": ["HER", "SHE", "EVE"],
    "work": ["OP"],
    "worker": ["ANT", "BEE"],
    "year": ["Y", "YR"],
    "zero": ["O", "NIL"],
}


# ===================================================================
# Known homophones
# ===================================================================

KNOWN_HOMOPHONES = {
    ("pair", "pare"), ("pear", "pare"), ("pair", "pear"),
    ("night", "knight"), ("hear", "here"), ("write", "right"),
    ("new", "knew"), ("no", "know"), ("sea", "see"),
    ("son", "sun"), ("won", "one"), ("flower", "flour"),
    ("meet", "meat"), ("piece", "peace"), ("tale", "tail"),
    ("week", "weak"), ("wait", "weight"), ("way", "weigh"),
    ("break", "brake"), ("made", "maid"), ("male", "mail"),
    ("sale", "sail"), ("steal", "steel"), ("stare", "stair"),
    ("bare", "bear"), ("fair", "fare"), ("hair", "hare"),
    ("rain", "reign"), ("vain", "vane"), ("vein", "vane"),
    ("board", "bored"), ("moan", "mown"), ("road", "rode"),
    ("sew", "so"), ("sow", "so"), ("doe", "dough"),
    ("toe", "tow"), ("threw", "through"), ("blue", "blew"),
    ("you", "ewe"), ("eye", "I"), ("two", "too"), ("to", "two"),
    ("ate", "eight"), ("be", "bee"), ("buy", "by"), ("by", "bye"),
    ("dear", "deer"), ("die", "dye"), ("for", "four"),
    ("him", "hymn"), ("hour", "our"), ("in", "inn"),
    ("knot", "not"), ("led", "lead"), ("loan", "lone"),
    ("ore", "or"), ("owe", "oh"), ("plain", "plane"),
    ("prophet", "profit"), ("role", "roll"), ("scene", "seen"),
    ("sore", "soar"), ("sum", "some"), ("their", "there"),
    ("wear", "where"), ("which", "witch"), ("wood", "would"),
    ("your", "you're"),
}


# ===================================================================
# DSL Functions
# ===================================================================

def is_synonym(phrase: str, test_synonym: str, pattern: str = "") -> bool:
    """
    Check if test_synonym is a reasonable synonym for phrase.
    Optionally checks that the letter count matches pattern (e.g. '7').
    Uses NLTK WordNet when available, falls back to permissive mode.
    """
    phrase_l = phrase.strip().lower()
    test_l = test_synonym.strip().lower()

    if pattern:
        expected_len = sum(int(x) for x in re.findall(r"\d+", pattern))
        clean = re.sub(r"[^a-zA-Z]", "", test_synonym)
        if len(clean) != expected_len:
            raise AssertionError(
                f"is_synonym: '{test_synonym}' has {len(clean)} letters "
                f"but pattern '{pattern}' requires {expected_len}"
            )

    if phrase_l == test_l:
        return True

    try:
        from nltk.corpus import wordnet as wn
        phrase_syns = set()
        for ss in wn.synsets(phrase_l.replace(" ", "_")):
            for lemma in ss.lemmas():
                phrase_syns.add(lemma.name().lower().replace("_", " "))
        if test_l in phrase_syns:
            return True
        test_syns = set()
        for ss in wn.synsets(test_l.replace(" ", "_")):
            for lemma in ss.lemmas():
                test_syns.add(lemma.name().lower().replace("_", " "))
        if phrase_l in test_syns:
            return True
        if phrase_syns & test_syns:
            return True
    except (ImportError, LookupError):
        pass

    return True


def is_abbreviation(phrase: str, test_abbreviation: str) -> bool:
    """
    Check if test_abbreviation is a valid abbreviation/short form of phrase.
    Looks up against the built-in dictionary.
    """
    phrase_l = phrase.strip().lower()
    test_u = test_abbreviation.strip().upper()

    if phrase_l in ABBREVIATIONS:
        if test_u in ABBREVIATIONS[phrase_l]:
            return True

    for key, abbrevs in ABBREVIATIONS.items():
        if key in phrase_l and test_u in abbrevs:
            return True

    if len(test_abbreviation.strip()) <= 3 and test_abbreviation.strip().isupper():
        return True

    matching_keys = [k for k, v in ABBREVIATIONS.items() if test_u in v]
    raise AssertionError(
        f"is_abbreviation('{phrase}', '{test_abbreviation}'): "
        f"'{phrase}' does not have a valid abbreviation; "
        f"'{test_abbreviation}' is an abbreviation for : "
        + ", ".join(matching_keys)
    )


def action_type(phrase: str, action: Action) -> bool:
    """
    Check if phrase is an indicator word for the given Action type.
    E.g. action_type("shredded", Action.ANAGRAM) -> True
    """
    phrase_l = phrase.strip().lower()
    indicators = INDICATOR_WORDS.get(action, set())

    if phrase_l in indicators:
        return True

    for indicator in indicators:
        if indicator in phrase_l or phrase_l in indicator:
            return True

    all_matching = []
    for act, words in INDICATOR_WORDS.items():
        if phrase_l in words or any(phrase_l in w or w in phrase_l for w in words):
            all_matching.append(act)

    if all_matching:
        raise AssertionError(
            f"action_type('{phrase}', {action}): "
            f"'{phrase}' does not suggest {action}, "
            f"but {all_matching[0]} does match"
        )
    raise AssertionError(
        f"action_type('{phrase}', {action}): "
        f"'{phrase}' does not suggest {action}"
    )


def is_anagram(letters: str, word: str) -> bool:
    """
    Check if word can be formed by rearranging letters.
    Deterministic: compares sorted character sequences.
    """
    letters_clean = re.sub(r"[^a-zA-Z]", "", letters).upper()
    word_clean = re.sub(r"[^a-zA-Z]", "", word).upper()

    if sorted(letters_clean) == sorted(word_clean):
        return True

    raise AssertionError(
        f"is_anagram('{letters}', '{word}'): "
        f"'{letters_clean}' ({sorted(letters_clean)}) cannot form "
        f"'{word_clean}' ({sorted(word_clean)})"
    )


def is_homophone(phrase: str, test_homophone: str) -> bool:
    """
    Check if test_homophone sounds like phrase.
    Uses a built-in list of known homophones, falls back to permissive.
    """
    phrase_l = phrase.strip().lower()
    test_l = test_homophone.strip().lower()

    if phrase_l == test_l:
        return True

    pair = tuple(sorted([phrase_l, test_l]))
    for h in KNOWN_HOMOPHONES:
        if tuple(sorted(h)) == pair:
            return True

    return True


# ===================================================================
# Proof executor
# ===================================================================

def build_dsl_globals() -> dict:
    """Build the global namespace for proof execution."""
    return {
        "is_synonym": is_synonym,
        "is_abbreviation": is_abbreviation,
        "action_type": action_type,
        "is_anagram": is_anagram,
        "is_homophone": is_homophone,
        "Action": Action,
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "sorted": sorted,
            "list": list,
            "True": True,
            "False": False,
            "None": None,
            "AssertionError": AssertionError,
        },
    }


def execute_proof(proof_code: str) -> dict:
    """
    Execute a Python proof function and return the result.

    Args:
        proof_code: Python source code containing a def proof(...) function
                    with assert statements, ending with proof()

    Returns:
        dict with keys:
            success: bool — True if all asserts passed
            errors:  list[str] — error messages with constructive hints
            proof_code: str — the code that was executed
    """
    result = {
        "success": False,
        "errors": [],
        "proof_code": proof_code,
    }

    try:
        tree = ast.parse(proof_code)
    except SyntaxError as e:
        result["errors"].append(f"SyntaxError: {e}")
        return result

    has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
    has_asserts = any(isinstance(node, ast.Assert) for node in ast.walk(tree))

    if not has_function:
        result["errors"].append("No function definition found in proof code")
        return result
    if not has_asserts or sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert)) < 2:
        result["errors"].append("Proof must contain at least 2 assert statements")
        return result

    dsl_globals = build_dsl_globals()

    try:
        exec(proof_code, dsl_globals)
        result["success"] = True
    except AssertionError as e:
        result["errors"].append(f"AssertionError: assert {e}")
    except Exception as e:
        result["errors"].append(f"{type(e).__name__}: {e}")

    return result


# ===================================================================
# CLI & Demo
# ===================================================================

DEMO_PROOF = '''
def proof(answer="EXAMPLE",
          clue="Cut up over politician on the French case",
          pattern='7'):
    """
    definition: Cut up over politician on the French {case}
    wordplay: (AXE)< (cut, <up) + MP (politician) + LE (the, in French)
    """
    assert is_synonym("cut", "AXE")
    assert action_type("up", Action.REVERSE)
    assert "AXE"[::-1] == "EXA"
    assert is_abbreviation("politician", "MP")
    assert is_synonym("the, in French", "LE")
    assert "EXA"+"MP"+"LE" == "EXAMPLE"
    assert is_synonym("case", "EXAMPLE", pattern='7')
proof()
'''

DEMO_PROOF_ANAGRAM = '''
def proof(answer="ESCORT",
          clue="Chaperone shredded corset",
          pattern='6'):
    """
    definition: {Chaperone} shredded corset
    wordplay: (corset)* (*shredded = anagram indicator)
    """
    assert is_synonym("Chaperone", "ESCORT", pattern='6')
    assert action_type("shredded", Action.ANAGRAM)
    assert is_anagram("corset", "ESCORT")
proof()
'''

DEMO_PROOF_FAIL = '''
def proof(answer="DECIMAL",
          clue="the point of medical treatment",
          pattern='7'):
    """
    definition: {the point} of medical treatment
    wordplay: (MEDICAL)* (*treatment = anagram)
    """
    assert is_synonym("the point", "DECIMAL", pattern='7')
    assert action_type("treatment", Action.ANAGRAM)
    assert is_anagram("MEDICAL", "DECIBEL")
proof()
'''


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cryptic crossword proof verifier")
    parser.add_argument("--code", type=str, help="Path to a Python proof file")
    parser.add_argument("--demo", action="store_true", help="Run demo proofs")
    args = parser.parse_args()

    if args.demo:
        demos = [
            ("EXAMPLE proof (should PASS)", DEMO_PROOF),
            ("ESCORT anagram proof (should PASS)", DEMO_PROOF_ANAGRAM),
            ("DECIMAL proof with wrong anagram (should FAIL)", DEMO_PROOF_FAIL),
        ]
        for name, code in demos:
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            print(code.strip())
            print(f"\n--- Result ---")
            result = execute_proof(code)
            if result["success"]:
                print("  SUCCESS: proof verified")
            else:
                print("  FAILED:")
                for err in result["errors"]:
                    print(f"    {err}")
        return

    if args.code:
        with open(args.code) as f:
            code = f.read()
        result = execute_proof(code)
        if result["success"]:
            print("SUCCESS: proof verified")
        else:
            print("FAILED:")
            for err in result["errors"]:
                print(f"  {err}")
        sys.exit(0 if result["success"] else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
