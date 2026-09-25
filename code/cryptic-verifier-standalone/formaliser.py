"""
Formaliser + Verifier (Steps 3-6)

Takes English wordplay text and:
  3. Formalises it into Python proof code (using gemma-2-9b-it with ICL)
  4. Runs the verifier on the proof
  5. If proof fails, sends error back to model for rewrite (up to 2x)
  6. Returns whether the answer is verified

Usage:
    python formaliser.py \
        --model-path ./models/gemma-2-9b-it \
        --clue "Chaperone shredded corset" \
        --answer "ESCORT" \
        --definition "Chaperone" \
        --wordplay "(corset)* (*shredded = anagram)" \
        --pattern 6
"""

import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from verifier import execute_proof, Action


# ===================================================================
# ICL prompts for formalisation
# ===================================================================

PREAMBLE = """A Cryptic crossword question involves using the words in \
the given clue to yield an answer that matches the letter pattern.
The clue will provide a definition of the answer, as well \
as some 'wordplay' that can also be used to confirm the answer.

The task is to produce a formal proof using python code, \
where the docstring will also include an informal proof as an aid.
The following are functions that can be used in your output code:

Action=Enum('Action', 'ANAGRAM,REMOVE_FIRST,INITIALS,REMOVE_LAST,'+
                    'GOES_INSIDE,GOES_OUTSIDE,REVERSE,SUBSTRING,HOMOPHONE')
# External definitions
def is_synonym(phrase:str, test_synonym:str, pattern:str='') -> bool:
def is_abbreviation(phrase:str, test_abbreviation:str) -> bool:
def action_type(phrase:str, action:Action) -> bool:
def is_anagram(letters:str, word:str) -> bool:
def is_homophone(phrase:str, test_homophone:str) -> bool:

The following are examples of simple functions that prove that \
each puzzle solution is correct:
"""

EXAMPLES = [
    '''```python
def proof(answer="ONCE",
          clue="head decapitated long ago", pattern='4'):
    """
    definition: head decapitated {long ago}
    wordplay: [b]ONCE (head decapitated = remove first letter of BONCE)
    """
    assert is_synonym("head", "BONCE")
    assert action_type("decapitated", Action.REMOVE_FIRST) \\
           and "BONCE"[1:]=="ONCE"
    assert is_synonym("long ago", "ONCE", pattern='4')
proof()
```''',
    '''```python
def proof(answer="DECIMAL",
          clue="the point of medical treatment", pattern='7'):
    """
    definition: {the point} of medical treatment
    wordplay: (MEDICAL)* (*treatment = anagram)
    """
    assert is_synonym("the point", "DECIMAL", pattern='7')
    assert action_type("treatment", Action.ANAGRAM)
    assert is_anagram("MEDICAL", "DECIMAL")
proof()
```''',
    '''```python
def proof(answer="EXAMPLE",
          clue="Cut up over politician on the French case", pattern='7'):
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
```''',
    '''```python
def proof(answer="ESCORT",
          clue="Chaperone shredded corset", pattern='6'):
    """
    definition: {Chaperone} shredded corset
    wordplay: (corset)* (*shredded = anagram indicator)
    """
    assert is_synonym("Chaperone", "ESCORT", pattern='6')
    assert action_type("shredded", Action.ANAGRAM)
    assert is_anagram("corset", "ESCORT")
proof()
```''',
    '''```python
def proof(answer="PROPOSAL",
          clue="Offer of support also broadcast", pattern='8'):
    """
    definition: {Offer} of support also broadcast
    wordplay: PROP (support) + (ALSO)* (*broadcast)
    """
    assert is_synonym("Offer", "PROPOSAL", pattern='8')
    assert is_synonym("support", "PROP")
    assert action_type("broadcast", Action.ANAGRAM)
    assert is_anagram("ALSO", "OSAL")
    assert "PROP"+"OSAL" == "PROPOSAL"
proof()
```''',
]

REWRITE_INSTRUCTION = """# Please re-implement the SOLUTION above \
(altering both the docstring and the python code as required), \
taking care to fix each of the problems identified, \
and return the whole function:

```python
def proof(answer= ..."""


# ===================================================================
# Model loading & generation
# ===================================================================

def load_model(model_path, device="auto"):
    print(f"Loading formaliser model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    model.eval()
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ===================================================================
# Extract Python code from LLM output
# ===================================================================

def extract_python_code(text):
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "def proof(" in text:
        start = text.index("def proof(")
        lines = text[start:].split("\n")
        code_lines = []
        for line in lines:
            code_lines.append(line)
            if line.strip() == "proof()":
                break
        return "\n".join(code_lines)
    return text.strip()


# ===================================================================
# Steps 3-6: Formalise, Verify, Retry, Select
# ===================================================================

def build_formalisation_prompt(clue, answer, definition, wordplay, pattern):
    parts = [PREAMBLE]
    for ex in EXAMPLES:
        parts.append(ex)
        parts.append("\n\n")
    parts.append(
        "# Please complete the following in a similar manner, "
        "and return the whole function:\n\n"
        f"```python\n"
        f'def proof(answer="{answer}",\n'
        f'          clue="{clue}",\n'
        f"          pattern='{pattern}'):\n"
        f'    """\n'
        f"    definition: {definition}\n"
        f"    wordplay: {wordplay}\n"
        f'    """\n'
    )
    return "".join(parts)


def formalise_and_verify(model, tokenizer, clue, answer, definition,
                          wordplay, pattern, max_rewrites=2):
    """
    Steps 3-6:
      3. Formalise wordplay into Python proof
      4. Run verifier
      5. If fail, rewrite (up to max_rewrites times)
      6. Return result
    """
    # Step 3: formalise
    prompt = build_formalisation_prompt(clue, answer, definition, wordplay, pattern)
    output = generate_text(model, tokenizer, prompt)
    proof_code = extract_python_code(output)

    for attempt in range(max_rewrites + 1):
        # Step 4: verify
        result = execute_proof(proof_code)

        if result["success"]:
            # Step 6: answer verified
            return {
                "answer": answer,
                "verified": True,
                "proof_code": proof_code,
                "attempt": attempt,
            }

        # Step 5: rewrite on failure
        if attempt < max_rewrites:
            error_text = "\n".join(result["errors"])
            rewrite_prompt = (
                f"{proof_code}\n\n{error_text}\n\n{REWRITE_INSTRUCTION}"
            )
            rewrite_output = generate_text(model, tokenizer, rewrite_prompt)
            proof_code = extract_python_code(rewrite_output)

    # Step 6: failed after all retries
    return {
        "answer": answer,
        "verified": False,
        "proof_code": proof_code,
        "errors": result["errors"],
        "attempt": max_rewrites,
    }


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Formalise + Verify (Steps 3-6)")
    parser.add_argument("--model-path", required=True, help="Path to gemma-2-9b-it")
    parser.add_argument("--clue", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--definition", required=True)
    parser.add_argument("--wordplay", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--max-rewrites", type=int, default=2)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    result = formalise_and_verify(
        model, tokenizer,
        clue=args.clue,
        answer=args.answer.upper(),
        definition=args.definition,
        wordplay=args.wordplay,
        pattern=args.pattern,
        max_rewrites=args.max_rewrites,
    )

    print("\n" + "=" * 60)
    print(f"Answer:   {result['answer']}")
    print(f"Verified: {result['verified']}")
    print(f"Attempts: {result['attempt'] + 1}")
    if result["verified"]:
        print(f"\nProof:\n{result['proof_code']}")
    else:
        print(f"\nErrors:")
        for e in result.get("errors", []):
            print(f"  {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
