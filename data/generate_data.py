import os
import json
import random
import hashlib

SYSTEM_PROMPT = """You are Pocket-Agent, an offline mobile assistant. You have access to exactly
five tools: weather, calendar, convert, currency, sql.
When the user's request clearly maps to one of these tools, respond ONLY with:
<tool_call>{"tool": "<name>", "args": {<args>}}</tool_call>
When no tool fits (chitchat, ambiguous reference, unknown tool), respond in
plain natural language with no <tool_call> tag.
Always use exact ISO 4217 currency codes (USD, EUR, PKR, etc.), exact
YYYY-MM-DD dates, and match units precisely to user intent."""

# Helper sets to build dataset
CITIES = ["London", "New York", "Tokyo", "Paris", "Berlin", "Dubai", "Mumbai", "Sydney", "Toronto", "Karachi", "Lahore", "Madrid"]
DATES = ["2023-10-15", "2024-01-01", "2024-12-25", "2025-05-10", "2022-07-04"]
CURRENCIES = ["USD", "EUR", "PKR", "JPY", "GBP", "CAD", "AUD", "INR"]
UNITS = ["C", "F", "miles", "km", "kg", "lbs"]

def hash_prompt(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode('utf-8')).hexdigest()

def load_test_hashes() -> set:
    test_file = os.path.join(os.path.dirname(__file__), "..", "starter", "public_test.jsonl")
    hashes = set()
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                for msg in data.get("messages", []):
                    if msg.get("role") == "user":
                        hashes.add(hash_prompt(msg.get("content", "")))
    return hashes

# Generate slice A: In-distribution (40%) ~ 600 examples
def gen_slice_a():
    examples = []
    # weather
    for _ in range(120):
        city = random.choice(CITIES)
        unit = random.choice(["C", "F"])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"What is the weather like in {city} in {unit}?"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "weather", "args": {{"location": "{city}", "unit": "{unit}"}}}}</tool_call>'}
            ]
        })
    # calendar
    for _ in range(120):
        action = random.choice(["list", "create"])
        date = random.choice(DATES)
        title = f"Meeting {random.randint(1,100)}" if action == "create" else ""
        if action == "create":
            prompt = f"Create an event called '{title}' on {date}"
            args = f'{{"action": "create", "date": "{date}", "title": "{title}"}}'
        else:
            prompt = f"List my calendar events for {date}"
            args = f'{{"action": "list", "date": "{date}"}}'
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "calendar", "args": {args}}}</tool_call>'}
            ]
        })
    # convert
    for _ in range(120):
        val = round(random.uniform(1.0, 100.0), 2)
        from_u = random.choice(["miles", "kg"])
        to_u = "km" if from_u == "miles" else "lbs"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Convert {val} {from_u} to {to_u}"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "convert", "args": {{"value": {val}, "from_unit": "{from_u}", "to_unit": "{to_u}"}}}}</tool_call>'}
            ]
        })
    # currency
    for _ in range(120):
        amt = round(random.uniform(10.0, 5000.0), 2)
        fr = random.choice(CURRENCIES)
        to = random.choice([c for c in CURRENCIES if c != fr])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"How much is {amt} {fr} in {to}?"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "currency", "args": {{"amount": {amt}, "from": "{fr}", "to": "{to}"}}}}</tool_call>'}
            ]
        })
    # sql
    for _ in range(120):
        table = random.choice(["users", "orders", "products"])
        col = random.choice(["id", "name", "price"])
        q = f"SELECT {col} FROM {table} LIMIT 10;"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Run this query: {q}"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "sql", "args": {{"query": "{q}"}}}}</tool_call>'}
            ]
        })
    return examples

# Generate slice B: Paraphrased (20%) ~ 300 examples
def gen_slice_b():
    examples = []
    # weather
    for _ in range(60):
        city = random.choice(CITIES)
        unit = random.choice(["C", "F"])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Tell me if it's hot or cold over in {city}, but use degrees {unit} please."},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "weather", "args": {{"location": "{city}", "unit": "{unit}"}}}}</tool_call>'}
            ]
        })
    # calendar
    for _ in range(60):
        date = random.choice(DATES)
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"I need to know my schedule for the day of {date}."},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "calendar", "args": {{"action": "list", "date": "{date}"}}}}</tool_call>'}
            ]
        })
    # convert
    for _ in range(60):
        val = random.randint(1, 100)
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Can you change {val} kg into lbs for me?"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "convert", "args": {{"value": {val}, "from_unit": "kg", "to_unit": "lbs"}}}}</tool_call>'}
            ]
        })
    # currency
    for _ in range(60):
        amt = random.randint(10, 1000)
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate {amt} USD into European EUR."},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "currency", "args": {{"amount": {amt}, "from": "USD", "to": "EUR"}}}}</tool_call>'}
            ]
        })
    # sql
    for _ in range(60):
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Execute the following database lookup: SELECT * FROM customers;"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "sql", "args": {{"query": "SELECT * FROM customers;"}}}}</tool_call>'}
            ]
        })
    return examples

# Generate slice C: Adversarial (25%) ~ 375 examples
def gen_slice_c():
    examples = []
    # Typos
    for _ in range(75):
        city = random.choice(CITIES)
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Whats the temprature in {city} in C?"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "weather", "args": {{"location": "{city}", "unit": "C"}}}}</tool_call>'}
            ]
        })
    # Code-switched
    for _ in range(75):
        city = random.choice(CITIES)
        prompts = [
            (f"mujhe weather batao {city} ka in C", "C"),
            (f"¿cuánto es 50 USD en EUR?", "currency"),
        ]
        p = random.choice(prompts)
        if p[1] == "C":
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p[0]},
                    {"role": "assistant", "content": f'<tool_call>{{"tool": "weather", "args": {{"location": "{city}", "unit": "C"}}}}</tool_call>'}
                ]
            })
        else:
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p[0]},
                    {"role": "assistant", "content": f'<tool_call>{{"tool": "currency", "args": {{"amount": 50, "from": "USD", "to": "EUR"}}}}</tool_call>'}
                ]
            })
    # Unit ambiguity
    for _ in range(75):
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"convert 50 c to f"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "convert", "args": {{"value": 50, "from_unit": "C", "to_unit": "F"}}}}</tool_call>'}
            ]
        })
    # Hallucination-bait (refusal)
    for _ in range(75):
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Turn on the living room lights."},
                {"role": "assistant", "content": f"I'm sorry, but I don't have a tool to control smart home devices."}
            ]
        })
    # Numerical edge cases
    for _ in range(75):
        amt = -50.5
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"convert {amt} km to miles"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "convert", "args": {{"value": {amt}, "from_unit": "km", "to_unit": "miles"}}}}</tool_call>'}
            ]
        })
    return examples

# Generate slice D: Refusals & multi-turn (15%) ~ 225 examples
def gen_slice_d():
    examples = []
    # Chitchat
    for _ in range(75):
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Tell me a joke."},
                {"role": "assistant", "content": f"Why did the scarecrow win an award? Because he was outstanding in his field!"}
            ]
        })
    # Ambiguous
    for _ in range(75):
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Convert that."},
                {"role": "assistant", "content": f"I don't know what you want me to convert. Please specify the value and units."}
            ]
        })
    # Multi-turn
    for _ in range(75):
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"What is 10 km in miles?"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "convert", "args": {{"value": 10, "from_unit": "km", "to_unit": "miles"}}}}</tool_call>'},
                {"role": "user", "content": f"now what about 20?"},
                {"role": "assistant", "content": f'<tool_call>{{"tool": "convert", "args": {{"value": 20, "from_unit": "km", "to_unit": "miles"}}}}</tool_call>'}
            ]
        })
    return examples

def main():
    print("Generating data...")
    test_hashes = load_test_hashes()
    
    slices = [gen_slice_a(), gen_slice_b(), gen_slice_c(), gen_slice_d()]
    all_examples = []
    for s in slices:
        for ex in s:
            # Check collisions
            user_msg = [m["content"] for m in ex["messages"] if m["role"] == "user"][-1]
            h = hash_prompt(user_msg)
            if h in test_hashes:
                print(f"Collision detected for prompt: {user_msg}")
                assert False, f"Collision detected for prompt: {user_msg}"
                continue
            all_examples.append(ex)
            
    random.shuffle(all_examples)
    out_file = os.path.join(os.path.dirname(__file__), "train.jsonl")
    with open(out_file, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Generated {len(all_examples)} examples. Saved to {out_file}.")
    
    assert len(all_examples) >= 1500, f"Generated only {len(all_examples)} examples, less than 1500"

if __name__ == "__main__":
    main()
