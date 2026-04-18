import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Pocket-Agent, an offline mobile assistant. You have access to exactly
five tools: weather, calendar, convert, currency, sql.
When the user's request clearly maps to one of these tools, respond ONLY with:
<tool_call>{"tool": "<name>", "args": {<args>}}</tool_call>
When no tool fits (chitchat, ambiguous reference, unknown tool), respond in
plain natural language with no <tool_call> tag.
Always use exact ISO 4217 currency codes (USD, EUR, PKR, etc.), exact
YYYY-MM-DD dates, and match units precisely to user intent."""

# Global model variables
_model = None
_tokenizer = None
_is_gguf = False

def _load_model():
    global _model, _tokenizer, _is_gguf
    if _model is not None:
        return

    gguf_path = os.path.join(os.path.dirname(__file__), "artifacts", "model.Q4_K_M.gguf")
    int4_path = os.path.join(os.path.dirname(__file__), "artifacts", "model_int4")

    if os.path.exists(gguf_path):
        logger.info(f"Loading GGUF model from {gguf_path}")
        from llama_cpp import Llama
        _model = Llama(
            model_path=gguf_path,
            n_ctx=2048,
            n_threads=os.cpu_count() or 4,
            verbose=False
        )
        _is_gguf = True
    elif os.path.exists(int4_path):
        logger.info(f"Loading INT4 transformers model from {int4_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(int4_path)
        _model = AutoModelForCausalLM.from_pretrained(
            int4_path,
            device_map="cpu"
        )
        _is_gguf = False
    else:
        raise FileNotFoundError(f"Neither {gguf_path} nor {int4_path} found. Please train and quantize the model first.")

def run(prompt: str, history: list[dict]) -> str:
    """
    Run inference on the model.
    history should be a list of {"role": "user"|"assistant", "content": "..."}
    """
    _load_model()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    
    if _is_gguf:
        # llama_cpp supports create_chat_completion which formats using the model's chat template
        response = _model.create_chat_completion(
            messages=messages,
            max_tokens=128,
            temperature=0.0
        )
        return response["choices"][0]["message"]["content"].strip()
    else:
        import torch
        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id
            )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        return _tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def parse_tool_call(output: str) -> dict | None:
    """
    Extracts and JSON-parses the tool call from the output.
    Returns None for refusals or invalid tool calls.
    """
    try:
        if "<tool_call>" in output and "</tool_call>" in output:
            start = output.find("<tool_call>") + len("<tool_call>")
            end = output.find("</tool_call>")
            json_str = output[start:end].strip()
            return json.loads(json_str)
    except Exception:
        pass
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Pocket-Agent inference")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to run")
    args = parser.parse_args()
    
    output = run(args.prompt, [])
    print("Output:", output)
    parsed = parse_tool_call(output)
    if parsed:
        print("Parsed Tool Call:", json.dumps(parsed, indent=2))
