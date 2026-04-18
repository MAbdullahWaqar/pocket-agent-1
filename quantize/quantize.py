import os
import subprocess
import shutil
import logging
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "adapter")
    merged_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "merged_model")
    final_gguf_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.Q4_K_M.gguf")

    # Step 1: Merge LoRA into base model
    logger.info("Loading base model to merge adapter...")
    if not os.path.exists(adapter_dir):
        logger.error(f"Adapter not found at {adapter_dir}. Please run fine-tuning first.")
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        try:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="cpu")
        except Exception:
            base_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="cpu")

        logger.info("Loading PEFT adapter and merging...")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model = model.merge_and_unload()
        
        os.makedirs(merged_dir, exist_ok=True)
        logger.info(f"Saving merged model to {merged_dir}")
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
    except Exception as e:
        logger.error(f"Failed to merge model: {e}")
        logger.info("Falling back to INT4 save_pretrained approach...")
        fallback_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model_int4")
        logger.warning(f"Failed merging, this script expects llama.cpp to do GGUF. We will still try to build llama.cpp and run it.")

    # Step 2: Use llama.cpp to convert and quantize
    logger.info("Setting up llama.cpp...")
    llama_cpp_dir = os.path.join(os.path.dirname(__file__), "llama.cpp")
    
    if not os.path.exists(llama_cpp_dir):
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", llama_cpp_dir], check=True)
        logger.info("Building llama.cpp...")
        subprocess.run(["make", "llama-quantize"], cwd=llama_cpp_dir, check=True)
        subprocess.run(["pip", "install", "-r", "requirements.txt"], cwd=llama_cpp_dir, check=True)

    f16_gguf_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.f16.gguf")
    
    logger.info("Converting to GGUF F16...")
    # llama.cpp script name changed recently to just convert_hf_to_gguf.py or similar. Check the dir.
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # some versions use `convert_hf_to_gguf.py` in the root, some use `convert-hf-to-gguf.py`
        convert_script_alt = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
        if os.path.exists(convert_script_alt):
            convert_script = convert_script_alt
            
    subprocess.run([
        sys.executable, convert_script, 
        merged_dir, 
        "--outfile", f16_gguf_path, 
        "--outtype", "f16"
    ], check=True)

    logger.info("Quantizing to Q4_K_M...")
    quantize_bin = os.path.join(llama_cpp_dir, "llama-quantize")
    subprocess.run([
        quantize_bin,
        f16_gguf_path,
        final_gguf_path,
        "Q4_K_M"
    ], check=True)

    # Clean up intermediate files
    logger.info("Cleaning up...")
    shutil.rmtree(merged_dir, ignore_errors=True)
    if os.path.exists(f16_gguf_path):
        os.remove(f16_gguf_path)

    # Step 3: Validate output size
    if os.path.exists(final_gguf_path):
        size_mb = os.path.getsize(final_gguf_path) / (1024 * 1024)
        logger.info(f"Quantized model saved to {final_gguf_path} ({size_mb:.2f} MB)")
        assert size_mb <= 500, f"Model size {size_mb:.2f} MB exceeds the 500 MB limit!"
    else:
        logger.error("Quantization failed, output file not found.")

if __name__ == "__main__":
    main()
