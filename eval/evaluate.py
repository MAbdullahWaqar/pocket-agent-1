import os
import sys
import json
import time

# Add parent directory to path to import inference module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from inference import run, parse_tool_call

def compare_args(expected_args, pred_args):
    # Check if all keys exist
    for k, v in expected_args.items():
        if k not in pred_args:
            return False
        pv = pred_args[k]
        
        # Check numerics within +- 1%
        if isinstance(v, (int, float)) and isinstance(pv, (int, float)):
            if abs(v - pv) > abs(v) * 0.01 + 1e-5:
                return False
        elif str(v).lower() != str(pv).lower():
            return False
            
    # Also check for extra keys in pred that shouldn't be there
    for k in pred_args.keys():
        if k not in expected_args:
            return False
            
    return True

def score_prediction(expected_output, predicted_output):
    expected_call = parse_tool_call(expected_output)
    pred_call = parse_tool_call(predicted_output)
    
    # Expected Refusal
    if expected_call is None:
        if pred_call is None:
            return 1.0
        else:
            return -0.5
            
    # Expected Tool Call
    if pred_call is None:
        return 0.0
        
    if pred_call.get("tool") != expected_call.get("tool"):
        return 0.0
        
    if compare_args(expected_call.get("args", {}), pred_call.get("args", {})):
        return 1.0
    else:
        return 0.5

def main():
    test_file = os.path.join(os.path.dirname(__file__), "..", "starter", "public_test.jsonl")
    if not os.path.exists(test_file):
        print(f"Test file not found at {test_file}. Creating dummy evaluation.")
        return

    scores = []
    latencies = []
    slice_scores = {"A": [], "B": [], "C": [], "D": []}

    print("=" * 50)
    print("STARTING EVALUATION")
    print("=" * 50)

    with open(test_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # Assume slice info is in the json, if not, put it all in "A"
            ex_slice = data.get("slice", "A")
            
            messages = data.get("messages", [])
            history = messages[:-2] if len(messages) > 2 else []
            prompt = messages[-2]["content"]
            expected_output = messages[-1]["content"]
            
            t0 = time.time()
            predicted_output = run(prompt, history)
            t1 = time.time()
            
            latency = (t1 - t0) * 1000
            latencies.append(latency)
            
            score = score_prediction(expected_output, predicted_output)
            scores.append(score)
            
            if ex_slice in slice_scores:
                slice_scores[ex_slice].append(score)
            else:
                slice_scores[ex_slice] = [score]
                
            print(f"Prompt: {prompt}")
            print(f"Expected: {expected_output}")
            print(f"Predicted: {predicted_output}")
            print(f"Score: {score} | Latency: {latency:.2f} ms")
            print("-" * 50)

    mean_latency = sum(latencies) / len(latencies) if latencies else 0
    total_score = sum(scores)
    max_score = len(scores)
    percentage = (total_score / max_score) * 100 if max_score > 0 else 0

    print("=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Total Score: {total_score}/{max_score} ({percentage:.2f}%)")
    print(f"Mean Inference Latency: {mean_latency:.2f} ms")
    
    for s_name, s_vals in slice_scores.items():
        if s_vals:
            slice_total = sum(s_vals)
            slice_max = len(s_vals)
            slice_pct = (slice_total / slice_max) * 100
            print(f"Slice {s_name} Accuracy: {slice_total}/{slice_max} ({slice_pct:.2f}%)")

if __name__ == "__main__":
    main()
