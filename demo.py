import streamlit as st
import time
import json
import sys
import os

# Add directory to path
sys.path.append(os.path.dirname(__file__))
from inference import run, parse_tool_call

st.set_page_config(page_title="Pocket-Agent", page_icon="📱", layout="centered")

st.title("📱 Pocket-Agent")
st.markdown("Offline mobile assistant running on Qwen2.5-0.5B-Instruct (GGUF INT4).")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Format history correctly for inference
    # inference history expects a list of {"role": "user"|"assistant", "content": "..."}
    # We should exclude the latency metadata from the sent history
    formatted_history = []
    for msg in st.session_state.messages:
        # Extract the original text if possible, but sending raw text is usually okay
        content = msg["content"]
        # Remove tool call metadata and latency string if we want pure history,
        # but the prompt didn't specify strict history cleaning. 
        # For simplicity, we pass the stored 'raw_content' if it's from the assistant.
        formatted_history.append({
            "role": msg["role"], 
            "content": msg.get("raw_content", content)
        })

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "raw_content": prompt})

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.time()
            try:
                response = run(prompt, formatted_history)
            except Exception as e:
                response = f"Error during inference: {str(e)}\n\nPlease ensure you have trained and quantized the model first."
            t1 = time.time()
            
            latency = (t1 - t0) * 1000
            
            parsed = parse_tool_call(response)
            if parsed:
                output_str = f"**🛠️ Tool Call:**\n```json\n{json.dumps(parsed, indent=2)}\n```"
            else:
                output_str = f"**💬 Assistant:**\n{response}"
                
            display_str = f"{output_str}\n\n*(Latency: {latency:.2f} ms)*"
            st.markdown(display_str)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": display_str, 
                "raw_content": response
            })
