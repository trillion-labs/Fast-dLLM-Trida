import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import random
import types
from generation_functions import setup_model_with_custom_generation


# Check available GPU
device_accelerated = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Accelerated model using device: {device_accelerated}")


# Set random seed
def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


fix_seed(42)

# Load model and tokenizer - using Fast_dLLM model
model_name = "trillionlabs/Trida-7B-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load Fast_dLLM model instance
model_accelerated = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device_accelerated,
    trust_remote_code=True,
)

# Set up custom generation functions
model_accelerated = setup_model_with_custom_generation(model_accelerated)

# Constants
MASK_TOKEN = "<|mask|>"
MASK_ID = 128012
question_ai = """Write a piece of code to implement quick sort."""
question_math = """A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?"""
question_gsm8k = """Question: Skyler has 100 hats on his hand with the colors red, blue, and white. Half of the hats are red, 3/5 of the remaining hats are blue, and the rest are white. How many white hats does Skyler have?"""

# Removed parse_constraints function - no longer needed


def format_chat_history(history):
    """
    Format chat history for the LLaDA model

    Args:
        history: List of [user_message, assistant_message] pairs

    Returns:
        Formatted conversation for the model
    """
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Skip if None (for the latest user message)
            messages.append({"role": "assistant", "content": assistant_msg})

    return messages


@torch.no_grad()
def generate_response_with_visualization_fast_dllm(
    model,
    tokenizer,
    device,
    messages,
    max_new_tokens=1024,
    temperature=0.0,
    block_length=32,
    threshold=0.9,
    top_p=0.9,
):
    """
    Generate text with Fast_dLLM model with visualization using custom generation function

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        block_length: Block size for generation
        threshold: Threshold for generation
        top_p: Top-p sampling parameter

    Yields:
        Visualization states showing the progression and final text
    """

    # Prepare the prompt using chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Use custom mdm_sample_with_visualization method
    generator = model.mdm_sample_with_visualization(
        model_inputs["input_ids"],
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        small_block_size=block_length,
        temperature=temperature,
        threshold=threshold,
        top_p=top_p,
    )

    # Collect all states and final text
    states = []
    for item in generator:
        if isinstance(item, list):  # Visualization state
            states.append(item)
            yield item
        else:  # Final text
            final_text = item
            break

    # Return final text
    yield final_text


css = """
.category-legend{display:none}
.message, .bubble, .chatbot .message, .chatbot .bubble {
    max-width: 80% !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    box-sizing: border-box !important;
}
/* HighlightedText allows auto line wrapping and sets fixed height */
.highlighted-text-container {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    height: 200px !important;
    overflow-y: auto !important;
}
.generating {
    border: none;
}
#input-row {
    align-items: center !important;
}
"""


def create_chatbot_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            "# Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding"
        )
        gr.Markdown(
            "[code](https://github.com/NVlabs/Fast-dLLM), [project page](https://nvlabs.github.io/Fast-dLLM/)"
        )

        # STATE MANAGEMENT
        chat_history_cache = gr.State([])

        # UI COMPONENTS

        # Input area - moved above Fast-dLLM Accelerated section
        with gr.Group():
            with gr.Row(elem_id="input-row"):
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=8,
                )
                send_btn = gr.Button("Send", scale=1)
                clear_btn = gr.Button("Clear Conversation", scale=1)

        # Fast-dLLM Accelerated conversation interface
        gr.Markdown("## Fast-dLLM Model (7B Parameters)")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot_ui = gr.Chatbot(
                    label="Conversation (Fast-dLLM Model)", height=520
                )
            with gr.Column(scale=2):
                with gr.Row():
                    generation_time = gr.Textbox(
                        label="Generation Time",
                        value="wait for generation",
                        interactive=False,
                    )
                    throughput = gr.Textbox(
                        label="Generation Speed",
                        value="wait for generation",
                        interactive=False,
                    )
                output_vis = gr.HighlightedText(
                    label="Denoising Process Visualization (Real-time)",
                    combine_adjacent=False,
                    show_legend=True,
                    elem_classes=["highlighted-text-container"],
                )
                output_vis_slow = gr.HighlightedText(
                    label="Denoising Process Visualization (Slow Motion)",
                    combine_adjacent=False,
                    show_legend=True,
                    elem_classes=["highlighted-text-container"],
                )

        # Examples moved below the conversation interfaces
        gr.Examples(
            examples=[
                [question_ai],
                [question_gsm8k],
                [question_math],
            ],
            inputs=user_input,
            label="Example Inputs",
        )

        # Advanced generation settings
        with gr.Accordion("Generation Settings", open=True):
            with gr.Row():
                max_new_tokens = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Max New Tokens",
                )
                block_length = gr.Slider(
                    minimum=4, maximum=32, value=16, step=4, label="Block Size"
                )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p"
                )
            with gr.Row():
                threshold = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.95, step=0.05, label="Threshold"
                )
                visualization_delay = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.1,
                    label="Visualization Delay (seconds)",
                )

        # Current response text box (hidden)
        current_response = gr.Textbox(
            label="Current Response",
            placeholder="The assistant's response will appear here...",
            lines=3,
            visible=False,
        )

        # HELPER FUNCTIONS
        def add_message(history, message, response):
            """Add a message pair to the history and return the updated history"""
            history = history.copy()
            history.append([message, response])
            return history

        def user_message_submitted(message, history_cache, max_new_tokens):
            """Process a submitted user message"""
            # Skip empty messages
            if not message.strip():
                # Return current state unchanged
                history_cache_for_display = history_cache.copy()
                return (
                    history_cache,
                    history_cache_for_display,
                    "",
                    [],
                    [],
                    "wait for generation",
                    "wait for generation",
                )

            # Add user message to history
            history_cache = add_message(history_cache, message, None)

            # Format for display - temporarily show user message with empty response
            history_cache_for_display = history_cache.copy()

            # Clear the input
            message_out = ""

            # Return immediately to update UI with user message
            return (
                history_cache,
                history_cache_for_display,
                message_out,
                [],
                [],
                "processing...",
                "processing...",
            )

        def accelerated_response(
            history_cache,
            max_new_tokens,
            temperature,
            top_p,
            block_length,
            threshold,
            visualization_delay,
        ):
            """Generate accelerated model response independently"""
            if not history_cache:
                return (
                    history_cache,
                    [],
                    [],
                    "",
                    "wait for generation",
                    "wait for generation",
                )

            # Get the last user message
            last_user_message = history_cache[-1][0]

            try:
                # Format all messages except the last one (which has no response yet)
                messages = format_chat_history(history_cache[:-1])

                # Add the last user message
                messages.append({"role": "user", "content": last_user_message})

                # Start timing
                start_time = time.time()

                # Generate with accelerated model and yield states in real-time
                with torch.no_grad():
                    generator = generate_response_with_visualization_fast_dllm(
                        model_accelerated,
                        tokenizer,
                        device_accelerated,
                        messages,
                        max_new_tokens,
                        temperature,
                        block_length,
                        threshold,
                        top_p,
                    )

                    # Collect all states and get final text
                    states = []
                    for item in generator:
                        if isinstance(item, list):  # Visualization state
                            states.append(item)
                            yield history_cache, item, [], "", "processing...", "processing..."
                        else:  # Final text
                            cache_response_text = item
                            break

                accelerated_complete_time = time.time() - start_time
                cache_generation_time_str = f"{accelerated_complete_time:.2f}s"

                # Calculate throughput
                cache_response_tokens = tokenizer.encode(
                    cache_response_text, add_special_tokens=False
                )
                cache_num_tokens = len(cache_response_tokens)
                cache_throughput = (
                    cache_num_tokens / accelerated_complete_time
                    if accelerated_complete_time > 0
                    else 0
                )
                cache_throughput_str = f"{cache_throughput:.2f} tokens/s"

                # Update history
                history_cache[-1][1] = cache_response_text

                # Final yield with complete information and start slow motion visualization
                if states:
                    # First, yield the final real-time state
                    yield history_cache, states[-1], states[
                        0
                    ], cache_response_text, cache_generation_time_str, cache_throughput_str

                    # Then animate through slow motion visualization
                    for state in states[1:]:
                        time.sleep(visualization_delay)
                        yield history_cache, states[
                            -1
                        ], state, cache_response_text, cache_generation_time_str, cache_throughput_str

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                error_vis = [(error_msg, "red")]
                yield history_cache, error_vis, error_vis, error_msg, "Error", "Error"

        def clear_conversation():
            """Clear the conversation history"""
            empty_history = []
            empty_response = ""
            empty_vis = []
            time_str = "wait for generation"
            throughput_str = "wait for generation"

            return (
                empty_history,  # chat_history_cache
                empty_history,  # chatbot_ui
                empty_response,  # current_response
                empty_vis,  # output_vis
                empty_vis,  # output_vis_slow
                time_str,  # generation_time
                throughput_str,  # throughput
            )

        # EVENT HANDLERS

        # Clear button handler
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[
                chat_history_cache,
                chatbot_ui,
                current_response,
                output_vis,
                output_vis_slow,
                generation_time,
                throughput,
            ],
        )

        # User message submission flow (2-step process)
        # Step 1: Add user message to history and update UI
        msg_submit = user_input.submit(
            fn=user_message_submitted,
            inputs=[user_input, chat_history_cache, max_new_tokens],
            outputs=[
                chat_history_cache,
                chatbot_ui,
                user_input,
                output_vis,
                output_vis_slow,
                generation_time,
                throughput,
            ],
        )

        # Also connect the send button
        send_click = send_btn.click(
            fn=user_message_submitted,
            inputs=[user_input, chat_history_cache, max_new_tokens],
            outputs=[
                chat_history_cache,
                chatbot_ui,
                user_input,
                output_vis,
                output_vis_slow,
                generation_time,
                throughput,
            ],
        )

        # Step 2: Generate accelerated model response
        msg_submit.then(
            fn=accelerated_response,
            inputs=[
                chat_history_cache,
                max_new_tokens,
                temperature,
                top_p,
                block_length,
                threshold,
                visualization_delay,
            ],
            outputs=[
                chatbot_ui,
                output_vis,
                output_vis_slow,
                current_response,
                generation_time,
                throughput,
            ],
        )

        send_click.then(
            fn=accelerated_response,
            inputs=[
                chat_history_cache,
                max_new_tokens,
                temperature,
                top_p,
                block_length,
                threshold,
                visualization_delay,
            ],
            outputs=[
                chatbot_ui,
                output_vis,
                output_vis_slow,
                current_response,
                generation_time,
                throughput,
            ],
        )

    return demo


# Launch the demo
if __name__ == "__main__":
    demo = create_chatbot_demo()
    demo.queue().launch(server_port=10086, share=True)
