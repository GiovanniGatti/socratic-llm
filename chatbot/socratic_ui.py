import argparse
import urllib.request
from threading import Thread

import gradio as gr
import torch
from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="SOCRATIC-CHATBOT", description="Socratic chatbot")

    parser.add_argument("--load-in-4bit",
                        action="store_true",
                        help="Load base model with 4bit quantization (requires GPU)")

    parser.add_argument("--server-port",
                        type=int,
                        default=2121,
                        help="The port the chatbot server listens to")

    args = parser.parse_args()

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        with urllib.request.urlopen(
                "https://raw.githubusercontent.com/GiovanniGatti/socratic-llm/kdd-2024/templates/inference.txt"
        ) as f:
            inference_prompt_template = f.read().decode('utf-8')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = AutoModelForCausalLM.from_pretrained(
            "eurecom-ds/Phi-3-mini-4k-socratic",
            torch_dtype=torch.bfloat16,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=True,
            device_map=device,
        )

        tokenizer = AutoTokenizer.from_pretrained("eurecom-ds/Phi-3-mini-4k-socratic")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


        def user(user_message, history):
            return "", history + [[user_message, ""]]


        def bot(history):
            user_query = "".join(f"Student: {s}\nTeacher: {t}\n" for s, t in history[:-1])
            last_query: str = history[-1][0]
            user_query += f"Student: {last_query}"
            content = inference_prompt_template.format(input=user_query)

            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}, ], tokenize=False, add_generation_prompt=True
            )

            encoded_inputs = tokenizer([formatted, ], return_tensors="pt").to(device)

            thread = Thread(target=model.generate, kwargs=dict(encoded_inputs, max_new_tokens=250, streamer=streamer))
            thread.start()

            for word in streamer:
                history[-1][1] += word
                yield history


        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, [chatbot], chatbot)

        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.server_port)
