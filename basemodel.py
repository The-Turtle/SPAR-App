from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
import torch
import os
from util import binary_prompt, nonempty_prompt

class BaseModel:
    def __init__(self) -> None:
        self.model_name = "distilgpt2"
        self.tokenizer = None
        self.model = None
        self.generator = None

        self.load()

    def load(self) -> None:
        """Loads the tokenizer, DistilGPT2 model, and text generation pipeline."""

        print("Loading tokenizer...", end=" ")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token
        print(f"loaded {self.model_name} tokenizer")

        print("Loading model...", end=" ")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        print(f"loaded {self.model_name} model")

        print("Loading generator...", end=" ")
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        print("generator loaded")

    def generate_text(
        self, prompt: str, *, max_new_tokens=100, temperature=1, top_p=0.8
    ) -> str:
        """Generates text using the loaded model."""

        output = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
        )

        generated_text = output[0]["generated_text"]
        generated_text = generated_text.strip()
        return generated_text


def main():
    model = BaseModel()

    while True:
        prompt = nonempty_prompt("Enter a prompt:")

        print("Generated text:", model.generate_text(prompt, max_new_tokens=100))

        will_continue = binary_prompt(
            "Would you like to (c)ontinue or (q)uit?", "c", "q"
        )
        if not will_continue:
            print("Exiting.")
            exit(0)




if __name__ == "__main__":
    main()
