from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    pipeline,
)
import torch
import os
from matplotlib import pyplot as plt

from util import divider, binary_prompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to supress a warning


class FineTunedModel:
    def __init__(
        self,
        *,
        model_name: str,
        model_path: str,
        data_path: str,
        plot_path: str,
        num_train_epochs=8,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.plot_path = plot_path

        self.num_train_epochs = num_train_epochs
        self.learning_rate = 2e-5

        self.tokenizer = None
        self.dataset = None
        self.model = None
        self.train_losses = None
        self.eval_losses = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.load_tokenizer()

    def load_tokenizer(self) -> None:
        """Loads the tokenizer. Automatically called by the constructor."""
        divider("LOADING TOKENIZER")

        print("Loading tokenizer...", end=" ")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        print(f"loaded {self.model_name}")

        self.tokenizer = tokenizer

    def load_data(self) -> None:
        """Loads the data. Automatically called before training."""
        divider("LOADING DATA")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} does not exist.")

        print(f"Reading {self.data_path}...", end=" ")
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"read {len(text)} characters")

        # splitting text into chunks since tokenizer can only output 1024 tokens at a time
        print("Splitting text into chunks...", end=" ")
        chunk_size = 500
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        print(f"created {len(chunks)} chunks of size {chunk_size}")

        print("Tokenizing text...", end=" ")
        tokenized_chunks = [self.tokenizer(chunk)["input_ids"] for chunk in chunks]
        tokenized_text = sum(tokenized_chunks, [])
        print(f"tokenized {len(tokenized_text)} tokens")

        print("Blocking tokens...", end=" ")
        block_size = 128
        blocks = []
        for i in range(0, len(tokenized_text), block_size):
            block = tokenized_text[i : i + block_size]
            # Pad block if it's shorter than block_size
            if len(block) < block_size:
                block = block + [self.tokenizer.pad_token_id] * (
                    block_size - len(block)
                )
            blocks.append(block)
        print(f"created {len(blocks)} training blocks of {block_size} tokens each")

        print("Creating dataset...", end=" ")
        dataset = Dataset.from_dict(
            {
                "input_ids": blocks,
                "labels": blocks,
            }  # For language modeling, labels = input_ids
        )
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        print(
            f"{len(dataset['train'])} training and {len(dataset['test'])} testing examples"
        )

        self.dataset = dataset

    def train_model(self) -> None:
        """Trains the model on the dataset. Automatically called by
        load_model if the user wants to train a new model, or if no
        existing model was found."""
        self.load_data()

        divider("TRAINING MODEL")

        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        print("Setting arguments...", end=" ")
        per_device_train_batch_size = 2
        training_args = TrainingArguments(
            output_dir=self.model_path,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=self.num_train_epochs,
            save_total_limit=2,
            save_strategy="epoch",  # Save after each epoch
            eval_strategy="epoch",  # Evaluate at the end of each epoch
            logging_dir="./logs",
            logging_steps=1,
            logging_strategy="epoch",  # Log at the end of each epoch
            dataloader_pin_memory=False,  # needed if running on Apple Silicon Mac
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            dataloader_num_workers=0 if self.device.type == "cuda" else 4,

        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
        )
        print(
            f"learning rate: {self.learning_rate}, batch size: {per_device_train_batch_size}, epoch count: {self.num_train_epochs}"
        )

        print("Training model...")
        trainer.train()

        print("Saving model...", end=" ")
        trainer.save_model(self.model_path)
        print(f"saved to {self.model_path}")

        print("Extracting loss values...", end=" ")
        train_losses = {}
        eval_losses = {}
        for log in trainer.state.log_history:
            epoch = log["epoch"]
            if "loss" in log:
                train_losses[epoch] = log["loss"]
            if "eval_loss" in log:
                eval_losses[epoch] = log["eval_loss"]
        print(
            f"extracted {len(train_losses)} train losses and {len(eval_losses)} eval losses"
        )

        self.train_losses = train_losses
        self.eval_losses = eval_losses

        self.model = model

    def save_plot(self):
        """Plots the training and evaluation losses and saves it to
        a png file. Can only be called after the model has been
        trained using the `train` method."""

        print(f"Generating plot...", end=" ")

        plt.figure(figsize=(10, 6))
        plt.plot(
            self.train_losses.keys(), self.train_losses.values(), label="Train Loss"
        )
        plt.plot(self.eval_losses.keys(), self.eval_losses.values(), label="Eval Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss")
        plt.legend()

        plt.savefig(self.plot_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory
        print(f"saved plot to {self.plot_path}")

    def load_model(self) -> None:
        """Loads the model if one exists at the specified file path
        and the user wants to load it; otherwise trains a new model,
        then prompts if the user wants to plot the training losses."""
        will_train = True
        if os.path.exists(self.model_path):
            will_load = binary_prompt(
                f"Found existing model at {self.model_path}, (l)oad existing model or (t)rain new model?",
                "l",
                "t",
            )
            will_train = not will_load

        if will_train:
            self.train_model()
            self.save_plot()

        else:
            print(f"Loading existing model...", end=" ")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            print(f"loaded {self.model_path}")

    def load_generator(self) -> None:
        """Loads the text generator. Must be called after loading the model."""

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(f"Loading generator...", end=" ")
        self.generator = pipeline(
            "text-generation", model=self.model_path, tokenizer=self.tokenizer
        )
        print(f"generator loaded from {self.model_path}")

    def generate_text(
        self, prompt: str, *, max_new_tokens=50, temperature=1.0, top_p=0.8
    ) -> str:
        """Generates text using the loaded model. Must be called after
        loading the text generator."""

        if self.generator is None:
            raise ValueError("Generator not loaded. Call load_generator() first.")

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
