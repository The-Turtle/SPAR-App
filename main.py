from util import divider, binary_prompt, nonempty_prompt
from finetunedmodel import FineTunedModel
import os

if __name__ == "__main__":
    txt_files = [f.removesuffix(".txt") for f in os.listdir("datasets") if f.endswith(".txt")]

    while True:
        id = input(
            f"Which model would you like to load or train? The options are {txt_files}: "
        )
        if id not in txt_files:
            print("Invalid option.")
        else:
            break

    model_name = "distilgpt2"
    model_path = f"models/{id}"
    data_path = f"datasets/{id}.txt"
    plot_path = f"plots/{id}.png"

    num_train_epochs = 8

    model = FineTunedModel(
        model_name=model_name,
        model_path=model_path,
        data_path=data_path,
        plot_path=plot_path,
        num_train_epochs=num_train_epochs,
    )
    model.load_model()

    model.load_generator()

    divider("TEXT GENERATION")

    while True:
        prompt = nonempty_prompt("Enter a prompt:")

        print("Generated text:", model.generate_text(prompt, max_new_tokens=100))

        will_continue = binary_prompt(
            "Would you like to (c)ontinue or (q)uit?", "c", "q"
        )
        if not will_continue:
            print("Exiting.")
            exit(0)
