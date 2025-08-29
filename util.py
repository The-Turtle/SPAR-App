# helper function to draw a divider in the console
def divider(text: str, l: int = 80) -> None:
    n = len(text)
    split = max((l - n) // 2, 0)
    print("=" * split + text + "=" * split)


# helper function to get a response from the user. The response
# must be either choice1 or choice2 in order to continue.
def binary_prompt(prompt: str, choice1: chr, choice2: chr) -> bool:
    choice1 = choice1.lower()
    choice2 = choice2.lower()
    while True:
        user_choice = input(prompt + " ").strip().lower()
        if user_choice == choice1:
            return True
        elif user_choice == choice2:
            return False
        else:
            print(f"Invalid input. Please enter '{choice1}' or '{choice2}'.")


# helper function to get a nonempty string from the user.
def nonempty_prompt(prompt: str) -> str:
    while True:
        prompt = input(prompt + " ")
        if prompt.strip() == "":
            print("Error: empty string.")
        else:
            return prompt
