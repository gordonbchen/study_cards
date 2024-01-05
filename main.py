import pandas as pd
import numpy as np

from argparse import ArgumentParser


def read_cards(file_path: str) -> pd.DataFrame:
    """Read study cards."""
    cards = pd.read_csv(file_path)
    if "correct" not in cards.columns:
        # Add correct and incorrect columns. Start with 1 of each (50-50).
        cards["correct"] = 1
        cards["incorrect"] = 1
    
    return cards


def calc_incorrect_softmax(cards: pd.DataFrame) -> np.ndarray:
    """
    Calculate softmax over incorrect percentages.
    Gives probability of picking card. High incorrect -> high probability.
    """
    incorrect_percents = cards["incorrect"] / cards["correct"]

    incorrect_percents -= incorrect_percents.max()  # For softmax numerical safety.
    e_incorrect_percents = np.exp(incorrect_percents.values)
    incorrect_softmax = e_incorrect_percents / np.sum(e_incorrect_percents)
    return incorrect_softmax


def get_random_ind(softmax_probs: np.ndarray) -> int:
    """
    Choose random ind with softmax probs.
    Higher temp = higher sureness (more probable to pick incorrect).
    """
    ind = np.random.choice(
        np.arange(len(softmax_probs)),
        p=softmax_probs
    )
    return ind


if __name__ == "__main__":
    # Parse args to find path to cards.
    parser = ArgumentParser()
    parser.add_argument("path", help="path to cards csv file")

    args = parser.parse_args()
    cards_file_path = args.path

    # Read cards.
    cards = read_cards(cards_file_path)

    # Print instructions.
    print("Welcome to study_cards!")
    print("Enter \"QUIT\" at anytime to end the program.")

    # Run study loop.
    while True:
        incorrect_softmax = calc_incorrect_softmax(cards)
        ind = get_random_ind(incorrect_softmax)

        front_word = cards.loc[ind, "front"]
        back_word = cards.loc[ind, "back"]
        guess = input(f"\n{front_word}: ")

        if guess == "QUIT":
            break

        if guess != back_word:
            print(f"Wrong. {front_word} = {back_word}, not {guess}.")
            cards.loc[ind, "incorrect"] += 1
        else:
            print(f"Correct. {front_word} = {back_word}.")
            cards.loc[ind, "correct"] += 1

    # Update csv file.
    cards.to_csv(cards_file_path, index=False)

    # Print thank you.
    print("\nThanks for using study_cards!")
