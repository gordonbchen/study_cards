# study_cards
A hackable and lightweight program to study cards.

## Run
* Install `requirements.txt` (`numpy` and `pandas`)
* run `py main.py (replace_with_card_set_path)`
  * Included example: run `py main.py cards/san_fermin.csv`

## Card set format
* Takes a card set as a .csv file with `front` and `back` columns (ex: `cards/san_fermin_raw.csv` as raw vocab csv)
* Automatically adds `correct` and `incorrect` columns to track progress

## Probability of showing card
* Uses `softmax(incorrect/correct)` to find probability of showing card
  * Higher `incorrect/correct` -> needs more studying -> higher prob of showing card
* Picks a random card using softmax probabilities

## Possible improvements
* Add temperature to modify softmax probabilities, making card selection randomness variable
* Multiple correct answers
* Override wrong answer
* Nicer GUI
