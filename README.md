# Ghostwriter AI ✍
This was a small, over-the-weekend kind of project to explore using RNN's with LSTM for automatically generating text character-by-character. The model architecture looks something like this:

- A number of GRUs stacked on top of one another, doing the heavy lifting in terms of the aforementioned *"RNN with LSTM"*
- A number of dense, fully-connected layers transferring the output of the GRUs to a vector of probabilities for the next character

Generation is then done by (more or less) randomly sampling from the resulting distributions, given an initial "seed" character to serve as the initial input to the model.
The sampled character's index is then appended to a list, which holds all randomly sampled character indices up to this point, and is then fed through the model to produce the next sample.
This is repeated a set number of times to yield the new text.

## Example results
The final model, trained on Harry Potter and the Goblet of Fire, is available in this repo as `textnet_base.pth`. This ended up being capable of producing such striking pieces of writing as the following:


>"You, mentioning your mother," said Hermione. "Nooly!"
>Foel.  Mrs. Weasley syets one once whispered. "Roosport."  said Voldemort. "Snape," she said.  "You will too sound to think they's thirtened," said Sroush, his stormwerts, murding tillmor up.  "Mrss you stopped two tony, sosting something."
>It was something.


Perhaps not that likely to put JK Rowling out of her job anytime soon, but still mildly entertaining in its own right!

## Using this yourself
The code is currently very poorly documented, and I am not sure when (if at all) I will get around to fixing that 🤷‍♂️
However, the main structure looks something like this:

- `model.py` holds the `TextNet` model definition
- `data.py` contains a `Dataset` implementation for reading and processing in .txt files for the network
- `train.py` is where you will find the training loop
- `generate.py` has a class for generating new text samples given a trained model and its corresponding dataset (as loaded by the class in `data.py`)

Notice that both `train.py` and `generate.py` have some snippets in their main clauses which to some extent demonstrate how to run the code.

### Dependencies
You will need to have `pytorch` and `numpy` installed. I worked in Python version 3.8.10, but I imagine any relatively recent Python3 version should do.
