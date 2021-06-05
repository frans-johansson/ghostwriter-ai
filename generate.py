import argparse
import torch
from data import encode_one_hot, decode_indices, TextFileData


class TextGenerator:
    def __init__(self, data_set, p_margin=0.1):
        self.char_map = data_set.features
        self.n_chars = data_set.n_features
        self.p_margin = p_margin

    def generate_sample(self, model, n_output, seed="a", burn_in=100, device="cpu"):
        hn = model.init_hidden().to(device)
        inp = encode_one_hot(seed, self.char_map).to(device)

        idxs = torch.empty(n_output).to(dtype=torch.int, device=device)
        idxs[0] = (inp[0, 0, ...] == 1).nonzero().squeeze(0)

        model.eval()

        # Burn in loop to generate warm start for context
        for _ in range(burn_in):
            with torch.no_grad():
                out, hn = model(inp, hn)
            inp = torch.zeros(1, 1, self.n_chars, device=device)

        # Actual generator loop
        inp = encode_one_hot(seed, self.char_map).to(device)
        for i in range(n_output-1):
            with torch.no_grad():
                out, hn = model(inp, hn)
                P = (torch.rand(1, device=device) *
                     (1-self.p_margin)) + self.p_margin
                char = (out.cumsum(2) > P).nonzero()[0, 2]

            inp = torch.zeros(1, 1, self.n_chars, device=device)
            inp[..., char] = 1
            idxs[i+1] = (inp[0, 0, ...] == 1).nonzero().squeeze(0)

        model.train()

        return decode_indices(idxs, self.char_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate some new Harry Potter prose")
    parser.add_argument(
        "--seed", "-s", default="A", nargs="?", type=str, help="Initial character to start the prose with")
    parser.add_argument("-n", default=200, nargs="?", type=int,
                        help="Number of characters to generate")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    text_data = TextFileData("data/goblet_book.txt")
    generator = TextGenerator(text_data)
    model = torch.load("models/textnet_base.pth", map_location=device)

    print(generator.generate_sample(model, args.n, seed=args.seed, device=device))
