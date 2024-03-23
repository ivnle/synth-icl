from datasets import load_dataset
from tqdm import tqdm
from task import LanguageModelTokenizer
import numpy as np
import matplotlib.pyplot as plt
import statistics

dataset = load_dataset("roneneldan/TinyStories")
tokenizer = LanguageModelTokenizer()

for split in ["train", "validation"]:
    # Tokenize dataset and store token lengths
    token_lengths = []
    for i, example in enumerate(tqdm(dataset[split])):
        text = example["text"]
        tokens = tokenizer.encode(text, False, False)
        text_len = len(tokens)
        # if text_len < 50:
        #     print(f"{text=}")
        #     print(f"{tokens=}")
        #     continue

        token_lengths.append(text_len)
        # if i > 100000:
        #     break

    # Calculate summary statistics
    avg_token_length = statistics.mean(token_lengths)
    std_deviation = statistics.stdev(token_lengths)
    median_token_length = statistics.median(token_lengths)
    min_token_length = min(token_lengths)
    max_token_length = max(token_lengths)

    # Print out summary statistics
    print("Summary Statistics:")
    print(f"Average Token Length: {avg_token_length}")
    print(f"Standard Deviation: {std_deviation}")
    print(f"Median Token Length: {median_token_length}")
    print(f"Minimum Token Length: {min_token_length}")
    print(f"Maximum Token Length: {max_token_length}")
    # total number of tokens
    print(f"Total Number of Tokens: {sum(token_lengths)}")
    # number of examples
    print(f"Number of Examples: {len(token_lengths)}")

    plt.hist(token_lengths, bins=100)
    # x axis title: tokens
    plt.xlabel("Tokens")
    # y axis title: count
    plt.ylabel("Count")
    plt.show()
    # save image
    plt.savefig(f"tiny-stories-hist-{split}.png")
