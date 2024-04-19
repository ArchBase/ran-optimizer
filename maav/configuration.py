import json
import atexit
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'  # or '3' to additionally suppress INFO messages

import tensorflow as tf

print("Started")

config = None

# Read 
# the JSON file
with open("maav/config.json", "r") as file:
    config = json.load(file)


def find_best_values():# not needed
    # Find the best suited output neurons length
        for i in range(config["MAX_VOCAB_FACTOR"]):

            if (2**i) > config["VOCABULARY_SIZE"]:
                config["VOCAB_NEURONS"] = i
                print(f"\t\tbest fitting output neuron size: {config['VOCAB_NEURONS']}")
                break
            if i == config["MAX_VOCAB_FACTOR"] - 1:
                print("Failed to find best vocab size")
                quit()


def to_key(number):# not needed
    return int(number)

def to_value(key):# not needed
    return int(key)


def split(input_string):# not needed
    return input_string.split()

def close():
    # Write the updated dictionary back to the JSON file
    with open("maav/config.json", "w") as file:
        json.dump(config, file, indent=4)
    print()

def progress_bar(label, progress, total):
    percent = 100 * (progress / float(total))
    print(f"{label}: {progress}/max     {round(percent, 3)}%" ,end='\r')

def clean_text(text):# not needed
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text


atexit.register(close)