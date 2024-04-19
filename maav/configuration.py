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



def to_key(number):
    return int(number)

def to_value(key):
    return int(key)


def split(input_string):
    return input_string.split()

def close():
    #print("Closed after saving weights")
    # Write the updated dictionary back to the JSON file
    with open("maav/config.json", "w") as file:
        json.dump(config, file, indent=4)
    print()

def progress_bar(label, progress, total):
    percent = 100 * (progress / float(total))
    print(f"{label}: {progress}/max     {round(percent, 3)}%" ,end='\r')

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text
    # Define a pattern for matching special characters
    pattern = r'[^A-Za-z0-9\s]'  # Matches any character that is not a letter, digit, or whitespace

    # Use the sub method of the re module to replace matched characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
            
    # Convert the cleaned text to lowercase
    cleaned_text = cleaned_text.lower()

    cleaned_text = cleaned_text.replace('\n', ' ')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

atexit.register(close)