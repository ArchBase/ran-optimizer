import pandas as pd
import maav.configuration as config
import random

def create_dataset():
    number_words = {
        '0': "zero",
        '1': "one",
        '2': "two",
        '3': "three",
        '4': "four",
        '5': "five",
        '6': "six",
        '7': "seven",
        '8': "eight",
        '9': "nine",
        '.': " point ",
        "-": "negative"
    }

    def get_in_words(number):
        num_digits = [number_words[dig] for dig in str(number)]
        return ' '.join(num_digits)

    def get_formatted(string):
        return string
        return ' '.join(string.split('-'))


    # Define the range of numbers and number of samples
    MIN_LIMIT = 1
    MAX_LIMIT = 1000
    NUM_OF_SAMPLES = 100000

    # Create a list of operations
    operations = ['add', 'subtract', 'multiply', 'divide']

    # Initialize empty lists for questions and answers
    questions = []
    answers = []

    # Generate random mathematical operations
    for _ in range(NUM_OF_SAMPLES):
        
        config.progress_bar("Generating dataset: ", _, NUM_OF_SAMPLES)

        num1 = random.randint(MIN_LIMIT, MAX_LIMIT)
        num2 = random.randint(MIN_LIMIT, MAX_LIMIT)
        num1_word = get_formatted(get_in_words(num1))
        num2_word = get_formatted(get_in_words(num2))
        operation = random.choice(operations)

        if operation == 'add':
            question = f"what is {num1_word} plus {num2_word}"
            answer = "\tIt is " + get_formatted(get_in_words(num1 + num2))
        elif operation == 'subtract':
            question = f"what is {num1_word} minus {num2_word}"
            answer = "\tIt is " + get_formatted(get_in_words(num1 - num2))
        elif operation == 'multiply':
            question = f"what is {num1_word} multiplied_by {num2_word}"
            answer = "\tIt is " + get_formatted(get_in_words(num1 * num2))
        elif operation == 'divide':
            question = f"what is {num1_word} divided_by {num2_word}"
            answer = "\tIt is " + get_formatted(get_in_words(round(num1 / num2, 2)))
        elif operation == 'modulus':
            question = f"what is {num1_word} modulus {num2_word}"
            answer = "\tIt is " + get_formatted(get_in_words(round(num1 % num2, 2)))

        questions.append(question)
        answers.append(answer)

    # Create a DataFrame from the lists
    df = pd.DataFrame({'prompt': questions, 'response': answers})

    # Save the DataFrame as a Parquet file
    df.to_parquet('dataset_created.parquet', index=False)

