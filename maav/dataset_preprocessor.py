from maav.configuration import config
import maav.configuration as configuration
from maav.tokenizer import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import csv
import numpy as np
import random
import pyarrow.parquet as pq


class Dataset:
    """
    This is the class to Store/Retrieve and to process the dataset
    Methods:
        read_dataset(): To read the dataset from file path specified in config.py
        process_dataset(): To process the readed dataset file
        save(): To save the processed dataset
        load(): To load the saved dataset
    """
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.X_train = []
        self.y_train = []
        self.table = []
        self.num_of_rows_read = 0
        self.y_train_index = config["Y_TRAIN_INDEX"]
        self.ids = []
        

    def read_dataset(self):
        # Path to your Parquet file
        print("Processing csv file:")
        # Read the Parquet file
        print("\tReading csv file")
        with open(config['DATASET_FILE_PATH'], mode='r') as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)

            # Iterate over each row in the CSV file
            for row in csv_reader:
                self.table.append(row)
                
        print("\tConverting to text")
        # Convert the table to a pandas DataFrame
        #self.table = self.table.to_pandas()
        text = ""
        print("\tTokenizing")
        self.num_of_rows_read=0
        for index, row in enumerate(self.table):
            configuration.progress_bar(label="Tokenizing", progress=index+1, total=config["NUM_OF_ROWS_TO_READ"])
            # Format the row as desired
            #row = row[:-1]
            for index, column in enumerate(row):
                if index in config["IGNORE_COLUMN_INDICES_TRAIN"]:
                    continue
                text += f"  {configuration.clean_text(column)}"
                #if index == 1:
                #    text += "__Y_train__"
            text += " \n"
            #text += f" {configuration.clean_text(row['prompt'])}{configuration.clean_text(row['response'])} \n"
            # Write the formatted row to the text file
            self.num_of_rows_read+=1
            if self.num_of_rows_read > config["NUM_OF_ROWS_TO_READ"]:
                break
        with open(config["RAW_DATASET_PATH"], 'w') as file:
            file.write(text)
        self.table = self.table[1:]
        text = configuration.split(text)
        print(f"\n\tRead {self.num_of_rows_read} rows")
        self.tokenizer.fit_on_chars(text)
        configuration.find_best_values()
        
    def process_dataset(self):
        #print(self.table)
        ij=0
        print("\tFormatting dataset")
        for index, row in enumerate(self.table):
            configuration.progress_bar(label="Formatting dataset", progress=index, total=config["NUM_OF_ROWS_TO_READ"])
            #print(len(self.table.iterrows))
            ij+=1
            #print(row['prompt'])
            #self.X_train.append(row['prompt'])
            # Format the row as desired
            #row = row[:-1]
            X_train = []
            y_train = []
            for i, column in enumerate(row):
                if i == 0:
                    self.ids.append(column)

                if i in config["IGNORE_COLUMN_INDICES_TRAIN"]:
                    continue


                if i == self.y_train_index:
                    y_train.append(int(column))
                    continue
                
                if i != config["Y_TRAIN_INDEX"]:
                    X_train.append(int(column))

            self.X_train.append(X_train)
            self.y_train.append(y_train)
                #x_text += f" {column}"
            #self.X_train.append((pad_sequences([self.tokenizer.text_to_sequences(configuration.split(configuration.clean_text(x_text)))], padding='pre', maxlen=config["MAX_SEQUENCE_LENGTH"], truncating='pre')[0]).tolist())
            #self.X_train.append(self.tokenizer.text_to_sequences(configuration.split(configuration.clean_text(x_text))))
            #self.X_train.append((pad_sequences([self.tokenizer.text_to_sequences(configuration.split(configuration.clean_text(row['prompt'])))], padding='pre', maxlen=config["MAX_SEQUENCE_LENGTH"], truncating='pre')[0]).tolist())

            #sequence = (pad_sequences([self.tokenizer.text_to_sequences(configuration.split(configuration.clean_text(row['response'])))], padding='pre', maxlen=config["MAX_OUTPUT_SEQUENCE_LENGTH"], truncating='post')[0]).tolist()
            #print(sequence)
            #y_train = []
            #for i in sequence:
            #    binary_number = bin(i)[2:].zfill(config["VOCAB_NEURONS"])
                #print(f"Check {i} = {binary_number}")
            #    binary_digits = [int(digit) for digit in binary_number]
            ##self.y_train.append(y_train)
            #print(f"Y_train size: {len(y_train)}")
            #if ij > config["NUM_OF_ROWS_TO_READ"]:
            #    break
                

            #self.y_train.append((pad_sequences([self.tokenizer.text_to_sequences(config.split(row['response']))], padding='pre', maxlen=config.MAX_OUTPUT_SEQUENCE_LENGTH, truncating='pre')[0]).tolist())
        #print(f"Sample dataset: \n")
        #print(len(self.X_train), len(self.y_train))
        #print(self.X_train[0], self.y_train[0])
        #print("X_train: " + ' '.join(self.tokenizer.sequence_to_text(self.X_train[0])))
        #print("y_train: " + ' '.join(self.tokenizer.sequence_to_text(self.y_train[0])))
        #print(self.tokenizer.index_pair)
        #print(f"Processed {ij} rows")
             
    def get_processed_dataset(self):
        return self.y_train

    def save(self, save_tokenizer=True):
        print("\n\tSaving dataset")
        if save_tokenizer:
            self.tokenizer.save()
        os.makedirs("saved_model/processed_dataset", exist_ok=True)
        with open('saved_model/processed_dataset/X_train.ip', 'wb') as file:
            pickle.dump(self.X_train, file)
        with open('saved_model/processed_dataset/y_train.op', 'wb') as file:
            pickle.dump(self.y_train, file) 
    def load(self):
        print("\nLoading dataset.")
        self.tokenizer.load()
        with open('saved_model/processed_dataset/X_train.ip', 'rb') as file:
            self.X_train = pickle.load(file)
        with open('saved_model/processed_dataset/y_train.op', 'rb') as file:
            self.y_train = pickle.load(file)
    
    def shuffle(self):# not needed
        self.X_train, self.y_train = zip(*(random.sample(list(zip(self.X_train, self.y_train)), len(self.X_train))))

        


class Trainer_Dataset(Dataset):
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.num_of_rows_read = 0

    def load(self):
        print("\nLoading dataset.")
        #self.tokenizer.load()
        with open('saved_model/processed_dataset/X_train.ip', 'rb') as file:
            self.X_train = pickle.load(file)
        with open('saved_model/processed_dataset/y_train.op', 'rb') as file:
            self.y_train = pickle.load(file)

class Dataset_Preprocessor(Dataset):
    """
    This class provides an easy interface to deal with the Dataset class
    
    Methods:
        process_and_save_dataset(): The single function to bake the dataset
    """
    
    def __init__(self):
        super().__init__()

    def process_and_save_dataset(self):

        #reader = Paraquet_Reader()
        #cleaner = Cleaner()
        print("Processing dataset:")
        self.read_dataset()
        try:
            self.process_dataset()
        except KeyboardInterrupt:
            print("\nYou forcefully stopped dataset processing\n")
            u =  input("Do you want to save current processed dataset? (y/n): ")
            if u != 'y':
                return
        self.save()
        print("\tDataset processed and saved successfully")


