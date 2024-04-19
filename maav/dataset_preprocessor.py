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
        text = ""
        print("\tTokenizing")
        self.num_of_rows_read=0
        for index, row in enumerate(self.table):
            configuration.progress_bar(label="Tokenizing", progress=index+1, total=config["NUM_OF_ROWS_TO_READ"])

            # Format the row as desired
            for index, column in enumerate(row):
                if index in config["IGNORE_COLUMN_INDICES_TRAIN"]:
                    continue
                text += f"  {configuration.clean_text(column)}"
            text += " \n"

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
        ij=0
        print("\tFormatting dataset")
        for index, row in enumerate(self.table):
            configuration.progress_bar(label="Formatting dataset", progress=index, total=config["NUM_OF_ROWS_TO_READ"])
            ij+=1

            # Format the row as desired
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


