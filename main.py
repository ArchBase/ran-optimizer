# Author Akshay
# License details are provided in the repository

# This project uses many libraries and dependencies, their licenses, terms, copyrights etc. will be applicable here also
"""
Libraries/Frameworks used

numpy
pyarrow
mathplotlib
json
atexit
os
re

pandas
pickle
tensorflow

"""


""" This is the Main python file to get started using Maav model
    You can start training or using the model by running this file
"""

from maav.configuration import config
from maav.trainer import Trainer, Augmented_Trainer
from maav.dataset_preprocessor import Dataset_Preprocessor
from maav.user import User
import maav.plotter
import maav.tester

print("\n\n\n\n\nWelcome to the Morning Model framework\n\n(If you're running first time you may need to bake the dataset first(Read manual.txt for more details))\n\nTrain Model --> 1\nUse Model --> 2\nBake dataset --> 3\nPlot last training logs --> 4\nCreate dataset --> 5\nTest model --> 6\nStart data augmented training section  --> 7")

option = input("\nEnter your option: ")

if option == '1':# Train model
    print("\nTrain new model --> 1\nContinue training --> 2")
    option = input("\nEnter your option: ")
    if option == '1':# Train new model
        trainer = Trainer("new")
        epochs = input("\nEnter epochs: ")
        batch_size = input("\nEnter batch size: ")
        trainer.train_model(epochs=epochs, batch_size=batch_size, save=True)
    elif option == '2':# Train or fine tune existing model
        trainer = Trainer("Train existing model")
        epochs = int(input("\nEnter epochs: "))
        batch_size = int(input("\nEnter batch size: "))
        trainer.train_model(epochs=epochs, batch_size=batch_size, save=True)
elif option == '2':# Use the saved model
    user = User()
    #print(user.tokenizer.index_pair[1])
    while True:
        query = input("\n\n\nEnter your Query>>: ")
        print("\n\n")
        print(user.generate_response(query, length=50))
elif option == '3':# Preprocess dataset
    dataset = Dataset_Preprocessor()
    dataset.process_and_save_dataset()

elif option == '4':# plot last training logs
    maav.plotter.plot()

elif option == '5':# create dataset
    maav.dataset_creator.create_dataset()

elif option == '6':# test model
    print("Note: test the model on test set will overwrite saved dataset")
    orginal_dataset_file_path = config["DATASET_FILE_PATH"]
    orginal_ignore_columns = config["IGNORE_COLUMN_INDICES_TRAIN"]
    config["IGNORE_COLUMN_INDICES_TRAIN"] = config["IGNORE_COLUMN_INDICES_TEST"]
    config["DATASET_FILE_PATH"] = config["TEST_DATASET_FILE_PATH"]
    print("Processing test datset")
    dataset = Dataset_Preprocessor()
    dataset.process_and_save_dataset()

    tester = maav.tester.Tester(dataset)
    tester.generate_predictions()
    tester.save_as_csv()

    config["IGNORE_COLUMN_INDICES_TRAIN"] = orginal_ignore_columns
    config["DATASET_FILE_PATH"] = orginal_dataset_file_path

elif option == '7':# Data augmented training

    try_out = int(input("How many combinations to try out?: "))
    epoch_per_aug = int(input("Enter # of epochs per each combination: "))
    batch_size = int(input("Enter batch_size: "))

    trainer = Augmented_Trainer(try_out_times=try_out, batch_size=batch_size, epoch_per_aug=epoch_per_aug)

    trainer.start_training()


