
from maav.trainer import Trainer
from maav.dataset_preprocessor import Dataset_Preprocessor
from maav.user import User
import maav.plotter
import maav.dataset_creator

"""
If you can understand how to manually use main.py just run this file to run complete cycle of model

this file will do the following in order

1.Generate dataset
2.Process the dataset generated for training
3.Train the model
4.Making it available to use the model
5.Plot the training loss and val_loss in a graph

"""

print("\n\nGenerating dataset")

maav.dataset_creator.create_dataset()


print("\n\nProcessing dataset")

dataset = Dataset_Preprocessor()
dataset.process_and_save_dataset()


del dataset

print("\n\nTrain model")

trainer = Trainer("new")
epochs = input("\nEnter epochs: ")
batch_size = input("\nEnter batch size: ")
trainer.train_model(epochs=epochs, batch_size=batch_size, save=True)


del trainer

print("\n\nUse model")


user = User()
print("Type <#bye#> to exit")
#print(user.tokenizer.index_pair[1])
while True:
    query = input("\n\n\nEnter your Query>>: ")
    if query == "<#bye#>":
        break
    print("\n\n")
    print(user.generate_response(query, length=50))

del user

maav.plotter.plot()