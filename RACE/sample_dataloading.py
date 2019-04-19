from RACE_dataloader import get_dataloaders
from RACE_transforms import GloVe_Transform

# Load dataloader for train, dev and test
train_loader, dev_loader, test_loader = get_dataloaders(batch_size = 32)

# Iterate thru loader 
for (article, question, answer, options, answer_index) in train_loader:
    
	# article: string
	# question: string
    # answer: string (element of options)
    # options: list of strings
    # answer_index: index of answer in the options list

    # Initalize transform 
    transformer = GloVe_Transform()

    # Convert a string to embedding
    transformer.embed_and_average(answer[0]).shape 
    
    break
