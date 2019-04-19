from RACE_dataloader import get_dataloaders
from RACE_transforms import GloVe_Transform

# Load dataloader for train, dev and test
batch_size = 32
transformer = GloVe_Transform(type = 'average')
train_loader, dev_loader, test_loader = get_dataloaders(batch_size = batch_size, transformer = transformer)

# Iterate thru loader 
for (article, question, answer, options, answer_index, article_emb, question_emb, answer_emd) in train_loader:

	# article: string
	# question: string
    # answer: string (element of options)
    # options: list of strings
    # answer_index: index of answer in the options list

    pass