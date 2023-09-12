# imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import torch

# load meta data
df = pd.read_csv("probing.meta")

for i in range(13):
	print(f"Train linear classifier on layer {i}...")
	data_all = torch.load("representations_"+str(i))
	train = data_all[:-33]
	test = data_all[-33:]
	labels_all = df["token"].tolist()
	tr_labels = labels_all[:-33]
	tt_labels = labels_all[-33:]
	print(len(test), len(tr_labels))
	lr_clf = LogisticRegression(random_state=42).fit(train, tr_labels)
	predictions = lr_clf.predict(test)
	scores = classification_report(predictions, tt_labels)
	print(f"Training finished. Score on test data: {scores}")
