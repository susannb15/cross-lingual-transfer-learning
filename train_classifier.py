# imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import torch
import os
import pickle

# load meta data
df = pd.read_csv("probing.meta")

for i in range(13):
	print(f"Train linear classifier on layer {i}...")
	data_all = torch.load(os.path.join("representations_native", "representations_"+str(i)))
	train = data_all[:-33]
	test = data_all[-33:]
	labels_all = df["token"].tolist()
	tr_labels = labels_all[:-33]
	tt_labels = labels_all[-33:]
	lr_clf = LogisticRegression(random_state=42).fit(train, tr_labels)
	predictions = lr_clf.predict(test)
	scores = classification_report(predictions, tt_labels)
	print(f"Training finished. Score on test data: {scores}")
	filename = os.path.join("classifiers", "clf_"+str(i))
	pickle.dump(lr_clf, open(filename, 'wb'))
	close(filename)
