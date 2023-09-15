# imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import torch
import os
import pickle

class Probing:
	def __init__(self, clf, reps):
		with open(clf, 'rb') as f:
			self.clf = pickle.load(f)
		self.reps = torch.load(reps)
		df = pd.read_csv("probing.meta")
		self.labels = df["token"].tolist()
		
	def _predict(self):
		score = self.clf.score(self.reps, self.labels)
		return score

