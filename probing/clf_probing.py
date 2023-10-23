# imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import torch
import os
import pickle
from probing_module import Probing
import numpy as np

# probing task for de on de
a = np.arange(5000,105000,5000)
with open("probing_de_shuffle_90.csv", "w+", encoding='utf-8') as f:
	f.write("layer,checkpoint,score")

for i in range(13):
	clf = os.path.join("classifiers", "clf_"+str(i))
	for j in a:
		rep_path = os.path.join("representations_de_shuffle_90", str(j))
		reps = os.path.join(rep_path, "representations_"+str(i))
		prob = Probing(clf, reps)
		score = prob._predict()
		with open("probing_de_shuffle_90.csv", "a", encoding='utf-8') as f:
			f.write("\n"+str(i)+","+str(j)+","+str(score))
