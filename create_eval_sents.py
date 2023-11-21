import pandas as pd

# read in nouns and verbs
nouns = pd.read_csv("nouns_labeled.csv")
verbs = pd.read_csv("verbs_labeled.csv")


# template 1: simple - DET NOUN VERB *POSS*
simple = list()
num_m = 0
num_f = 0
for idx_n, row_n in nouns.iterrows():
	# build subject DET NOUN
	if row_n.gender == "m" and row_n.number == "sg":
		subj = ("Der "+row_n.word, "m")
		num_m += 1
	elif row_n.gender == "n" and row_n.number == "sg":
		subj = ("Das "+row_n.word, "m")
		num_m += 1
	else:
		subj = ("Die "+row_n.word, "f")
		num_f += 1
	for idx_v, row_v in verbs.iterrows():
		if row_n.number == row_v.number:
			sent = subj[0]+" "+row_v.word
			simple.append((sent, subj[1]))
print(f"Checking for label balance m - f: {num_m - num_f}")
with open("eval_sents_simple.txt", "w+", encoding='utf-8') as f:
	f.write("sent\tlabel")
	for sent, label in simple:
		f.write("\n"+sent+"\t"+label)

# template 2: adv - DET NOUN VERB ADV *POSS* increased distance by one word

# template 3: genitiv - DET NOUN GEN(DET NOUN) VERB *POSS* genitiv as attractor
# create separate datasets: one with attractor (different gender than SUBJ) and one without (POSS matches both SUBJ and GENITIV)

# template 4: nebensatz - DET NOUN, REL(PP??), VERB
