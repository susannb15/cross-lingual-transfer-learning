from xml.etree import cElementTree as ET
from collections import defaultdict

with open("tiger.xml", encoding="ISO-8859-1") as f:
	text = f.read()

with open("tiger_UTF-8.txt", "w+", encoding='utf-8') as f:
	d = defaultdict(lambda: defaultdict())
	counter_id = 0
	root = ET.fromstring(text)
	for s in list(root)[1]:
		for graph in s:
			counter_id += 1
			sent = []
			pos_tags = []
			for t in list(graph)[0]:
				terminal = t.get('word')
				pos = t.get('pos')
				sent.append(terminal)
				pos_tags.append(pos)
			d[counter_id]['text'] = ' '.join(sent)
			d[counter_id]['pos'] = pos_tags
			f.write(' '.join(sent)+'\n')

for i in range(1, 5):
	print(d[i])
