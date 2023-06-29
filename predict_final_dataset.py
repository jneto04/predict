import pandas as pd

def preparing_inputs(inputs):
	d = {'input':[]}

	new_subjects, new_objects = [], []
	subjects, relations, objects = [], [], []

	for inpt in inputs:
		text_triples = inpt.split('trpl2txt: ')[1]
		splited_triples = text_triples.split(' & ')
		for triple in splited_triples:
			triple = triple.replace("(", "")
			triple = triple.replace(")", "")
			triple = triple.strip()
			subjects.append(triple.split(', ')[0])
			relations.append(triple.split(', ')[1])
			objects.append(triple.split(', ')[2])

		for s in subjects:
			if s in objects:
				special_token = '<|b|> '+s
				new_subjects.append(special_token)
			else:
				special_token = '<|s|> '+s
				new_subjects.append(special_token)

		for o in objects:
			if o in subjects:
				special_token = '<|b|> '+o
				new_objects.append(special_token)
			else:
				special_token = '<|o|> '+o
				new_objects.append(special_token)

		txt_triple = ''
		for s, r, o in zip(new_subjects, relations, new_objects):
			txt_triple+=s+' '+r+' '+o

		new_subjects, new_objects = [], []
		subjects, objects = [], []

		d['input'].append(txt_triple)

	return d


df_test_v3 = pd.read_csv('./inputs_test_v3.0.csv', sep='\t')

to_predict = df_test_v3['input'].values.tolist()

d = preparing_inputs(to_predict)

to_pred_inputs = pd.DataFrame.from_dict(d)

to_pred_inputs.to_csv('./inputs_test_v3_formated_strategy.csv', sep=',', encoding='utf-8', columns=['input'], index=False)