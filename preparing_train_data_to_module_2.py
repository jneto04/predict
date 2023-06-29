import pandas as pd

def prepare_data(data_path):
	d = {'input':[], 'target':[]}

	df = pd.read_csv(data_path, sep='\t')

	triples = df['input'].values.tolist()
	verbalizations = df['target'].values.tolist()

	for t, v in zip(triples, verbalizations):
		text_t = t.split('TRIPLES: ')[1]
		text_verb = v.replace('|s|>', '<|s|>')
		text_verb = text_verb.replace('|o|>', '<|o|>')
		text_verb = text_verb.replace('|b|>', '<|b|>')
		d['input'].append(text_t)
		d['target'].append(text_verb)

	train_dataset = pd.DataFrame.from_dict(d)

	train_dataset.to_csv('./train_module_2.csv', sep=',', encoding='utf-8', columns=['input', 'target'], index=False)

prepare_data('./predicted_sentences_to_module_2.csv')