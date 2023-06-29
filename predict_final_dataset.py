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

inputs = df_test_v3['input'].values.tolist()

d = preparing_inputs(inputs)

######
#PRED#
######

to_predict = d['input']

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="/home/netojoaquim/predict/modulo2/").to('cuda')

count, _count = 0, 0
with open('./hypothesis.txt', mode='w+', encoding='utf-8') as f:
  for txt in to_predict:
    inputs = tokenizer(txt, max_length=1024, return_tensors="pt").to('cuda')
    outputs = model.generate(max_length=1024, **inputs)
    verbalization = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    f.write(verbalization+'\n')
    if _count == 50:
      count+=50
      print(' :: Processing: '+str(count)+'/'+str(len(to_predict)))
      _count = 0
    _count+=1
  f.close()