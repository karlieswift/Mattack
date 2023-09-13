from evaluate import load
bertscore = load("bertscore")
predictions = ["cat",'plane',"cat",'sky']
references = ["dog",'jet',"sky",'plane']
results = bertscore.compute(predictions=predictions, references=references, lang="bert-base-uncased")
print(results['f1'])
results = bertscore.compute(predictions=predictions, references=references, model_type="t5-base")
print(results['f1'])



predictions = ["cat in the tree",'cat in the tree']
references = ["cat in the tree","cat in the tree"]
results = bertscore.compute(predictions=predictions, references=references, lang="bert-base-uncased")
print(results['f1'])
results = bertscore.compute(predictions=predictions, references=references, model_type="t5-base")
print(results['f1'])



predictions = ["cat in the grass",'cat in the tree','people in the tree','people and cat in the tree']
references = ["cat in the tree","dog in the tree",'people in the sky','cat and dog in the tree']
results = bertscore.compute(predictions=predictions, references=references, lang="bert-base-uncased")
print(results['f1'])
results = bertscore.compute(predictions=predictions, references=references, model_type="t5-base")
print(results['f1'])



answers = ['A plane is parked on the runway in the middle of the day.',
           'A fighter jet flying in a field with a sky background.',
           'A large propeller plane flying through a blue sky.', 'A statue of a bear on a wooden bench.',
           'A statue of a bear sitting on a wooden bench.', 'A large statue of a bear on a wooden bench.']

bertscore = load("bertscore")
references =[answers[0]]
for answer in answers[1:]:
    predictions=[answer]
    bert_score = bertscore.compute(predictions=predictions, references=references, lang="bert-base-uncased")
    print("bertscore:{}  ".format( round(bert_score['f1'][0],3)))


answers = ['A plane is parked on the runway in the middle of the day.',
           'A fighter jet flying in a field with a sky background.',
           'A large propeller plane flying through a blue sky.', 'A statue of a bear on a wooden bench.',
           'A statue of a bear sitting on a wooden bench.', 'A large statue of a bear on a wooden bench.']

bertscore = load("bertscore")
references =[answers[0]]
for answer in answers[1:]:
    predictions=[answer]
    bert_score = bertscore.compute(predictions=predictions, references=references, model_type="t5-base")
    print("bertscore:{}  ".format( round(bert_score['f1'][0],3)))

