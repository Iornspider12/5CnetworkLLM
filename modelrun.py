# Google colabe link : "'https://colab.research.google.com/drive/123XjkVsZdUKmqKTQUj07gdFt93EHaSOe'"



import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import nltk
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import os
import nltk
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"


dataset = load_dataset('csv', data_files='impression_300_llm.csv')


train_size = 250
train_dataset = dataset['train'].select(range(train_size))
eval_dataset = dataset['train'].select(range(train_size, len(dataset['train'])))


model_name = "gpt2"  


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  
model = AutoModelForCausalLM.from_pretrained(model_name)


def preprocess_function(examples):
    inputs = [" ".join([report, history, observation]) for report, history, observation in zip(
        examples['Report Name'], examples['History'], examples['Observation'])]
    targets = examples['Impression']

    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)

   
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)["input_ids"]


    model_inputs["labels"] = labels
    return model_inputs


tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

trainer.train()


eval_results = trainer.evaluate()
print(eval_results)


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def preprocess_text(text, target_word):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.lower() not in stop_words]
    stemmer = nltk.PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    lemmatizer = nltk.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    if target_word in words:
        print(f"'{target_word}' is present in the processed text.")
    elif stemmer.stem(target_word) in words:
        print(f"'{target_word}' (stemmed) is present in the processed text.")
    else:
        print(f"'{target_word}' is not present in the processed text.")
    
    return words


text = "This is an example sentence for preprocessing."
target_word = "example"
processed_words = preprocess_text(text, target_word)


w2v_model = Word2Vec([processed_words], min_count=1)


if target_word in w2v_model.wv:
    word_pairs = w2v_model.wv.similar_by_word(target_word, topn=5)
elif stemmer.stem(target_word) in w2v_model.wv:
    word_pairs = w2v_model.wv.similar_by_word(stemmer.stem(target_word), topn=5)
else:
    print(f"Neither '{target_word}' nor its stemmed version found in vocabulary.")
    word_pairs = None


if word_pairs:
    x = []
    y = []
    for word, _ in word_pairs:
        x.append(w2v_model.wv[word][0])
        y.append(w2v_model.wv[word][1])

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    for i, word in enumerate(word_pairs):
        plt.annotate(word[0], (x[i], y[i]))
    plt.title('Top Word Pairs Based on Similarity')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid()
    plt.show()