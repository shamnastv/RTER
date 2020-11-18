import fasttext

dataset = 'data/fil9'
# model = fasttext.train_unsupervised(dataset)
# model = fasttext.train_unsupervised('data/fil9', minn=2, maxn=5, dim=300)
model = fasttext.train_unsupervised('data/fil9')

print(model.get_word_vector("enviroment"))
model.save_model("model_new")

model = fasttext.load_model("model_new")
print(model.get_word_vector("enviroment"))

print('model saved')
# model = fasttext.load_model("model")
# x = model.get_word_vector("the")

