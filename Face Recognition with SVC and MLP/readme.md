### This is pretty trivial to implement

### The hard part is deciding who you want to recognize, and then collecting a ton of photos.

Save them to the dataset directory. 

process.py will take the whole dataset and quantify every face with 128-D encodings,
and save them with their associated names.

It uses a pre-trained model to find faces in a photo, and another pre-trained model that quantifies a face into an embedding.

tran_model.py will take those embeddings and names, and do the machine learning.

That means learning to recognize who each face belongs to.

A default SVC or MLP model is maybe good enough, but I've tried tweaking some custom built models to get better accuracy. 

But in any case, better data - more photos, and better quality photos - is the key to better accuracy.

classify.py is also pretty trivial. Just upload a photo. It will find any faces in it,
and compare them with the faces that the model has been trained to recognize.

In any case, again, the model is only as good as the data it has been trained with. A few photos per each person will not
be able to recognize anyone very well.
