import os, shutil

original_dataset_dir = 'C:/Users/Korisnik/Pop/Pmf/Pmf/Trece godina/Semestar II/Neuronske Mreze/Projekat/NN/dogs cats/dogs-vs-cats/train/train'
base_dir = 'C:/Users/Korisnik/Pop/Pmf/Pmf/Trece godina/Semestar II/Neuronske Mreze/Projekat/NN/dogs cats/dogs-vs-cats/train/example'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')