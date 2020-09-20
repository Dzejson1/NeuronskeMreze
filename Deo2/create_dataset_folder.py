import os, shutil
import constants as c

def ucitaj():    
    
    
    
    if(os.path.exists(c.base_dir)):     
        print("Exist path")
        return

    os.mkdir(c.base_dir)       
    os.mkdir(c.train_dir)
    os.mkdir(c.validation_dir) 
    os.mkdir(c.test_dir)
    os.mkdir(c.train_cats_dir)
    os.mkdir(c.train_dogs_dir)       
    os.mkdir(c.validation_cats_dir)    
    os.mkdir(c.validation_dogs_dir)   
    os.mkdir(c.test_cats_dir)   
    os.mkdir(c.test_dogs_dir)
    
  
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(c.original_dataset_dir, fname)
        dst = os.path.join(c.train_cats_dir, fname)
        shutil.copyfile(src, dst)
    
   
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(c.original_dataset_dir, fname)
        dst = os.path.join(c.validation_cats_dir, fname)
        shutil.copyfile(src, dst)
        
  
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(c.original_dataset_dir, fname)
        dst = os.path.join(c.test_cats_dir, fname)
        shutil.copyfile(src, dst)
        
   
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(c.original_dataset_dir, fname)
        dst = os.path.join(c.train_dogs_dir, fname)
        shutil.copyfile(src, dst)
        
  
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(c.original_dataset_dir, fname)
        dst = os.path.join(c.validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
        
   
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(c.original_dataset_dir, fname)
        dst = os.path.join(c.test_dogs_dir, fname)
        shutil.copyfile(src, dst)
        
ucitaj()        