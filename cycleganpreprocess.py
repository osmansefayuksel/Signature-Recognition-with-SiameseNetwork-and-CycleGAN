import shutil
import os
from addnoise import add_noise, add_name
from PIL import Image
import cv2
import numpy as np




root_path = 'C:/Users/osy/Desktop/sign-verification/signature-dataset/sign_data/images/'

for root, dirs, files in os.walk(root_path):
    for filename in files:
        shutil.move(os.path.join(root, filename), 'C:/Users/osy/Desktop/sign-verification/signature-dataset/sign_data/A/')




def process_image(image_path):

    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image = add_name(image)
    image = add_noise(image)

    return image



root_path = 'C:/Users/osy/Desktop/sign-verification/signature-dataset/sign_data/A/'

for root, dirs, files in os.walk(root_path):
    for filename in files:
        image = process_image(os.path.join(root, filename))
        cv2.imwrite(f'C:/Users/osy/Desktop/sign-verification/signature-dataset/sign_data/B/{filename}', image)




srcA_path = 'C:/Users/osy/Desktop/sign-verification/signature-dataset/sign_data/A/'
srcB_path = 'C:/Users/osy/Desktop/sign-verification/signature-dataset/sign_data/B/'

trainA_path = 'C:/Users/osy/Desktop/sign-verification/signature-dataset/sign_data/trainA/'
trainB_path = 'C:/Users/osy/Desktop/sign-verification/signature-dataset/sign_data/trainB/'
testA_path = 'C:/Users/osy/Desktop/sign-verification/signature-dataset/sign_data/testA/'
testB_path = 'C:/Users/osy/Desktop/sign-verification/signature-dataset/sign_data/testB/'


def split_data(src_path, train_path, test_path, split_ratio):
    files = np.array(os.listdir(src_path))
    np.random.shuffle(files)
    split_index = int(split_ratio * len(files))
    testA = files[0:split_index]
    trainA = files[split_index:]
    [shutil.move(os.path.join(src_path, path), os.path.join(train_path, path)) for path in trainA]
    [shutil.move(os.path.join(src_path, path), os.path.join(test_path, path)) for path in testA]

split_data(srcA_path, trainA_path, testA_path, 0.1)
split_data(srcB_path, trainB_path, testB_path, 0.1)





from PIL import Image
import os, sys
size_images = dict()

for dirpath, _, filenames in os.walk(trainA_path):
    for path_image in filenames:
        image = os.path.abspath(os.path.join(dirpath, path_image))
        with Image.open(image) as img:
            width, heigth = img.size
            size_images[path_image] = {'width': width, 'heigth': heigth}





im_size = 512
def make_square(image, min_size=512, fill_color=(255, 255, 255, 0)):
    
    x, y = image.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(image, (int((size - x) / 2), int((size - y) / 2)))
    new_im = new_im.resize((im_size, im_size))
    return new_im



def resize_images(path):
    
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            image = Image.open(path+item)
            image = make_square(image)
            image.save(path+item)



resize_images(trainA_path)
resize_images(trainB_path)
resize_images(testA_path)
resize_images(testB_path)