import os.path
from PIL import Image

img_dir = 'C:\\Users\\kryst\\Desktop\\ShapeRecognitionModel\\Etap3\\STL_RENDERS_SIMPLE_BG\\'

print('Bulk images resizing started...')

for img in os.listdir(img_dir):
	f_img = img_dir + img
	f, e = os.path.splitext(img_dir + img)
	img = Image.open(f_img)
	img = img.resize((299, 299))
	img.save(f + e)

print('Bulk images resizing finished...')