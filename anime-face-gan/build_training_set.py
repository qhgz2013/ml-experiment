import os
import cv2
import sys

cascade_file = "lbpcascade_animeface.xml"
if not os.path.isfile(cascade_file):
	raise RuntimeError("%s: not found" % cascade_file)
cascade = cv2.CascadeClassifier(cascade_file)

def detect(filename):
	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)
	
	faces = cascade.detectMultiScale(gray,
									 # detector options
									 scaleFactor = 1.1,
									 minNeighbors = 5,
									 minSize = (24, 24))
	return faces

def scan_all_files(path):
	files = os.listdir(path)
	ret = []
	for file in files:
		if os.path.isfile(os.path.join(path, file)) and (file[-3:] == 'png' or file[-3:] == 'jpg' or file[-4:] == 'jpeg'):
			ret.append(os.path.join(path, file))
	return ret

def crop_image(path, rect, dest_res, expand_ratio = 1.5):
	x, y, w, h = rect
	image = cv2.imread(path, cv2.IMREAD_COLOR)
	#scaling window size to fit the full face (default 2x)
	x -= int(w / expand_ratio / 2)
	y -= int(h / expand_ratio / 2)
	w = int(w * expand_ratio)
	h = int(h * expand_ratio)
	
	if x < 0 or y < 0 or (x + w) >= image.shape[1] or (y + h) >= image.shape[0]:
		return None
	
	cropped_image = image[y:y+h,x:x+w]
	resized_image = cv2.resize(cropped_image, (dest_res, dest_res), interpolation = cv2.INTER_CUBIC)
	if (w != h):
		print('Warning: width is not equal to height, resizing constraint failed')
	return resized_image

def build_training_set(path_in, path_out, dest_res = 100):
	files = scan_all_files(path_in)
	#creating directories containing training set
	if not os.path.exists(path_out):
		os.makedirs(path_out)
	
	idx = 0
	file_len = len(files)
	file_idx = 0
	for file in files:
		sys.stdout.write('Processing file %d/%d\r' % (file_idx, file_len))
		sys.stdout.flush()
		faces = detect(file)
		for rect in faces:
			image = crop_image(file, rect, dest_res)
			if image is not None:
				cv2.imwrite(os.path.join(path_out, '%05d.png') % idx, image)
				idx = idx + 1
		file_idx = file_idx + 1