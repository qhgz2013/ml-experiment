#this version uses multi-processing tech. to make max. usage of CPU

#importing v1 functions
from build_training_set import detect, scan_all_files, crop_image
import multiprocessing
import time
import os
import cv2
import sys

idx = None
file_idx = None

def global_var_init(idx_, file_idx_):
	global idx, file_idx
	idx = idx_
	file_idx = file_idx_

def on_compute_callback(input_path, output_path, dest_res):
	global idx, file_idx
	faces = detect(input_path)
	for rect in faces:
		image = crop_image(input_path, rect, dest_res)
		if image is not None:
			with idx.get_lock():
				cv2.imwrite(os.path.join(output_path, '%05d.png') % idx.value, image)
				idx.value += 1
	with file_idx.get_lock():
		file_idx.value += 1

def test_callback():
	print('yes, it works')

def build_training_set_v2(path_in, path_out, dest_res = 100, process_count = None):
	if process_count is None:
		process_count = multiprocessing.cpu_count()
	if process_count <= 0:
		raise ValueError('The value process_count must be a integer larger than zero')
	print('Started running build_training_set, %d process(s) in parallel', process_count)
	#just for locking the indexing value "file_idx"
	idx = multiprocessing.Value('i', 0)
	file_idx = multiprocessing.Value('i', 0)
	process_pool = multiprocessing.Pool(processes = process_count, initializer = global_var_init, initargs = (idx, file_idx,))
	
	files = scan_all_files(path_in)
	if not os.path.exists(path_out):
		os.makedirs(path_out)
	file_len = len(files)
	
	for file in files:
		process_pool.apply_async(on_compute_callback, args = (file, path_out, dest_res, ))
	
	while file_idx.value != file_len:
		sys.stdout.write('Processing file %d/%d\r' % (file_idx.value, file_len))
		sys.stdout.flush()
		time.sleep(1)
	
	process_pool.close()
	process_pool.join()