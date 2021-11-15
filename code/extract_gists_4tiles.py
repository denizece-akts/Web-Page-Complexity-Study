from glob import glob
import sys 
import os
from PIL import Image
import leargist
import numpy as np

'''Author: Dr .Selman Bozkır'''
'''4.5.2019'''
'''Gist Descriptor extractor and csv builder for a folder containing sub class folders
   Example usage: python3 extract_gist.py ./train gisttrain.csv
   If your files are located at train/class_specific_folder/*.* then the script will work ok
'''

def prepare_header(descriptor_length):
	'''This function prepares a suitable header according to the length of the descriptor'''
	character = "f"
	header = ""
	loopcounter = 1
	for loopcounter in range(1,descriptor_length+1):
		header+= character + str(loopcounter) + ","
	header += "gsi"
	return header

argument_count = len(sys.argv)
if(argument_count ==0):
	print("Specify relative root path for the files to be recursivly traversed")
	exit()
else:
	print("Argument is ok! {}".format(sys.argv[1]))
	print("Argument is ok! target csv file is {}".format(sys.argv[2]))
	print("I will extract 1+4 level gist descriptors for the source directory and produce {}".format(sys.argv[2]));
	file_proccessed = 0
	is_header_prepared = False
	data_to_write = open(sys.argv[2], "w")
	result = []
	
	result += [y for x in os.walk(sys.argv[1]) for y in glob(os.path.join(x[0], '*.*'))] #*.* all
	for file in result:
		print(str(file_proccessed+1) + " " + file)
		path_component_list = file.split(os.sep)
		im = Image.open(file)
		descriptors = leargist.color_gist(im)
		im_nmpy = np.array(im) #convert pil image to numpy array
		if(im_nmpy.shape[0]%2 == 0):
			M = int(im_nmpy.shape[0]/2) #width /2
		else:
			M = int((im_nmpy.shape[0]+1)/2) 
		if(im_nmpy.shape[1]%2 == 0):
			N = int(im_nmpy.shape[1]/2) #height /2
		else:
			N = int((im_nmpy.shape[1]+1)/2) #height /2
		tiles = [im_nmpy[x:x+M,y:y+N] for x in range(0,im_nmpy.shape[0],M) for y in range(0,im_nmpy.shape[1],N)] # 4 sub tiles
		
		for i in range(0,len(tiles)):
			desc = leargist.color_gist(Image.fromarray(tiles[i], 'RGB'))			
			descriptors = np.concatenate((descriptors, desc))			
	
		if(is_header_prepared == False):		
			header = prepare_header(len(descriptors))
			is_header_prepared = True
			data_to_write.write(header + "\n")
		VIstring = ','.join(['%.5f' % num for num in descriptors]) 
		#Convert numpy array to csv style record
		#print(VIstring + "," + str(path_component_list[-2]))'''
		data_to_write.write(VIstring + "," + str(path_component_list[-2]) +"\n")  #Get class_specific_folder name
		file_proccessed = file_proccessed + 1
	data_to_write.close()
	print("Operation is done. I have extracted " + str(file_proccessed) + " files.")
