from numpy import genfromtxt as gft
import cv2
import csv
import re
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

from load_ISIC import read_image_list

def retValid():

        image_list, Y_all=read_image_list()
        label=["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]
        TASK3_PRED_FILE = '/home/grads/k/kaihe/Documents/ISIC/models/deepModels/resnet_50/valid_results.csv'
        predLab = gft(TASK3_PRED_FILE, delimiter=' ')
        valid_list=image_list[Y_all.shape[0]-predLab.shape[0]:]
        Y_valid=Y_all[Y_all.shape[0]-predLab.shape[0]:,:]

        wrongPred=open('/home/grads/k/kaihe/Documents/ISIC/models/deepModels/resnet_50/wrongPred.csv','w')
        wrongPred.write("Image,Correct Label,Predicted Label\n")
        for i in range(0,predLab.shape[0]-1):
                if np.argmax(predLab[i,:])!=np.argmax(Y_valid[i,:]):
                        wrongPred.write("%s," % valid_list[i])
                        wrongPred.write("%s," % label[np.argmax(Y_valid[i,:])])
                        wrongPred.write("%s\n" % label[np.argmax(predLab[i,:])])
def hist():
	WRONG_PRED_FILE='/home/grads/k/kaihe/Documents/ISIC/models/deepModels/resnet_50/wrongPred.csv'
	label=["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]
	count=np.empty((len(label),2))
	with open(WRONG_PRED_FILE) as predfile:
		readPred=csv.reader(predfile,delimiter=',')
		next(readPred,None)
		count_gt=[0,0,0,0,0,0,0]
		count_pd=[0,0,0,0,0,0,0]
		for row in readPred:
			count_gt[label.index(row[1])]+=1
			count_pd[label.index(row[2])]+=1
	count[:,0]=count_gt
	count[:,1]=count_pd
	df=pd.DataFrame(count,index=label,columns=['Misclassified','Mispredicted to'])
	plot.figure()
	df.plot.bar()
	plot.show()

	return label, count_gt, count_pd
	

def hist_categ():
	WRONG_PRED_FILE='/home/grads/k/kaihe/Documents/ISIC/models/deepModels/resnet_50/wrongPred.csv'
	label=["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]
	#label.remove(categ)
	with open(WRONG_PRED_FILE) as predfile:
		readPred=csv.reader(predfile,delimiter=',')
		count=np.zeros((len(label),len(label)))
		for row in readPred:
			for i in range(0,len(label)):
				categ=label[i]
				print(categ)
				if row[1]==categ:
					#print(row[2])
					count[label.index(row[2]),i]+=1
	
	df=pd.DataFrame(count,index=label,columns=label)
	#fig,axes=plot.subplots(nrows=1,ncols=7)
	df.plot.bar()
	#df['MEL'].plot.bar(ax=axes[0],color='r');axes[0].set_title('MEL');
	#df['NV'].plot.bar(ax=axes[1]);axes[1].set_title('NV');
	#df['BCC'].plot.bar(ax=axes[2]);axes[2].set_title('BCC');
	#df['AKIEC'].plot.bar(ax=axes[3]);axes[3].set_title('AKIEC');
	
	#df['BKL'].plot.bar(ax=axes[4]);axes[4].set_title('BKL');
        #df['DF'].plot.bar(ax=axes[5]);axes[5].set_title('DF');
        #df['VASC'].plot.bar(ax=axes[6]);axes[6].set_title('VASC');
     

	plot.show()
	print(label)		
	print(count)
		





