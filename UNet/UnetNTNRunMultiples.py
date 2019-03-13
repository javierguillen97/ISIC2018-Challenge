from os.path import expanduser
import shutil
import NTN
import unet
import numpy as np 
import pdb
import tensorflow 
import TensorflowUtils

DataLoadDir = expanduser("~") + '/Documents/Projects/Datasets/ISIC2018'

Channel = 3
NClass = 3
FilterSize = 3
NumOfFeature = 16
NumOfLayers = 3
ImageSize = [256, 256]

SizeLimitation = 15
MaxEpoch1 = int(40)
MaxEpoch2 = int(25)
BatchSize = 3
Keep_Prob = 0.7
RegWeight1 = 0.0001 # 0.0001 
RegWeight2 = 0.0001 # 0.0001 / 0.00001
LearningRate = 1e-3

ReadAllNoisyImage = np.load(DataLoadDir + 'AllInput.npy')


ReadAllCleanImage = np.load(DataLoadDir + 'AllInput.npy')
ReadAllCleanLabel = np.load(DataLoadDir + 'AllGroundTruth.npy')

ReadNoisyImage = np.zeros((350, 256, 256, 3))
ReadNoisyLabel = np.zeros((350, 256, 256, 3))

ReadValImage = np.zeros((350, 256, 256, 3))
ReadValLabel = np.zeros((350, 256, 256, 3))

ReadCleanImage = np.zeros((150, 256, 256, 3))
ReadCleanLabel = np.zeros((150, 256, 256, 3))


NoisyStart = 0
CleanStart = 35



out = open('/home/asus/Desktop/DATA', 'a')

for j in range(1):

	# ReadAllNoisyLabel = np.load(DataLoadDir + 'NoisyGroundTruthFlip' + str(j / 10.) + '.npy')

	for i in range(10):
		ReadNoisyImage[i * 35 : i * 35 + 35] = ReadAllNoisyImage[NoisyStart + i * 50 : NoisyStart + i * 50 + 35]
		ReadNoisyLabel[i * 35 : i * 35 + 35] = ReadAllCleanLabel[NoisyStart + i * 50 : NoisyStart + i * 50 + 35]

		ReadValImage[i * 35 : i * 35 + 35] = ReadAllNoisyImage[NoisyStart + i * 50 : NoisyStart + i * 50 + 35]
		ReadValLabel[i * 35 : i * 35 + 35] = ReadAllCleanLabel[NoisyStart + i * 50 : NoisyStart + i * 50 + 35]

		ReadCleanImage[i * 15 : i * 15 + 15] = ReadAllNoisyImage[CleanStart + i * 50 : CleanStart + i * 50 + 15]
		ReadCleanLabel[i * 15 : i * 15 + 15] = ReadAllCleanLabel[CleanStart + i * 50 : CleanStart + i * 50 + 15]	

	for u in range(0, 1):
		# UnetFileName = 'unet-noisydata' + str(j) + '-' + 'run' + str(u) + '/'
		# NTNFileName =  'NTN-noisydata' + str(j) + '-' + 'run' + str(u)  + '/'

		UnetFileName = 'unet-noisydata-clean/' 
		# NTNFileName =  'NTN-noisydata-thres/' 	

		UnetDir = expanduser("~") + '/Desktop/Noise_tolerant_deep_learning/unet/Experiment on Histo data/MultipleRun/Model1/' + UnetFileName
		# NTNDir = expanduser("~") + '/Desktop/Noise_tolerant_deep_learning/NTN/Experiment on Histo data/MultipleRun/Model1/' + NTNFileName

		UnetCleanAcc, UnetNoiseAcc = unet.main(Channel, NClass, FilterSize, NumOfFeature, NumOfLayers, ReadNoisyImage, ReadNoisyLabel,
									         ReadValImage, ReadValLabel, MaxEpoch1, BatchSize, Keep_Prob, RegWeight1, LearningRate, False, 
									         SizeLimitation, UnetDir)

		# NTNCleanAcc, NTNNoiseAcc = NTN.main(Channel, NClass, FilterSize, NumOfFeature, NumOfLayers, ReadNoisyImage, ReadNoisyLabel, 
		# 									 ReadValImage, ReadValLabel, MaxEpoch2, BatchSize, Keep_Prob, RegWeight2, LearningRate, True, 
		# 									 SizeLimitation, NTNDir, UnetDir)
		
		# pdb.set_trace()
		ErrorUnet = unet.PredictAll(Channel, NClass, FilterSize, NumOfFeature, NumOfLayers, Keep_Prob, ReadCleanImage, ReadCleanLabel, UnetDir + 'weights.npy') 
		# ErrorNTN = unet.PredictAll(Channel, NClass, FilterSize, NumOfFeature, NumOfLayers, Keep_Prob, ReadCleanImage, ReadCleanLabel, NTNDir + 'UnetWeights.npy') 
		print('%d trial completed'%i)
		# pdb.set_trace()
		# out.write('Unet, Flipping prob: %s, Trial: %d, CleanError: %s, NoisyError: %s\n'%((j + 1)/10., i + 1, 1 - UnetCleanAcc, 1 - UnetNoiseAcc))
		# out.write('NTN, Flipping prob: %s, Trial: %d, CleanError: %s, NoisyError: %s\n\n'%((j + 1)/10., i + 1, 1 - NTNCleanAcc, 1 - NTNNoiseAcc))

		out.write('Unet, Flipping prob: %s, Trial: %d, CleanError: %s, NoisyError: %s\n'%(j / 10., u + 1, ErrorUnet, 1 - UnetNoiseAcc))
		# out.write('NTN, Flipping prob: %s, Trial: %d, CleanError: %s, NoisyError: %s\n\n'%(j / 10., u + 1, ErrorNTN, 1 - NTNNoiseAcc))
		# shutil.rmtree(UnetDir)
		# shutil.rmtree(NTNDir)
# pdb.set_trace()
# out.close()

############################################################
# For histo true-noise data
############################################################
# DataLoadDir = expanduser("~") + '/Desktop/DATASETS/Histo/Manual Segmentation from Dr. Chen/Augmentation/'

# Channel = 3
# NClass = 3
# FilterSize = 3
# NumOfFeature = 16
# NumOfLayers = 3
# ImageSize = [256, 256]

# SizeLimitation = 15
# MaxEpoch1 = int(40)
# MaxEpoch2 = int(10)
# BatchSize = 3
# Keep_Prob = 0.7
# RegWeight1 = 0.0001 # 0.0001 
# RegWeight2 = 0.0001 # 0.0001 / 0.00001
# LearningRate = 1e-3

# ReadAllNoisyImage = np.load(DataLoadDir + 'AllInput.npy')
# # ReadAllNoisyImage = TensorflowUtils.NormalizaImage(ReadAllNoisyImage)
# # ReadNoisyLabel = np.load(DataLoadDir + 'small_permution_y_' + str(j) + '.npy')[0 : 35]
# ReadAllkMeansLabel = np.load(DataLoadDir + 'AllKMeans.npy')
# ReadAllOtsuLabel = np.load(DataLoadDir + 'AllOtsu.npy')



# ReadAllCleanImage = np.load(DataLoadDir + 'AllInput.npy')
# # ReadAllCleanImage = TensorflowUtils.NormalizaImage(ReadAllCleanImage)
# ReadAllCleanLabel = np.load(DataLoadDir + 'AllGroundTruth.npy')

# ReadNoisyImage = np.zeros((350, 256, 256, 3))
# ReadNoisyLabel = np.zeros((350, 256, 256, 3))


# ReadValImage = np.zeros((350, 256, 256, 3))
# ReadValLabel = np.zeros((350, 256, 256, 3))

# ReadCleanImage = np.zeros((150, 256, 256, 3))
# ReadCleanLabel = np.zeros((150, 256, 256, 3))


# NoisyStart = 0
# CleanStart = 35



# # out = open('/home/asus/Desktop/DATA', 'w')
# # pdb.set_trace()
# for j in range(0, 2):

# 	if j == 1:
# 		ReadAllNoisyLabel = ReadAllkMeansLabel
# 		# ReadAllNoisyLabel = ReadAllCleanLabel
# 		UnetFileName = 'unet-noisyKMeansdata/' 
# 		NTNFileName =  'NTN-noisyKMeansdata/'
# 		Type = 'KMeans'
# 	else:
# 		ReadAllNoisyLabel = ReadAllOtsuLabel
# 		UnetFileName = 'unet-noisyOtsudata/' 
# 		NTNFileName =  'NTN-noisyOtsudata/'	
# 		Type = 'Otsu'

# 	for i in range(10):
# 		ReadNoisyImage[i * 35 : i * 35 + 35] = ReadAllNoisyImage[NoisyStart + i * 50 : NoisyStart + i * 50 + 35]
# 		ReadNoisyLabel[i * 35 : i * 35 + 35] = ReadAllNoisyLabel[NoisyStart + i * 50 : NoisyStart + i * 50 + 35]

# 		ReadValImage[i * 35 : i * 35 + 35] = ReadAllNoisyImage[NoisyStart + i * 50 : NoisyStart + i * 50 + 35]
# 		ReadValLabel[i * 35 : i * 35 + 35] = ReadAllCleanLabel[NoisyStart + i * 50 : NoisyStart + i * 50 + 35]

# 		ReadCleanImage[i * 15 : i * 15 + 15] = ReadAllNoisyImage[CleanStart + i * 50 : CleanStart + i * 50 + 15]
# 		ReadCleanLabel[i * 15 : i * 15 + 15] = ReadAllCleanLabel[CleanStart + i * 50 : CleanStart + i * 50 + 15]	

# 	if j == 0:
# 		Iter = 5
# 	else:
# 		Iter = 40

# 	for u in range(0, 1):
# 		# UnetFileName = 'unet-noisydata' + str(j) + '-' + 'run' + str(i) + '/'
# 		# NTNFileName =  'NTN-noisydata' + str(j) + '-' + 'run' + str(i)  + '/'

 	

# 		UnetDir = expanduser("~") + '/Desktop/Noise_tolerant_deep_learning/unet/Experiment on Histo data/MultipleRun/Model2/' + UnetFileName
# 		NTNDir = expanduser("~") + '/Desktop/Noise_tolerant_deep_learning/NTN/Experiment on Histo data/MultipleRun/Model2/' + NTNFileName

# 		UnetCleanAcc, UnetNoiseAcc = unet.main(Channel, NClass, FilterSize, NumOfFeature, NumOfLayers, ReadNoisyImage, ReadNoisyLabel,
# 									         ReadValImage, ReadValLabel, MaxEpoch1, BatchSize, Keep_Prob, RegWeight1, LearningRate, False, 
# 									         SizeLimitation, UnetDir)

# 		NTNCleanAcc, NTNNoiseAcc = NTN.main(Channel, NClass, FilterSize, NumOfFeature, NumOfLayers, ReadNoisyImage, ReadNoisyLabel, 
# 											 ReadValImage, ReadValLabel, MaxEpoch2, BatchSize, Keep_Prob, RegWeight2, LearningRate, True, 
# 											 SizeLimitation, NTNDir, UnetDir)
		
# 		# pdb.set_trace()
# 		ErrorUnet = unet.PredictAll(Channel, NClass, FilterSize, NumOfFeature, NumOfLayers, Keep_Prob, ReadCleanImage, ReadCleanLabel, UnetDir + 'weights.npy') 
# 		ErrorNTN = unet.PredictAll(Channel, NClass, FilterSize, NumOfFeature, NumOfLayers, Keep_Prob, ReadCleanImage, ReadCleanLabel, NTNDir + 'UnetWeights.npy') 
# 		print('%d trial completed'%u)
# 		# pdb.set_trace()
# 		# out.write('Unet, Flipping prob: %s, Trial: %d, CleanError: %s, NoisyError: %s\n'%((j + 1)/10., i + 1, 1 - UnetCleanAcc, 1 - UnetNoiseAcc))
# 		# out.write('NTN, Flipping prob: %s, Trial: %d, CleanError: %s, NoisyError: %s\n\n'%((j + 1)/10., i + 1, 1 - NTNCleanAcc, 1 - NTNNoiseAcc))

# 		out.write('Unet, Flipping Type: ' + Type + ', Trial: %d, CleanError: %s, NoisyError: %s\n'%(u + 1, ErrorUnet, 1 - UnetNoiseAcc))
# 		out.write('NTN, Flipping Type: ' + Type + ', Trial: %d, CleanError: %s, NoisyError: %s\n\n'%(u + 1, ErrorNTN, 1 - NTNNoiseAcc))
# 		# shutil.rmtree(UnetDir)
# 		# shutil.rmtree(NTNDir)
# # pdb.set_trace()
# out.close()
