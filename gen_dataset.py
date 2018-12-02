import numpy as np
import torch
import torch.utils.data as udata
import h5py

class Dataset(udata.Dataset):
	r"""Implements torch.utils.data.Dataset
	"""
	def __init__(self, faces, labels):
		super(Dataset, self).__init__()
# 		self.data_dir = data_dir
# 		hf = h5py.File(data_dir, 'r')
		self.data = faces
		self.label = labels
		assert self.data.shape[0] == self.label.shape[0], "data size should match label size"
# 		hf.close()
# 	def __init__(self, data_dir = "dataset"):
# 		super(Dataset, self).__init__()
# 		self.data_dir = data_dir
# 		hf = h5py.File(data_dir, 'r')
# 		self.data = np.array(hf.get('faces'))
# 		self.label = np.array(hf.get('labels'))
# 		assert self.data.shape[0] == self.label.shape[0], "data size should match label size"
# 		hf.close()


            
	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		faces = np.transpose(self.data[index],(2,0,1))
		sample = {'faces': np.asarray(faces,np.float32), 'labels': self.label[index]}
		return sample