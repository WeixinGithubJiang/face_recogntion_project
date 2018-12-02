import numpy as np
import cv2
import sys

scales = [1,1.05,1.1,1.15,1.2,1.25,1.3,1.35]

def RandomResize(img):
	scale = scales[np.random.randint(0,len(scales))]
	new_h = int(img.shape[1]*scale)
	new_w = int(img.shape[0]*scale)
	new_img = cv2.resize(img,(new_h,new_w))
	return new_img

def RandomCrop(img,size = (128,128)):
	img = RandomResize(img)
	assert img.shape[0] >= size[0]
	assert img.shape[1] >= size[1]
	if img.shape == size:
		return img
	else:
		x = np.random.randint(0,img.shape[0]-size[0]+1)
		y = np.random.randint(0,img.shape[1]-size[1]+1)
# 		print(x,y)
		return img[y:y+size[1],x:x+size[0]]
    
    
def Preprocess(data,size=(128,128)):
    new_data = []
    for i in range(data.shape[0]):
        img = data[i]
        new_data.append(RandomCrop(img,size))
    return np.array(new_data)


# progress bar
class ProgressBar():

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.current_step = 0
        self.progress_width = 50

    def update(self, step=None,loss = 0):
        self.current_step = step

        num_pass = int(self.current_step * self.progress_width / self.max_steps) + 1
        num_rest = self.progress_width - num_pass 
        percent = (self.current_step+1) * 100.0 / self.max_steps 
        progress_bar = '[' + '■' * (num_pass-1) + '▶' + '-' * num_rest + ']'
        progress_bar += '%.2f' % percent + '%' +'  loss = %.6f' % loss
        if self.current_step < self.max_steps - 1:
            progress_bar += '\r' 
        else:
            progress_bar += '\n' 
        sys.stdout.write(progress_bar) 
        sys.stdout.flush()
        if self.current_step >= self.max_steps:
            self.current_step = 0
            print