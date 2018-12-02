import numpy as np
import matplotlib.pyplot as plt
import rawpy
import face_recognition
import cv2
import glob
import os
import time
import h5py

def face_rescale(filepath,face_id):
    file_list = glob.glob(os.path.join(os.path.join(filepath,face_id),'*.nef'))
    faces = []
    for i in range(len(file_list)):
        img = rawpy.imread(file_list[i]).postprocess()
        new_shape_x = int(img.shape[1]/10)
        new_shape_y = int(img.shape[0]/10)
        new_img = cv2.resize(img,(new_shape_x,new_shape_y))
        faces.append(new_img)
    return faces

def filter_bad_face(faces):
    good_faces=[]
    for i in range(len(faces)):
        face = faces[i]
        face_location = face_recognition.face_locations(face)
        if face_location:
            face_location = face_location[0]
            good_faces.append(face)
    return np.array(good_faces)

def save_face_to_h5(faces,filepath):
    hf = h5py.File(os.path.join(filepath,'data.h5'), 'w')
    hf.create_dataset('faces', data=faces)
    hf.close()
    
def read_face_from_h5(filepath):
    hf = h5py.File(os.path.join(filepath,'data.h5'), 'r')
    faces = np.array(hf.get('faces'))
    hf.close()
    return faces

def file_exist(savepath):
    if os.path.isfile(os.path.join(savepath,'data.h5')):
        return True
    else:
        return False

def load_faces(filepath,savefolder,reverse=False):
    face_ids = os.listdir(filepath)
    face_imgs = []
    tmp = time.time()
    if reverse:
        i = len(face_ids)-1
        while i > 0:
            face_id = face_ids[i]
            savepath = os.path.join(savefolder,face_id)
            if file_exist(savepath):
                face_data = read_face_from_h5(savepath)
                print("load faces from pre-processed data")
            else:
                faces = face_rescale(filepath,face_id)
                face_data = filter_bad_face(faces)
                if not os.path.exists(savepath):
                    os.mkdir(savepath)
                save_face_to_h5(face_data,savepath)
                print("load faces from raw images and save into h5py file")
            print("loading %d/%d has been completed, time elapsed is %.2f,%d face images"%(i+1,len(face_ids),time.time()-tmp,len(face_data)))
            face_imgs.append(face_data)
            i -= 1
    else:
        for i in range(len(face_ids)):
            face_id = face_ids[i]
            savepath = os.path.join(savefolder,face_id)
            if file_exist(savepath):
                face_data = read_face_from_h5(savepath)
                print("load faces from pre-processed data")
            else:
                faces = face_rescale(filepath,face_id)
                face_data = filter_bad_face(faces)
                if not os.path.exists(savepath):
                    os.mkdir(savepath)
                save_face_to_h5(face_data,savepath)
                print("load faces from raw images and save into h5py file")
            print("loading %d/%d has been completed, time elapsed is %.2f,%d face images"%(i+1,len(face_ids),time.time()-tmp,len(face_data)))
            face_imgs.append(face_data)
    return face_imgs,face_ids


def crop_face_2(faces,ids,crop_size=(128,128)):
    assert len(faces) == len(ids)
    cropped_faces_train = []
    cropped_faces_val = []
    labels_train = []
    labels_val = []
    for i in range(len(faces)):
        for j in range(faces[i].shape[0]):
            face = faces[i][j]
#             print(face.shape)
            face_location = face_recognition.face_locations(face)[0]
#             print(face_location)
            cropped_face = cv2.resize(face[face_location[0]:face_location[2],face_location[3]:face_location[1]],crop_size)
            if np.random.rand() <0.2:
                cropped_faces_val.append(cropped_face)
                labels_val.append(i)
            else:
                cropped_faces_train.append(cropped_face)
                labels_train.append(i)
        print("%d/%d is completed" % (i,len(faces)))
    return np.array(cropped_faces_train),np.array(labels_train),np.array(cropped_faces_val),np.array(labels_val)

if __name__ == '__main__':
	filepath = "/Users/weixinjiang/Documents/PhDinUS/courses/2018 Fall/eecs 495 biometrics/hw3/dataset/face_data/"
	savefolder = "./dataset"

	faces,ids = load_faces(filepath,savefolder)
	cropped_faces_train, labels_train,cropped_faces_val, labels_val = crop_face_2(faces,ids,crop_size=(128,128))

	hf = h5py.File('./dataset.h5', 'w')
	hf.create_dataset('faces_train', data=cropped_faces_train)
	hf.create_dataset('labels_train', data=labels_train)
	hf.create_dataset('faces_val', data=cropped_faces_val)
	hf.create_dataset('labels_val', data=labels_val)
	hf.close()

