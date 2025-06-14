import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

Vis_CLSNAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

Vis_CLSNAMES_map_index = {}
for k, index in zip(Vis_CLSNAMES, range(len(Vis_CLSNAMES))):
	Vis_CLSNAMES_map_index[k] = index

CLSNAMES = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
CLSNAMES = ['al_224_light']
CLSNAMES = ['KolektorSDD2', 'steel_pipe', 'casting_billet']

CLSNAMES = ['steel_pipe', 'BSD_cls', 'DAGM2007_Class10', 'neu_rail', 'aluminum', 'steel_rail', 'moderately_thick_plates', 'DAGM2007_Class9', 'wood', 'Marbled', 'Mesh', 'cold_rolled_strip_steel', 'severstal_steel', 
                	'hot_rolled_strip_annealing_picking', 'Perforated', 'DAGM2007_Class7', 'Stratified', 'AITEX', 'Blotchy', 'BTech_02', 'bao_steel', 'Matted', 'KolektorSDD', 'BSData', 'medium_heavy_plate', 
                    'aluminum_strip', 'DAGM2007_Class1', 'DAGM2007_Class6', 'leather', 'aluminum_ingot', 'neu_leather', 'DAGM2007_Class3', 'tianchi_aluminum', 'neu_aluminum', 'wide_thick_plate', 'gc10_steel_plate', 'Woven_127', 
                    'rail_surface', 'neu_tile', 'Magnetic_tile', 'metal_plate', 'DAGM2007_Class4', 'Woven_068', 'grid', 'KolektorSDD2', 'Woven_104', 'road_crack', 'Woven_001', 'DAGM2007_Class8', 'neu_hot_rolled_strip', 
                    'hot_rolled_strip_steel', 'neu_magnetic_tiles', 'Fibrous', 'neu_steel', 'DAGM2007_Class5', 'Woven_125', 'DAGM2007_Class2', 'ssgd_glasses', 'wukuang_medium_plate', 'nan_steel', 'tile']
CLSNAMES = ['casting_billet', 'steel_pipe']
CLSNAMES = ['casting_billet', 'steel_pipe', 'KolektorSDD', 'KolektorSDD2']
CLSNAMES = ['casting_billet', 'steel_pipe']
CLSNAMES_map_index = {}
for k, index in zip(CLSNAMES, range(len(CLSNAMES))):
	CLSNAMES_map_index[k] = index

		
class MetalDatasetBatch(data.Dataset):
	def __init__(self, root, transform, target_transform, aug_rate, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug_rate = aug_rate

		self.data_all = []
		#meta_info = json.load(open('/home/data/liuchuni/projects/fsad_big_model/defect_lvlms/dataset/metal_images_for_pretrain_v3_pretrain_mini.json', 'r'))
		meta_info = json.load(open('/home/data/liuchuni/projects/fsad_big_model/defect_lvlms/dataset/dataset_to_public.json', 'r'))
		#meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		#meta_info = json.load(open(meta_path, 'r'))
		meta_info = meta_info[mode]

		if isinstance(obj_name, list):
			self.cls_names = obj_name
		else:
			self.cls_names = [obj_name]
		if mode == 'train':
			print('k_shot ', k_shot)
			save_dir = os.path.join(save_dir, str(k_shot)+'_shot.txt')

		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				print('cls_name ', cls_name)
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length


	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']

		img = Image.open(img_path)
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			#img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			#img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		# transforms
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path), "cls_id":CLSNAMES_map_index[cls_name]}


class MetalDataset(data.Dataset):
	def __init__(self, root, meta_path, transform, target_transform, mode='test', k_shot=0, save_dir=None, obj_name=None):
		
		self.transform = transform
		self.target_transform = target_transform
		self.dataset_dir = root

		self.data_all = []
		# meta_info = json.load(open('/home/data/liuchuni/projects/fsad_big_model/defect_lvlms/dataset/dataset_to_public.json', 'r'))
		# meta_info = json.load(open('/home/data/liuchuni/projects/fsad_big_model/defect_lvlms/dataset/metal_images_for_pretrain_v3_pretrain_mini.json', 'r'))
		meta_info = json.load(open(meta_path, 'r'))

		meta_info = meta_info[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
		else:
			self.cls_names = CLSNAMES  # list(meta_info.keys())
		
		if mode == 'train':
			save_dir = os.path.join(save_dir, 'k_shot.txt')

		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				print('cls_name ', cls_name)
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']

		img = Image.open(os.path.join(self.dataset_dir, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			if not os.path.exists(os.path.join(self.dataset_dir, mask_path)):
				img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
			else:
				img_mask = np.array(Image.open(os.path.join(self.dataset_dir, mask_path)).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		# transforms
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.dataset_dir, img_path), "cls_id":CLSNAMES_map_index[cls_name]}


class VisaDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		name = self.root.split('/')[-1]
		meta_info = meta_info[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			self.cls_names = list(meta_info.keys())
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path), "cls_id":Vis_CLSNAMES_map_index[cls_name]}



class MVTecDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, aug_rate, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug_rate = aug_rate

		self.data_all = []
		#meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		# meta_info = json.load(open('/data/account/liuchuni/code/fsad_big_model/defect_lvlms/data/metal_own_meta.json', 'r'))
		# meta_info = json.load(open('/data/datasets/pub/public/own_anomaly_detect/metal_meta_ksdd.json', 'r'))
		meta_info = json.load(open('./data/dataset_to_public.json', 'r'))
		name = self.root.split('/')[-1]
		meta_info = meta_info[mode]

		if isinstance(obj_name, list):
			self.cls_names = obj_name
		else:
			self.cls_names = [obj_name]
		if mode == 'train':
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		#else:
		#	self.cls_names = list(meta_info.keys())
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				print('cls_name ', cls_name)
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length


	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']

		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		# transforms
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path), "cls_id":CLSNAMES_map_index[cls_name]}


