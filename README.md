# Image Segmentation in Autonomous Driving

Python code for running mask-rcnn and YOLOV5 for instance segmentation of cars using bdd (berkeley deepdrive) datasets.
 
 ## Mask-rcnn:  Training and Segmentation
 
 Simply, we can use COCO.h5 (pre-trained model) for car segmentation.
 But we are going to train the segmentation model (pre-weights are based on coco.h5) using bdd datasets, and expect a better segmentation based on our car.h5 model.
 
 ### Dependency
 1. python ----  3.7.7
 /
 2. install following dependencies (refer to requirements) as my versions
 2.1 numpy --- 1.20.1
 2.2 scipy --- 1.6.1
 2.3 Pillow --- 8.1.2
 2.4 cython --- 0.29.22
 2.5 matplotlib ---3.3.4
 2.6 scikit-image --- 0.18.1
 2.7 tensorflow --- 2.4.1
 2.8 keras  ---2.4.3
 2.9 h5py ---3.2.1
 2.10 imgaug --- 0.4.0
 2.11 IPython[all]
 2.12 opencv-python
 /
 3. Any conflict, try to upgrade or downgrade versions for best performance
 
 ### Running in GPU
 the running enviroment for GPU is using CHTCondor at UW-Madison
 Run:    
    
    condor_submit run.sub
    
 Check status:
    
    condor_q -all
    
 remove jobs: 
     
    condor_rm --jobid
 /
 
 The code and files for running in CHTC are attached as chtc.tar.gz
 
 ## Warm-up
    move your interested pictures into folder "test_images_objects"
 
 Run:
    
    python3 seg_objects_by_COCO.py
    
 Then you will find segmented pictures with the same file name.

<img src="/display/obj_1_orig.jpg" alt="obj1"/>
<img src="/display/obj_1_coco.jpg" alt="obj1_coco"/>
<img src="/display/obj_2_orig.jpg" alt="obj2"/>
<img src="/display/obj_2_coco.jpg" alt="obj2_coco"/>
<img src="/display/obj_3_orig.jpg" alt="obj3"/>
<img src="/display/obj_3_coco.jpg" alt="obj3_coco"/>
<img src="/display/obj_4_orig.jpg" alt="obj4"/>
<img src="/display/obj_4_coco.jpg" alt="obj4_coco"/>

### Training

The reason we want to train a dataset is some objects may not exist in coco classes.
Here are two examples to segment using COCO.h5 and our trained Balloon.h5.
orig
\
<img src="/display/10_orig.jpg" alt="b10_orig"/>
\
\
coco
<img src="/display/10_coco.jpg" alt="b10_coco"/>
\
\
balloon
<img src="/display/10_balloon.jpg" alt="b10_ball"/>
 
 

 ### Trainning Car Model Based On COCO pre-set Model or Based on BDD datasets-trained Model
 
 Run 
 
    python3 car.py --dataset=datasets/car --weights=coco train
 
 (datasets/car contains two folders, train and val, and each with pictures and .json file with their annotations)
 
 The .json file should be in VIA format
 
 
  #### Original
  <img src="/display/00d79c0a-23bea078.jpg" alt="00d79c0a-23bea078.jpg"/>
  
   Run with coco weights
 
      python3 seg_car_by_COCO.py 
    
  #### COCO model
 <img src="/display/00d79c0a-23bea078_coco.jpg" alt="00d79c0a-23bea078_coco.jpg"/>
 
  
   
   Run with our trained weights
 
     python3 seg_car_by_Car.py 
 
  #### Car model
 <img src="/display/00d79c0a-23bea078_car.jpg" alt="00d79c0a-23bea078_car.jpg"/>
 
 
  #### original picture
 <img src="/display/aca32929-00000000.jpg" alt="aca32929-00000000.jpg"/>
  #### COCO model
 <img src="/display/aca32929-00000000_coco.jpg" alt="aca32929-00000000_coco.jpg"/>
  #### Car model
 <img src="/display/aca32929-00000000_car.jpg" alt="aca32929-00000000_car.jpg"/>
 
  ## YOLOV5:  Training and Segmentation
  
### Dependency
numpy==1.17
scipy==1.4.1
cudatoolkit==10.2.89
opencv-python
torch==1.5
torchvision==0.6.0
matplotlib
pycocotools
tqdm
pillow
tensorboard
pyyaml

### Setup the YAML files for training
#### data.yaml (custom your own class yaml)
train: ./train/images 
val: ./valid/images  
nc: 1 
names: ['car']
#### custom_yolov5s.yaml (configure yaml)
# parameters
nc: 1  # number of classes  # CHANGED HERE
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]


#### train my custom car model

python train.py --img 416 --batch 80 --epochs 1000 --data './data.yaml' --cfg ./models/custom_yolov5s.yaml --weights './yolov5s.pt'

