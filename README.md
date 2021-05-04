# Segmentation And Car Detection in Autonomous Driving
#### Zichen Qiu (zqiu3@wisc.edu), Jiun-Ting Chen (jchen857@wisc.edu), Zhonggang Li (zli769@wisc.edu), Yimiao Cao (cao223@wisc.edu)
#

### Presentation: http://www.zichenqiu.com/uploads/1/3/6/7/13670916/cs_766_final_project__1_.pptx.pdf
### Github Repo: https://github.com/lizhonggang1984/cs766
### Dockerhub Repo: https://hub.docker.com/repository/docker/lizhonggang1984/mask:v2, https://hub.docker.com/repository/docker/lizhonggang1984/yolov5:v1

# 1. Introduction
## 1.1 Autonomous Driving 
Self-driving car, also known as an autonomous vehicle, is a vehicle that senses its surrounding environment and is able to move safely with minor human input and manipulation (Bagloee, Tavana et al. 2016). Autonomous driving is one of the key feature of future cars (Figure 1). Over the past decades, with the advance of new technologies in communication and robotics, there is an improvement of autonomous driving technique and an explode of its commercial application. The key technique to aid autonomous driving is the modern driving assistance systems, such as Radar, Lidar, GPS and of course, computer vision techniques (Zhao, Liang et al. 2018).
<p align="center">
<img src="/display/figure1.jpg" alt="figure1" align="center"/>
</p>
Figure1 Autonomous driving as the key feature of future cars. No/minor human manipulation and real-time surroundings detection are major components of future cars.

## 1.2 Image Segmentation in Computer Vision
Image segmentation is an important subject of computer vision. It is the process of dividing an image into different regions based on the characteristics of pixels to identify objects in a more efficient way (Opara and Worgotter 1997) (Sarkate, Kalyankar et al. 2013). There are two major types of image segmentation — semantic segmentation and instance segmentation. In semantic segmentation, all objects of the same type are marked using one class label while in instance segmentation similar objects get their own separate labels (Dudkin, Mironov et al. 1995). In Figure 2, we showed the major difference of semantic segmentation and instance segmentation. And in this study, we will mainly focused on vehicle instance segmentation, which is applicable in practice in autonomous driving context.

<img src="/display/figure2.png" alt="figure2"/>

**Figure2 Two types of segmentation in computer vision.a:Semantic segmentation is focusing on to the full image and label each pixel of an image with a corresponding class. b:Instance segmentation is used in more specific manner, which distinguishes separate objects labeled as the same identified class.**


The basic architecture in image segmentation consists of an encoder and a decoder. The encoder extracts features from the image through filters. and decoder is responsible for generating the final output which is usually a segmentation mask containing the outline of the object (Figure 3). 

<img src="/display/figure3.png" alt="figure3" width = "1000">

**Figure 3 Basic architecture of image segmentation. An architecture consists of encoder layer and decoder layer(Badrinarayanan, Kendall et al. 2017)**.

There are various segmentation methods proposed in recent years to solve image segmentation problems in computer vision. These algorithms include threshold method (Tobias and Seara 2002), edge-based method (Ognard, Mesrar et al. 2019), region-based method (Fang, Liu et al. 2021) clustering based method (Arifin and Asano 2006) and artificial neural network-based method (Yang and Yu 2021) (Cervantes-Sanchez, Cruz-Aceves et al. 2019) (Perez-Perez, Golparvar-Fard et al. 2019) (Ozan and Iheme 2019). 

As shown in figure 3, a general CNN model consist of some convolutional and pooling layers followed by fully connected layer.  This leads to slow computation and limits its practical usage. As shown in table 1, we listed some of the improved segmentation models in computer vision, including Mask-RCNN (He, Gkioxari et al. 2017), YOLO (Redmon, Divvala et al. 2016), SOLO (Wang, Choi et al. 2020), RCNN (Zhao, Li et al. 2016) (Le, Zheng et al. 2016) and ResNet (Jung, Choi et al. 2017). Their performance varies due to wide range of layer numbers and architecture, which can be applied in different scenario to satisfy custom’s demand. 
In this study, we will focus two performance evaluation markers: running-time and mAP. Running-time is marker to evaluate the computation efficiency and computation costs to obtain trained model and finish detection tasks. AP (Average Precision) and mAP (Mean Average Precision) are the most popular metrics used to evaluate the model accuracy. As shown in table 1, the parameters of different segmentation models vary from 7 million to 25 million and apply different library, which contributes to the great difference of running performance. The mAP for COCO dataset also varies from 37.8 to 58.5, and the accuracy of detection may be major factor needs to be considered in our study.
<img src="/display/table1.png" alt="table1"/>
**Table1 Comparison of different object detection models.**.

## 1.3 Application of Image Segmentation in Autonomous Driving
Due to its advantages in object tracking, there are many efforts of applying image segmentation in autonomous driving system. In the meanwhile, there are several challenges for the application of image segmentation in autonomous driving. One is the moving character of objects on the road, which makes the detection work hard to track and needs to be done within seconds (Bai, Luo et al. 2016). There are many methods that can be used for moving objects or video segmentation task, such as YOLO (Redmon, Divvala et al. 2016) and YOLACT (Bolya, Zhou et al. 2019). The real-time video segmentation requires a faster computation speed and acceptable accuracy. Therefore speeding up is the main concern in video segmentation in real-time. This is a new subject in computer vision and will be boosted with high-performance GPU devices in future (Martinez, Schiopu et al. 2021) (Yu, Ma et al. 2020).

Another challenging problem in autonomous driving segmentation is the multi object detection.  In order to achieve robust and accurate scene understanding, autonomous vehicles are usually equipped with different sensors (e.g. cameras, LiDARs, Radars), and multiple sensing modalities can be fused to exploit their complementary properties. Many methods have been proposed for deep multi-modal perception problems and is another rising topic in autonomous driving field (Geng, Dong et al. 2020) (Nobis, Geisslinger et al. 2019) (Figure 4). Open datasets and background information pre-loading might be a good way for object detection and semantic segmentation in autonomous driving. 

<img src="/display/figure4.png" alt="figure4"/>

**Figure 4 Segmentation Technology used in autonomous driving**.

# 2. Methods
## 2.1 Dataset
There are multiple datasets available for autonomous driving algorithm development, such as Landmarks (Google open-sourced dataset), Level 5, (Lyft open-sourced the Level 5 dataset) and Oxford Radar Robot-Car Dataset (Oxford radar detection dataset). The Berkeley deep-driving dataset (BDD100K) is a dataset in vehicle detection algorithm development. This dataset collected by UC Berkeley consists of over 100K video sequences with diverse kinds of annotations including image-level tagging, object bounding boxes, drivable areas, lane markings, and full-frame instance segmentation. The dataset possesses geographic, environmental, and weather diversity. Thus we choose BDD100K (https://bdd-data.berkeley.edu/portal.html) (Figure 5). 
<img src="/display/figure5.jpg" alt="figure5"/>

**Figure 5 Berkeley DeepDrive (BDD) Dataset with over 100k images on the road**.

## 2.2 Data Type and Pre-processing
Our dataset and model includes two kinds of annotation and prediction: mask and bounding box. For the mask prediction task, we choose Multi-object tracking (MOTS) 2020 images and labels from BDD as our training, validation and testing data source. Since the images are tracking images, we select the first picture from each scene. In total we obtained 157 images as training dataset, 39 images as validation dataset and 10 images as testing dataset (Table 2). Regarding the bounding box task, we use another 10,000 dataset in BDD and split the dataset into 8000 training data, 1000 validation data, and 1000 testing data, where the ratio of day/night in the testing data is close to 1. 

<img src="/display/table2.png" alt="table2" width="1000"/>
Table2 Images extracted and used in our study.

<br/> 
<br/> 
For mask tasks, we also split data into day-time images and night-time images  (shown in Figure 6). For compatibility with YOLO formatting, we used Roboflow to convert our annotation to box regions. 

<img src="/display/figure6.png" alt="figure6"/>

**Figure 6: Representative images selected from BDD tracking dataset into our study. a,b: Images with annotation as day time c,d: Images with annotation as night time**

## 2.3 Hardware and Workflow
In our study, we submitted our jobs to Euler cluster (WACC Computational Infrastructure) and CHTC's high-throughput computing servers and ran with a single GPU. We further set and fixed our configurations by anaconda (conda create) and docker image   

To accelerate our training, we  used pre-trained weighting from MS COCO to start our training of car image. Then, we inputted our processed data to different models and tried to generate good performance ones by adjusting hyperparameters. Table 3 lists the parameters we changed during the training process. 
<img src="/display/table3.png" alt="table3"/>
**Table 3 Parameters we adjusted in training process.**
#

In the end, we validated and tested our results by the generated model files (*.ptx, *.h5). Figure 7 shows the high-level workflow that we did for this study.
<img src="/display/figure7.png" alt="figure7"/>
**Figure 7:  Working flow of our training and segmentation to use Mask-RCNN in BDD datasets. We obtained images from BDD dataset and generated Docker images. Then we submit our job to the CHTC/Euler for model training. Finally we tested our model for car segmentation.**


## 2.4 Model Implementation

### 2.4.1 faster-RCNN
Faster R-CNN is a framework which achieves real-time object detection with ROI and region proposal networks.The main components of Faster R-CNN includes Convolution layers, Region Proposal Networks,  Region of Interest (ROI) Pooling, and Classifications layers.

In the beginning, Faster RCNN used a conv+relu+pooling structure to extract the feature maps from the image. Then, Region Proposal Networks generate region proposals and prediction object locations based on different sizes of anchors (bounding box), where the anchors were adjusted by bounding box regression to derive accurate proposals. After the feature maps and proposals were generated by previous steps. Roi Pooling layer integrates this information and output information to a fully connected layer to classify the object. 

In our study, we referred to a model built in the mmdetction codebase, which is implemented in Pytorch, and adjusted hyperparameters to train our dataset. 

### 2.4.2 Mask-RCNN
As shown in Figure 8, Mask R-CNN is a fully convolutional network added to the top of the CNN features of a faster R-CNN to generate a mask segmentation output. Compared to Fast-RCNN, Mask R-CNN has an additional branch for predicting segmentation masks on each ROI in a pixel to pixel manner. ROI Align is used to align the ROI pool more precisely. After the generation of masks, ROI align combines them with classification and bounding boxes from faster R-CNN to generate the precise segmentation (Ahmed, Gulliver et al. 2020). The current version of Mask-RCNN is using Tensorflow and Keras for its library and best use GPU for its computation due to the large parameter numbers, 
<p align="center">
<img src="/display/figure9.png" alt="figure9" width = "800" />
</p>

**Figure 8 The Mask R-CNN framework for instance segmentation.**


In our study, we forked a Mask-RCNN code from a public Github repo  and made changes to specify CAR as our segmentation class. The annotation files of Seg-Track images are converted to specified format for training and validation. 

### 2.4.3 YOLO
Compared to Faster-RCNN and Mask-RCNN, you only look once (YOLO) is a lightweight, easy implemented, training quickly and inference quickly model. Due to its speed in real-time detection, it is considered the first choice for real-time object detection among many computer vision and machine learning experts and this is simply because of it’s the state-of-the-art real-time object detection algorithm in terms of performance (FPS), ease of use (setting up and configurations) and versatility (models can be converted to serve different devices, such as iOS or Android) (Figure 9). The naive version of YOLO models are basically composed of convolution layers, while the structure becomes more complex to achieve data augmentation and computation efficiency. 
<p align="center">
<img src="/display/figure12.jpg" alt="figure12" width="600"/>
</p>

**Figure9: The YOLOV5 for realtime detection. YOLO is considered as the first choice in real time segmentation task and widely used in iPhone and Android applications.**

We trained our data with v4 and v5 in this study. For v4, we also referred to the model and the pre-trained weighting  that was provided in the mmdection codebase. For v5, we followed the following public Github repo. 

We made changes to the data.yaml file and the configuration file in model to better accommodate car class segmentation and detection. Similar to our implementation method in Mask-RCNN, we generate a docker image with pytorch library and train our model in CHTC. The pre-trained weights from COCO named yolov5x.pt  is used to accelerate our training. After training, the model file (named as best.pt) will be saved for our following evaluation and segmentation task. 


# 2.5 Evaluation Metric
We split images by day/night time and trained them with different models. We then evaluated the model by mean Average Precision (mAP) and compared the result in different scenarios. In statistics, precision measures how accurate your predictions are, that is, the percentage of your predictions are correct. The definition of precision are shown as follows:
<br/>
<img src="/display/formula.png" alt="formula"/>
<br/>
The mean Average Precision is calculated by taking the mAP over all classes and/or over all Intersection over Union (IoU) thresholds. IoU is a common metric when it comes to object localization tasks, which evaluates the performance of the predicted bounding box. Figure 10 illustrates the idea.  
<p align="center">
<img src="/display/figure13.png" alt="figure13" height="300" width="400"/>
</p>
Figure 10: Definition of Intersection over Union to calculate mAP.

# 3. Results 
## 3.1 Mask-RCNN (for Car Image Segmentation)
### 3.1.1 COCO pre-trained weighting: poor performance 
We started our training using pre-weight based on MS COCO training. COCO is a large-scale object detection, segmentation and captioning dataset with over 330K images. It uses 80 classes and generates 80 categories ready for detection. So we first used the pre-weight for detection of our images (Figure 11), and found some missing labels of cars in our car datasets. The mAP based on the validation dataset for COCO pre-weight for car segmentation is only 0.4024. Thus we need to train our own car datasets to achieve better performance for our Mask-RCNN model.

<img src="/display/figure14.png" alt="figure14"/>
Figure 11 Applying COCO pre-weight for car segmentation task in Mask-RCNN. The mAP calculated based on validation dataset for pre weight is only 0.4024, which can’t predict the car on the road. We can observe the red car in the right of the image and the truck in the back can’t be segmented as car labeling properly.


### 3.1.2 Training with new processed data: better performance
Next, we added 200 images from the BDD database to train a new model based on the COCO pre-trained weights. Considering the model performance and time consuming, we chose to run 1000 steps x 30 epochs. After adding new training data, the model performs better, which shows increased mAP value for both training and validation set. The following pictures show that our trained model also outperforms the COCO model in the segmentation of testing images.

<img src="/display/figure15.png" alt="figure15"/>

Figure 12 Mask-RCNN model performs better than pre-trained model. The mAP for COCO pre-trained model and trained Mask-RCNN model are 0.40 and 0.46, respectively. From above images, we can see (a) COCO model fails to detect the red cars in the right of image but (b) our trained model is able to detect it.

However, the increase of steps and epoch number will significantly increase the running time of the training process. In our initial running, we attempted to finish our task in a CPU condor, but that can’t be done within 72 hours limit. We found the running time of Mask-RCNN can be significantly reduced by using a GPU device. Table 4 showed the running time for CPU and GPU
 <p align="center">
<img src="/display/table4.png" alt="table4"/>
 </p>
Table 4 CPU and GPU running time for Mask-RCNN.

So we tested different hyperparameters and tried to get the best-fit model as our Mask-RCNN model. In figure 13, we observed that with 100 steps and 30 epoch, we were able to obtain the best accuracy for validation data. And 100 steps and 50 epochs may be overfitting and lead to reduced mAP for validating dataset.
<p align="center">
<img src="/display/figure16.png" alt="figure16"/>
 </p>
Figure 13 Performance evaluation of different steps and epoch. With more epochs and more steps, the accuracy for training data is increasing, but the optimum setup is 100 x 30epoch. 100 x 50 epoch caused a drop of mAP in validation data, indicating the existence of overfitting.


### 3.1.3 Highly Dependent Different Scenario: Day/Night
In the next step, we split our validation dataset into two groups, day-time group and night-time group. Not surprisingly, we found that the difference of mAP between two groups is significant (0.54 vs 0.38), which might be due to lighting and resolution at night time (Figure 14, 15).
<p align="center">
<img src="/display/figure17.png" alt="figure17" align="center"/>
</p>
Figure 14 mAP varies for day-time and night-time images. Based on our optimized Mask-RCNN model, day-time mAP is as high as 0.54 but night-time mAP is only 0.38.
<img src="/display/figure182.png" alt="figure182"/>
Figure 15 Demon of segmentation from day-time and night-time images. We can observe some incorrect labeling in the night-time images, which are rarely found in day-time images.

### 3.1.4 Improved mAP When Training by Separate Datasets

Then we split the training data into day-time and night-time groups and train our Mask-RCNN model separately. This strategy, although slightly increased the mAP for night-time images, decreased the day-time images segmentation accuracy, which might be due to the relatively small of our dataset and shrink of the total images number in the separate training process (Figure 16, 17).
<p align="center">
<img src="/display/figure19.png" alt="figure19" align="center"/>
 </p>
Figure 16 Comparison of combined-training and separate-training for day and night time images segmentation. There is a minor difference between using the above two strategies, and getting more equal mAP when training separately.
<img src="/display/figure20.png" alt="figure20" width="1000"/>
Figure 17 Night-specific training improves mAP than the combined training model. (a) combined-training model for segmentation of night-time images (b) night-time specific images training model for segmentation of night-time images. We can see night-specific training outputted better results than the combined model.

## 3.2 Object detection with Bounding Box Annotation
### 3.2.1 Faster RCNN
In this experiment we trained Faster RCNN with 8000 training data with bounding box annotation from the 10,000 BDD dataset. We included 6 main objects in the dataset and evaluated the performance of detection by recall and AP. 

The following are the prediction results of 1000 testing data based on the trained model. Considering the tradeoff between model performance and time cost, the model was trained for 12 epoches, where each one includes 4000 steps. The training took about 5 hours when Euler is quite busy. 

### 3.2.2 YOLOv4 Model 
The following shows the result of YOLOv4 ran with the same dataset. In this part. We split testing data into daytime and nighttime. The  models were trained by 6 epoches, where each one includes 1000 steps. The model didn’t perform better when set a bit more epoches. And therefore we stopped the training in a relatively short epoch. The training took about 1 hour during busy hours, which is much faster than faster R-CNN. Based on our result, we first find out that this light-weighted has worse performance, where recall and ap are lower in general. Besides, we did not see a significant difference when it comes to the performance of cars and traffic lights. However, the traffic signs, rider, motor, and person ones have different performance, which means that their features may be different between two scenarios.
#
<img src="/display/table5.png" alt="table5" width="1000"/>
In table 5, we see that the Faster RCNN model prediction performance depends on the object, but it performs well on detecting cars. And Faster RCNN performs better than YOLOV4.

### 3.2.3 YOLOv5 Model
YOLOv5 is the latest version of the series of models. We trained our car model in YOLOV5 starting from the pre-weight given by COCO dataset (the big pre-weight, yolov5x.pt). Due to the time limit, we cannot make it to transform the 10K dataset to the specific format for this model. Hence, we used the 200 samples one instead. In the training process, users could visualize the segmentation and make changes.


The most important character of YOLOv5 is its high performance. We found the training process is faster and efficient than Mask-RCNN. For epochs of 500, it only takes 1 hour to finish the training.  The trained model obtained pretty good accuracy (mAP at around 0.85 for training dataset and 0.44 for validation dataset) and PR-curve area can be 0.821 (Figure 18), suggestive of its acceptable performance. However, the performance is worse than YOLOv4. Although it’s not a good idea to compare the results of different datasets. It is also reasonable to infer that there may exist room for improvement. We may want to do more model tuning in our future works.


<img src="/display/figure21.png" alt="figure21"/>
Figure 18:  Precision, recall, mAP and PR-curve of YOLOV5. The mAP for training data can come to 0.9 with 500 epochs, and the PR-curve area can come to 0.821 in our training model.


## 3.3 Comparison of Mask-RCNN and YOLOv5 
As we can see in Table 5, YOLOv5 is a much faster algorithm to train our own car model and get acceptable segmentation than Mask-RCNN. As shown below, YOLOV5 can also finish the car detection task in a faster and efficient way for cars under both day-time and night-time conditions, which might be more useful in future autonomous driving context. 

The major difference between Mask-RCNN and YOLOV5 is they are focusing on different problems. Mask-RCNN is powerful in instance segmentation and YOLOV5 is more efficient in object detection. They can both detect the object of interest in the images but YOLOV5 doesn’t have the function to finish the object segmentation. There are many extensions of YOLO architecture for instance segmentation tasks, such as SEG-YOLO or YOLACT. For instance, SEG-YOLO is able to conduct instance segmentation by adding a fully connected neural network to YOLOV3. Due to the time of this project, these extensions are beyond the scope of this study.


<img src="/display/figure22.png" alt="figure22"/>
Figure 19:  Compare the instance segmentation and object detection results based on Mask-RCNN and YOLOV5 for day-time images. (a, b) Mask-RCNN segmented images (c,d) YOLOV5  object detection images.

<img src="/display/figure23.png" alt="figure23"/>
Figure 20: Compare the instance segmentation and object detection results based on Mask-RCNN and YOLOV5 for night-time images segmentation. (a, b) Mask-RCNN segmented images (c,d) YOLOV5 object detection images.

# 4. Conclusion
## 4.1 For image segmentation tasks, we got a good result by 200 well-annotated car dataset.
## 4.2 We see that the results of daytime/night data differ and our seperate training model overperforms the combined model.
## 4.3 For object detection tasks, we use a larger dataset and try two types of models. We found that the faster RCNN has better performance while the YOLO one took much shorter time to train.

# 5. Future works
## 5.1 Use larger dataset to see if the performance of mask-rcnn could be even better. 

## 5.2 Transform the 10K dataset to the format that could be run by YOLOv5 and compare the object detection performance of related models.
## 5.3 It would be better that we’re able to find out a low peak period to get related metrics to evaluate. 
## 5.4 Do more model tuning or modify the model structure to gain performance.

<br/>




# References
Ahmed, B., T. A. Gulliver and S. alZahir (2020). "Image splicing detection using mask-RCNN." Signal Image and Video Processing 14(5): 1035-1042.
Al-qaness, M. A. A., A. A. Abbasi, H. Fan, R. A. Ibrahim, S. H. Alsamhi and A. Hawbani (2021). "An improved YOLO-based road traffic monitoring system." Computing 103(2): 211-230.<br/>
Arifin, A. Z. and A. Asano (2006). "Image segmentation by histogram thresholding using hierarchical cluster analysis." Pattern Recognition Letters 27(13): 1515-1521.
Badrinarayanan, V., A. Kendall and R. Cipolla (2017). "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." Ieee Transactions on Pattern Analysis and Machine Intelligence 39(12): 2481-2495.<br/>
Bagloee, S. A., M. Tavana, M. Asadi and T. Oliver (2016). "Autonomous vehicles: challenges, opportunities, and future implications for transportation policies." Journal of Modern Transportation 24(4): 284-303.<br/>
Bai, M., W. J. Luo, K. Kundu and R. Urtasun (2016). "Exploiting Semantic Information and Deep Matching for Optical Flow." Computer Vision - Eccv 2016, Pt Vi 9910: 154-170.<br/>
Bolya, D., C. Zhou, F. Y. Xiao and Y. J. Lee (2019). "YOLACT Real-time Instance Segmentation." 2019 Ieee/Cvf International Conference on Computer Vision (Iccv 2019): 9156-9165.
Cao, Z. G., T. B. Liao, W. Song, Z. H. Chen and C. S. Li (2021). "Detecting the shuttlecock for a badminton robot: A YOLO based approach." Expert Systems with Applications 164.<br/>
Cervantes-Sanchez, F., I. Cruz-Aceves, A. Hernandez-Aguirre, M. A. Hernandez-Gonzalez and S. E. Solorio-Meza (2019). "Automatic Segmentation of Coronary Arteries in X-ray Angiograms using Multiscale Analysis and Artificial Neural Networks." Applied Sciences-Basel 9(24).<br/>
Dudkin, K. N., S. V. Mironov and A. K. Dudkin (1995). "Adaptive Image Segmentation Via Computer Simulation of Vision Processes." Human Vision, Visual Processing, and Digital Display Vi 2411: 310-320.<br/>
Fang, J. X., H. X. Liu, L. T. Zhang, J. Liu and H. S. Liu (2021). "Region-edge-based active contours driven by hybrid and local fuzzy region-based energy for image segmentation." Information Sciences 546: 397-419.<br/>
Geng, K. K., G. Dong, G. D. Yin and J. Y. Hu (2020). "Deep Dual-Modal Traffic Objects Instance Segmentation Method Using Camera and LIDAR Data for Autonomous Driving." Remote Sensing 12(20).<br/>
He, K. M., G. Gkioxari, P. Dollar and R. Girshick (2017). "Mask R-CNN." 2017 Ieee International Conference on Computer Vision (Iccv): 2980-2988.<br/>
Jung, H., M. K. Choi, J. Jung, J. H. Lee, S. Kwon and W. Y. Jung (2017). "ResNet-based Vehicle Classification and Localization in Traffic Surveillance Systems." 2017 Ieee Conference on Computer Vision and Pattern Recognition Workshops (Cvprw): 934-940.<br/>
Le, T. H. N., Y. T. Zheng, C. C. Zhu, K. Luu and M. Savvides (2016). "Multiple Scale Faster-RCNN Approach to Driver's Cell-phone Usage and Hands on Steering Wheel Detection." Proceedings of 29th Ieee Conference on Computer Vision and Pattern Recognition Workshops, (Cvprw 2016): 46-53.<br/>
Martinez, R. P., I. Schiopu, B. Cornelis and A. Munteanu (2021). "Real-Time Instance Segmentation of Traffic Videos for Embedded Devices." Sensors 21(1).<br/>
Nobis, F., M. Geisslinger, M. Weber, J. Betz and M. Lienkamp (2019). "A Deep Learning-based Radar and Camera Sensor Fusion Architecture for Object Detection." 2019 Symposium on Sensor Data Fusion: Trends, Solutions, Applications (Sdf 2019).<br/>
Ognard, J., J. Mesrar, Y. Benhoumich, L. Misery, V. Burdin and D. Ben Salem (2019). "Edge detector-based automatic segmentation of the skin layers and application to moisturization in high-resolution 3 Tesla magnetic resonance imaging." Skin Research and Technology 25(3): 339-346.<br/>
Opara, R. and F. Worgotter (1997). "Introducing visual latencies into spin-lattice models for image segmentation: A neuromorphic approach to a computer vision problem." 1997 Ieee International Conference on Neural Networks, Vols 1-4: 2300-2303.<br/>
Ozan, S. and L. O. Iheme (2019). "Artificial Neural Networks in Customer Segmentation." 2019 27th Signal Processing and Communications Applications Conference (Siu).<br/>
Perez-Perez, Y., M. Golparvar-Fard and K. El-Rayes (2019). "Artificial Neural Network for Semantic Segmentation of Built Environments for Automated Scan2BIM." Computing in Civil Engineering 2019: Data, Sensing, and Analytics: 97-104.<br/>
Redmon, J., S. Divvala, R. Girshick and A. Farhadi (2016). "You Only Look Once: Unified, Real-Time Object Detection." 2016 Ieee Conference on Computer Vision and Pattern Recognition (Cvpr): 779-788.<br/>
Redmon, J. and A. Farhadi (2017). "YOLO9000: Better, Faster, Stronger." 30th Ieee Conference on Computer Vision and Pattern Recognition (Cvpr 2017): 6517-6525.<br/>
Sarkate, R. S., N. V. Kalyankar and P. B. Khanale (2013). "Application of computer vision and color image segmentation for yield prediction precision." Proceedings of the 2013 International Conference on Information Systems and Computer Networks (Iscon): 9-13.<br/>
Tobias, O. J. and R. Seara (2002). "Image segmentation by histogram thresholding using fuzzy sets." Ieee Transactions on Image Processing 11(12): 1457-1465.<br/>
Wang, Y., J. Choi, K. T. Zhang, Q. Huang, Y. R. Chen, M. S. Lee and C. C. J. Kuo (2020). "Video object tracking and segmentation with box annotation." Signal Processing-Image Communication 85.<br/>
Yang, R. X. and Y. Y. Yu (2021). "Artificial Convolutional Neural Network in Object Detection and Semantic Segmentation for Medical Imaging Analysis." Frontiers in Oncology 11.<br/>
Yu, B., S. H. Ma, H. Y. Li, C. G. Li and J. B. An (2020). "Real-Time Pedestrian Detection for Far-Infrared Vehicle Images and Adaptive Instance Segmentation." Laser & Optoelectronics Progress 57(2).<br/>
Zhao, J. D., C. J. Li, Z. Xu, L. X. Jiao, Z. M. Zhao and Z. B. Wang (2021). "Detection of passenger flow on and off buses based on video images and YOLO algorithm." Multimedia Tools and Applications.<br/>
Zhao, J. D., B. D. Liang and Q. X. Chen (2018). "The key technology toward the self-driving car." International Journal of Intelligent Unmanned Systems 6(1): 2-20.<br/>
Zhao, X. T., W. Li, Y. F. Zhang, T. A. Gulliver, S. Chang and Z. Y. Feng (2016). "A Faster RCNN-based Pedestrian Detection System." 2016 Ieee 84th Vehicular Technology Conference (Vtc Fall).<br/>
mmdetection: https://github.com/open-mmlab/mmdetection.<br/>
<br/>


