
# Pothole Detection System Using YOLOv9 model

YOLO v9(You Only Look Once version 9) model is deep learning model which is used for the object detection. This project develops a deep learning model to detect potholes on road images, leveraging the YOLOv9 architecture. Aimed at enhancing urban infrastructure through timely and efficient road maintenance, this solution offers real-time detection capabilities with high accuracy.

# Key Features
   1. Real-time pothole detection with YOLOv8.
   2. High precision and recall, demonstrating effective identification of potholes in diverse road conditions.
 3.   Suitable for integration into urban maintenance workflows to facilitate road repairs.


# Results Summary
## Validation Performance


*    Precision: 79.8%
  *  Recall: 70.8%
   * mAP@0.5: 80.5%
  *  mAP@0.5:0.95: 53.3%
   * Inference Speed: 22.1ms per image, enabling fast processing suitable for real-time applications.

 ## Test Dataset Performance

  *  Demonstrated robust detection across 67 test images, with effective identification of varying numbers of potholes.
  *  Average Inference Speed: ~20.0ms per image, showcasing the model's efficiency in processing images quickly.
# Model Configuration

   * Architecture: YOLOv9 -e variant (YOLOv9-e)
   * Training Parameters: Trained for 30 epochs with a batch size of 15 and image size of 640.
  *  Optimizer: SGD with a learning rate of 0.01, momentum of 0.937, and weight decay of 0.0005.

# Dataset
The model was trained and validated on a custom dataset comprising over 1,300 images, specifically curated for pothole detection, ensuring high model accuracy and reliability.

# Technologies Used

  *  Programming Language: Python
   * Frameworks/Libraries: PyTorch, Ultralytics YOLOv8, OpenCV

# Getting Started
Instructions on setting up the project environment, including prerequisites and installation steps, are provided to help you replicate the model training or to use the model for detecting potholes in new images.
# Usage
Detailed guidelines on how to use the trained model for pothole detection in images or video streams are included, ensuring you can easily integrate this solution into your projects or applications.



# Deployment
 ## Steup Environment
To setup the rquired environment

```bash
  !pip install -r requirements.txt -q
```

  ## Train the model
  Using the following command train the model on pothole dataset.
  ```bash
  !python train_dual.py --workers 8 --device 0 --batch 8 --data 'Pothole-1/data.yaml' --img 640 --cfg models/detect/yolov9-e.yaml --weights '{HOME}/weights/yolov9-e.pt' --name yolov9-e-finetuning --hyp hyp.scratch-high.yaml --min-items 0 --epochs 30 --close-mosaic 15
```
## testing on local machine:
Then test the trained model on the locan ubantu machine.
```bash
  python3 detect.py --weights best.pt --source demo2.mp4 --view-img

```
**##_Colab-notebook_**

**https://github.com/ShubhKokate/Pothole_detection/blob/master/Pothole_Detection_Model_Using_YOLOv9.ipynb**


The tested resulat are shown in the follwing points.


## Demo
**_Result image of the trained model_:**

![results](https://github.com/user-attachments/assets/81c3933d-eb61-4c17-b525-4d9553bdfdf0)

**_Confusion matrix of the trained model_:**

![confusion_matrix](https://github.com/user-attachments/assets/2c0b8886-7fa6-48f7-9bee-eceecd55d03a)

**_Testing on demo video, resulat are shown in the following video_:**

https://github.com/user-attachments/assets/c7e73d76-3678-4398-a6a1-cfe93c1af5a0


**_Testing on the local machine of excution shooted video is below_:**

https://github.com/ShubhKokate/Pothole_detection/blob/master/runs/detect/exp/demo2.mp4

# Youtube video of the trained model and execution for testing on local machine.

**_Model training_:** https://youtu.be/cLdp8E50rhE

**_Testing on local machine_:** https://www.youtube.com/watch?v=2IFUl-RsUVI



## Acknowledgements

 
   * Dataset provided by Roboflow.
   * Model architecture and training supported by Ultralytics.

