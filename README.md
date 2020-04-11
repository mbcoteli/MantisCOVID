# MantisCOVID
AI Dataset organizer for the current open datasets for Chest Radiography XRAY images.
Output is train and test txt files via data filtering through the open datasets :

https://github.com/ieee8023/covid-chestxray-dataset

https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

test and train text files are generated to be used for lesion detection (YOLO object detection) and object classifier for 3 cases (COVID-19/ Pneumonia / Normal).

Trained algorithms (Lesion detection and Object classifier) are available at the cloud. You can access through the web page :

https://scan.mantiscope.com/

Calling procedure :

1. git clone https://github.com/ieee8023/covid-chestxray-dataset.git
2. check rsna-pneumonia-detection-challenge folder is available at the current folder
3. python generate_dataset.py

If you have any question or contribution, please contact with us through the email address:

contact@mantiscope.com

