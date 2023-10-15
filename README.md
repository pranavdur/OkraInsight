# OkraInsight : Deeplab mobilenet based Okra segmentation network

Input | Ground Truth | Prediction
---|---|---
![Input](./__artifacts/input_image.png) | ![Ground Truth Mask ](./__artifacts/ground_truth.png) | ![Predicted Output](./__artifacts/predicted_segment_mask.png)


The final goal of this project is to build a machine learning based mobile application on iphone that uses just the real-time camera vision based images of Okra to predict the result of traditional "tip-break-off" test without actually doing it. This is a feasible concept because experts (such as my grandmom) actually are able to confidently "tell" by visually inspecting Okra and willing to forego the tip-break off test. 

The intended impact of this application is to give non-expert users an alternative way to pick right Okra at the grocery store without resorting to destructive (and often discouraged) tip-break-off test. The destructive testing actually contributes to food spoilage and wastage. This act of mutilation leaves "test failed" Okra to be dumped back into the pile along with prospective good Okra. But doing so reduces the likelyhood of even the good ones in the rest of the pile from getting sold. Hence, many vendors explicitly place a notice asking customers not to break Okra to test. And some vendors package them so customers can see but cannot individually break-test before purchase. This iphone based aplication is a cleaner alternative to selecting Okra at your next grocery purchase!

## Folder Organization

The main executable scripts are all Jupyter Notebooks that can be executed in Google colab (or anywhere else where Jupyter execution environment is correctly setup)

File | Purpose
--|--
OkraSegMaskLabelsWithSegmentAnything.ipynb | This notebook can be used to interactively iterate over all Okra images and use segment-anything model from Meta AI research to generate ground truth Okra segmentation masks
ExploreOkraDataSet.ipynb | This notebook can be used to interactively iterate over all the images in the database and visualize both the input data and ground truth masks
MobileOkraSegTrain.ipynb | This notebook can be used to train a customized deeplab mobilenet v3 model for Okra segmentation
ExploreOkraSegnetModel.ipynb | This notebook can be used to interactively iterate over all Okra images and evaluate the trained deeplab mobilenet v3 predictions visually.
*.py | Utility scripts from [<u>Torch Vision Segmentation</u>][DeepLabV3Website]

Folder | Purpose
--|--
training_data | It has 2 sub-folders. The sub-folder <b>okra_images</b> has the iphone images (taken by me for this project) of Okra and converted to jpeg format. The sub-folder <b>okra_segmentation_target_masks</b> has the ground truth mask prediction file obtained from segment-anything model.
__artifacts | images used in README.md
utils | utils sub-folder from [<u>Pytorch-Unet</u>][PytorchUnet]

## Details

### Customizing Deeplab V3 model for Okra segmentation

- A DeeplabHead classifier with single output classification class was added to the model
- Tuned the learning rate and training durations (epochs)

[<i>images captured from wandb dashboard</i>]
![training loss ](./__artifacts/training_loss_function.png) | ![training loss ](./__artifacts/gpu_usage.png)

### Okra mobile segnet
The code in this project can be used to build, train and evaluate a semantic segmentation network for identifying Okra surface within a 2D image containing single Okra against arbitrary background. This project includes preparing images and groundtruth masks as well. Since the goal is to deploy this model on smartphone like device, I build the segmentation network using the mobile hardware compatible open source torch vision model [<u>DeepLabV3 Mobile-Net</u>][DeepLabV3Website] is customized. 

### Image Dataset
To construct the images, the author used iphone 14 to capture images of okra purchased from local grocery stores. Each okra has 2 images (posterior and anterior pose) to cover all sides of the entire Okra surface. Traditionally customers test Okra for ripeness before buying in the market by selecting each Okra based on a tip-break test. This test is done by giving the tip a quick bend to see if it snaps (pass) or just sags (fail). I applied this same test to generate pass/fail results and stores the results in a table linking it to the corresponding images of the Okra surface. Additionally, the Okra is cut in the middle longitudinally and images of the internal guts of Okra are also taken. But these images are not used in training the model at the moment. 

### Ground Truth Label Generation
I tried 2 approaches to generating the ground truth segmentation masks for Okra

#### 1. Manual region marking by drawing bounding-polygon 
- For this, I used [<u>Labelme</u>][LabelmeWebsite] software to manually edit each Okra image to add an aproximately bounding polygon to define the region of pixels that are part of Okra surface. 

#### 2. Manual pointing of 1 or 2 coordinate points to exemplify the region of pixels in Okra surface. 

- For this, I used a pretrained version of [<u>Segment Anything Model</u>][SAMGithub] that was published recently by [<u>Meta AI research</u>][SAMWebsite] which was able to take as inputs 1 or more coordinates that are valid Okra surface region and generate Okra segmentation masks

The approach 2 produced a higher quality Okra segmentation masks. So I relied on this for training the mobile network for Okra segmentation.



[DeepLabV3Website]: <https://github.com/pytorch/vision/tree/main/references/segmentation> "example text"
[SAMGithub]: <https://github.com/facebookresearch/segment-anything>
[SAMWebsite]: <https://ai.meta.com/research/publications/segment-anything/>
[LabelmeWebsite]: <http://labelme.csail.mit.edu/guidelines.html>
[PytorchUnet]: <https://github.com/milesial/Pytorch-UNet/tree/master>

