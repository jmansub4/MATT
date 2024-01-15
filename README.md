# Multispectral Automated Transfer Technique (MATT)

Segment Anything (SAM) is drastically accelerating the speed and accuracy of automatically segmenting and labeling large Red-Green-Blue (RGB) imagery datasets. However, SAM is unable to segment and label images outside of the visible light spectrum, for example, for multispectral or hyperspectral imagery. Therefore, this paper outlines a method we call the Multispectral Automated Transfer Technique (MATT). By transposing SAM segmentation masks from RGB images we can automatically segment and label multispectral imagery with high precision. For example, the results demonstrate that segmenting and labeling a 2,400-image dataset utilizing MATT achieves a 99% time reduction in developing a trained model, from approximately 50.6 hours of manual labor, to only 1.1 hours. We also find that when training multispectral models with MATT, the overall mean average precision (mAP) decreased by only 6.7% when compared to models trained on a manually labeled dataset. We consider this an acceptable level of precision loss when considering the time saved during training, especially for rapidly prototyping experimental modeling methods. This research greatly contributes to the study of multispectral object detection by providing a novel and open-source method to rapidly segment, label, and train multispectral object detection models with minimal human interaction. Future research needs to focus on applying these methods to (i) space-based multispectral, and (ii) drone-based hyperspectral imagery. 

![de:hub.de-Projekt](matt2.png)
