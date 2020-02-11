# LNDbChallange2019
CISC881 LNDb Challange 2019

- our group attempted sub-challenge A: nodule detection
- we constructed our 3D CNN architecture inspired by this paper (Q Dou, “Multilevel Contextual 3-D CNNs for False Positive Reduction in Pulmonary Nodule Detection”, Nov 2018)
- we also borrowed ideas from this github repo (https://github.com/shartoo/luna16_multi_size_3dcnn/tree/f5ac4d4b5458e90c6739ab4ba427706ab2477f28)




https://lndb.grand-challenge.org/Home/

### Aims
The main goal of this challenge is the automatic classification of chest CT scans according to the 2017 Fleischner society pulmonary nodule guidelines for patient follow-up recommendation. The Fleischner guidelines are widely used for patient management in the case of nodule findings, and are composed of 4 classes, taking into account the number of nodules (single or multiple), their volume (<100mm³, 100-250mm³ and ⩾250mm³) and texture (solid, part solid and ground glass opacities (GGO)). Furthermore, three additional sub-challenges will be held related to the different tasks needed to calculate a Fleischner score. The challenge is thus made up of four different parts:

Main Challenge - Fleischner Classification: From chest CT scans, participants must predict the correct follow-up according to the 2017 Fleischner guidelines;<br>
Sub-Challenge A - Nodule Detection: From chest CT scans, participants must detect pulmonary nodules;<br>
Sub-Challenge B - Nodule Segmentation: Given a list of >3mm nodule centroids, participants must segment the nodules in the corresponding chest CT scans;<br>
Sub-Challenge C - Nodule Texture Characterization: Given a list of nodule centroids, participants must classify nodules into three texture classes - solid, sub-solid and GGO.

Teams may choose whether to participate only in the main challenge, in a single or multiple sub-challenges or in all challenges. Each task will be evaluated separately and a prize for the best performing method in each challenge will be awarded.
