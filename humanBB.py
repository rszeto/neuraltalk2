#!/bin/env python
import json
import shutil
import cv2
import os
import sys
sys.path.append('cocotools/PythonAPI');

from pycocotools.coco import COCO

annFiles = ['coco/annotations/instances_train2014.json', 'coco/annotations/instances_val2014.json']

# Make directories if needed
if not os.path.exists('humanBB/train2014'):
    os.makedirs('humanBB/train2014')
if not os.path.exists('humanBB/val2014'):
    os.makedirs('humanBB/val2014')

# Object to store discovered bounding boxes
storedBoxes = []

for annFile in annFiles:
    # Get whether we're processing training or validation images
    datasetType = 'train2014' if 'train' in annFile else 'val2014'
    
    # Init COCO API
    coco = COCO(annFile)
    # Find "person" category ID
    personId = coco.getCatIds(catNms='person');
    # Get file names of images containing a person
    imgIds = coco.getImgIds(catIds=personId)
    imgNames = [x['file_name'] for x in coco.loadImgs(imgIds)]
    
    # Get annotations corresponding to people in the found images
    annIds = coco.getAnnIds(imgIds=imgIds, catIds=personId)
    anns = coco.loadAnns(annIds)
    
    # Count how many boxes we saved
    numSavedBoxes = 0
    for imgId in imgIds:
        imgFileName = coco.loadImgs(imgId)[0]['file_name']
        imgAnns = filter(lambda x: x['image_id'] == imgId and x['iscrowd'] == 0, anns)
        # Get bounding boxes from the annotations
        bboxes = [ann['bbox'] for ann in imgAnns]
        # Get the largest bounding box
        bigBB = sorted(bboxes, key=lambda bb: bb[2]*bb[3])[-1]
        # Convert bounding box to integers
        bigBB = [int(x) for x in bigBB]
        if bigBB[2] > 128 and bigBB[3] > 128:
            # Copy image
            origPath = 'coco/images/' + datasetType + '/' + imgFileName
            newPath = 'humanBB/' + datasetType + "/" + imgFileName
            shutil.copy2(origPath, newPath)
            # Add to stored bounding boxes
            storedBoxes.append({'file_name':datasetType+"/"+imgFileName, 'bbox':bigBB})
            numSavedBoxes += 1
    
    print 'Saved %d of %d images in %s' % (numSavedBoxes, len(imgIds), datasetType)

json.dump(storedBoxes, open('humanBB/boxesInfo.json', 'w'))