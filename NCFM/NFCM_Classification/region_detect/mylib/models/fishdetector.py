import os, sys
import numpy as np
from mylib.models import Ipv3BboxClassifier, Ipv3BboxRegressor
from mylib.dataio import TestDataReader


file_dir = os.path.dirname(os.path.abspath('__file__'))

# settings
ipv3_bbox_classifier_weights = os.path.join(
    file_dir, 
    'ipv3_bbox_classifier_weights/ipv3_bbox_classifier-00--0.07--0.97.h5',
)
input_shape = (299, 299, 3)
ipv3_bbox_regressor_weights = os.path.join(
    file_dir,
    'ipv3_bbox_regressor_weights_2/ipv3_bbox_regressor_nb_5-184--10.47-19.39.h5',
)
nb_bbox = 5


class AbstractFishDetector(object):
    
    def __init__(self, bbox_regressor, bbox_classifier):
        self.bbox_regressor = bbox_regressor
        self.bbox_classifier = bbox_classifier
        
    def detect(images):
        pass


class FishDetector(AbstractFishDetector):
    
    default_bbox_regressor = Ipv3BboxRegressor(weights_file=ipv3_bbox_regressor_weights, nb_bbox=nb_bbox)
    default_bbox_classifier = Ipv3BboxClassifier(weights_file=ipv3_bbox_classifier_weights, input_shape=input_shape)
    batch_size = 32
    
    def __init__(self, bbox_regressor=default_bbox_regressor, bbox_classifier=default_bbox_classifier):
        super(FishDetector, self).__init__(bbox_regressor, bbox_classifier)
    
    def detect(self, images, thresh=0.5, batch_size=batch_size, verbose=0):
        bboxes_all = self.bbox_regressor.predict_with_slide_bbox(images, batch_size=batch_size)
        
        # predict scores of all boxes
        scores_all = []
        for i in range(len(images)):
            
            if verbose and i % verbose == 0: print("Detecting %s / %s"%(i, len(images)))
            
            image = images[i]
            bboxes = bboxes_all[i]
            bboxes_img = []
            for bbox in bboxes:
                x, y, w, h, p = bbox
                if x <= 0 or y<=0 or w < 1 or h <1:
                    x, y, w, h = 0, 0, 1, 1
                bboxes_img.append(image[y:y+h, x:x+w])
            if len(bboxes_img) == 0:
                scores = np.array([])
            else:
                scores = self.bbox_classifier.predict(bboxes_img, batch_size)
            scores_all.append(scores)
        
        fish_boxes_all = []
        for i in range(len(images)):
            bboxes = bboxes_all[i]
            scores = scores_all[i]
            fish_bboxes = []
            for j in range(len(bboxes)):
                if scores[j] >= 0.5:
                    bboxes[j][0] = int(bboxes[j][0])
                    bboxes[j][1] = int(bboxes[j][1])
                    bboxes[j][2] = int(bboxes[j][2])
                    bboxes[j][3] = int(bboxes[j][3])
                    bboxes[j][4] = int(scores[j]*100)
                    fish_bboxes.append(bboxes[j])
            fish_boxes_all.append(fish_bboxes)
        
        return fish_boxes_all
