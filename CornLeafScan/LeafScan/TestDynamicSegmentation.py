# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-24

import cv2
from LeafExtractor import KmeansLeafExtractor

from Misc.ResizeForDisplay import resize_for_display

original = cv2.imread("/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/View Window_screenshot_26.08.20252.png")   
#original = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    
print(original.shape)

leafExtractor = KmeansLeafExtractor()

isolated_leaf = leafExtractor.Extract(original, display=True, deep_display=True)

cv2.imshow("Original", resize_for_display(original))
cv2.waitKey(0)
cv2.destroyAllWindows()
