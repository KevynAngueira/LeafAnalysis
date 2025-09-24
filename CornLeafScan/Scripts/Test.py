# Author: Kevyn Angueira Irizarry
# Created: 2025-09-24
# Last Modified: 2025-09-24

from LeafScan.Utils import GetLeafRecord
import joblib

f_id = 7
p_id = 6
l_id = 12

X,y = GetLeafRecord(f_id, p_id, l_id)

# Load model
model = joblib.load(MODEL_PATH)
MODEL_PATH = "SavedModels/gradient_boosting_model.pkl"

pred_original_area = model.predict(X)

print(pred_original_area)
