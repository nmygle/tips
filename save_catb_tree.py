# catboostでtreeの図を保存する方法
# 参考：https://github.com/catboost/tutorials/blob/master/model_analysis/visualize_decision_trees_tutorial.ipynb

import pickle
import numpy as np
import catboost
from catboost import CatBoostClassifier, Pool

filename = "202111_r2_training_code/model_weights/format_hood_1216_SED_cvfull.cbm"
model = CatBoostClassifier()
model.load_model(filename)

with open("202111_r2_training_code/x_hood.p", "rb") as fp:
    x_data, y_data = pickle.load(fp)
pool = Pool(x_data, y_data)#, cat_features=cat_features_index, feature_names=list(X.columns))

for k in range(100):
    g = model.plot_tree(tree_idx=k, pool=pool)
    g.format = 'png'
    g.render(f"fig/{k:03d}")
