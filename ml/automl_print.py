from autogluon.tabular import TabularPredictor

predictor = TabularPredictor.load("/Users/mar/BIO/PROJECTS/MPOX/COMMENT/APOBEC3-Mpox-Hairpins/ml/AutogluonModels/ag-20260709_001339")

print(predictor.leaderboard())