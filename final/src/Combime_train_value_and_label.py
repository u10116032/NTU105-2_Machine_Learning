import pandas as pd

# in:
# pre_train_value.csv
# pre_train_label.csv
# pre_test.csv

# out:
# pre_train_data.csv

train_value = pd.read_csv("../pre_train_value.csv")
train_label = pd.read_csv("../pre_train_label.csv")
test = pd.read_csv("../pre_test.csv")

train_data = train_value.merge(train_label, how = "outer", on = "id", sort = True)
train_data = train_data.fillna(train_data.median())

train_data.to_csv("../pre_train_data.csv", index = False)
