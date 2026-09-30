import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

# BINARY CLASSIFICATION BECAUSE OF (APPROVED , NOT APPROVED)
df = pd.read_csv("loan_approval_data.csv")
# print(df.head())
# df.head()
# print(df.info())
# print(df.isnull().sum())
# df.describe()
# print(df.columns)
# print("float columns:", df.select_dtypes(include="float").shape[1])

# Handle missing values
# ======================================
from sklearn.impute import SimpleImputer
df = df.drop("Applicant_ID", axis=1)
numerical_col = df.select_dtypes(include=["number"]).columns
# print(numerical_col)
categorical_col = df.select_dtypes(include=["str"]).columns
# print(categorical_col)

num_imp = SimpleImputer(strategy = "median")
df[numerical_col] = num_imp.fit_transform(df[numerical_col])

cat_col = SimpleImputer(strategy= "most_frequent")
df[categorical_col] = cat_col.fit_transform(df[categorical_col])

# EDA = exploritory data analysis
class_count = df["Loan_Approved"].value_counts()
# plt.pie(class_count, labels= ["yes","No"],autopct="%1.1f%%")
# plt.title("is loan approved or not")
# plt.show()

# gen_cnt = df["Gender"].value_counts()
# plt.pie(gen_cnt, labels= ["male","female"],autopct="%1.1f%%")
# plt.title("MALE OR FEMALE")
# plt.show()

import matplotlib.pyplot as plt
# approved = df[df["Loan_Approved"] == "Yes"]

# gender_count = approved["Gender"].value_counts()

# print(gender_count)

# approved["Gender"].value_counts().plot(
#     kind="pie",
#     autopct="%1.1f%%"
# )

# plt.title("Loan Approval by Gender")
# plt.ylabel("")
# plt.show()

# fig,axes = plt.subplots(2,3)
# sns.boxplot(ax = axes[0,0], data = df,x = "Loan_Approved", y = "Applicant_Income")
# sns.boxplot(ax = axes[0,1], data = df,x = "Loan_Approved", y = "Credit_Score")
# sns.boxplot(ax = axes[0,2], data = df,x = "Loan_Approved", y = "DTI_Ratio")

# sns.boxplot(ax = axes[1,0], data = df,x = "Loan_Approved", y = "Savings")
# sns.boxplot(ax = axes[1,1], data = df,x = "Loan_Approved", y = "Loan_Amount")
# sns.boxplot(ax = axes[1,2], data = df,x = "Loan_Approved", y = "Age")
# plt.tight_layout()
# plt.show()

# sns.histplot(
#     data = df,
#     x = "Credit_Score",
#     hue = "Loan_Approved",
#     multiple = "dodge"
# )
# plt.show()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

col = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
ohe = OneHotEncoder(drop = "first",sparse_output = False , handle_unknown = "ignore")
encoded = ohe.fit_transform(df[col])
encoded_df = pd.DataFrame(encoded, columns = ohe.get_feature_names_out(col),index = df.index)
df = pd.concat([df.drop(columns = col), encoded_df], axis = 1)

# EDA = Heat map for understanding the realtionship between the numerical columns with load approved
num_re = df.select_dtypes(include="number")
cor_matrix = num_re.corr()
num_re.corr()["Loan_Approved"].sort_values(ascending=False)

plt.figure(figsize=(15,8))
sns.heatmap(
    cor_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.tight_layout()
# plt.show()

x = df.drop(["Loan_Approved"],axis = 1)
y = df["Loan_Approved"]

x_trian ,x_test ,y_train ,y_test = train_test_split(
    x,y ,test_size = 0.2, random_state = 42
)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_trian)
x_test_scaled = scalar.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix

# ===========================================
# LOGISTIC REGRESSION
# ===========================================
# log_model = LogisticRegression()
# log_model.fit(x_train_scaled, y_train)
# y_pred = log_model.predict(x_test_scaled)

# ===========================================
# KNN MODEL 
# ===========================================
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors= 7)
knn_model.fit(x_train_scaled,y_train)
y_pred = knn_model.predict(x_test_scaled)

# ===========================================
# GUASSIAN MODEL
# ===========================================
# from sklearn.naive_bayes import GaussianNB
# nb = GaussianNB()
# nb.fit(x_train_scaled,y_train)
# y_pred = nb.predict(x_test_scaled)

# ===========================================
# Decision tree classifier
# ===========================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

Dt = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 5, random_state = 42)



Dt.fit(x_train_scaled,y_train)

y_pred = Dt.predict(x_test_scaled)


print("precision_score: ",precision_score(y_test, y_pred))
print("accuracy_score: ",accuracy_score(y_test, y_pred))
print("recall_score,: ",recall_score(y_test, y_pred))
print("f1_score: ",f1_score(y_test, y_pred))
print("confusion_matrix:",confusion_matrix(y_test, y_pred))