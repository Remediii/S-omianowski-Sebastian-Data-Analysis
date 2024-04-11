import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
import pydotplus


df = pd.read_csv(r"cardio_train.csv", sep=';')

df = df.drop(columns = ['id'])

X = df.iloc[:, 0:11]
y = df.iloc[:, 11]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

export_graphviz(clf, out_file='main_tree.dot', feature_names=X.columns, class_names=['0', '1'], filled=True)
graph = pydotplus.graph_from_dot_file('main_tree.dot')
graph.write_png('main_tree.png')

fig = plt.figure(figsize=(15, 10))
_ = plot_tree(clf, filled=True)
plt.show()

'''
Age: 1 is over 50 years old, 0 is under 50
Gender: 1 woman, 0 man
Systolic Blood Pressure: 1 is above 140, 0 is under 140
Diastolic Blood Pressure: 1 is above 90, 0 is under 90
Cholesterol Level: 1 is above normal, 0 is normal
Glucose Level: 1 is above normal, 0 is normal
Smoking: 1 smoking, 0 not smoking
Alcohol intake: 1 drinking, 0 not drinking
Physical activity: 1 active, 0 not active
'''
df.groupby('gender')['height'].mean()

df['age'] = (df['age'] / 365).round().astype(int)
df['age'] = np.where(df['age'] >= 50, 1, 0)
df['ap_hi'] = np.where(df['ap_hi'] >= 140, 1, 0)
df['ap_lo'] = np.where(df['ap_lo'] >= 90, 1, 0)
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)
df['gender'] = np.where(df['gender'] > 1, 0, 1)

target = df.columns[-1]

#Gender
print(f"\nDistribution of Gender by CVD:")
print(df.groupby(target)['gender'].value_counts())

percantageOfSickWoman = round(len(df[(df['gender'] == 1) & (df[target] == 1)]) / len(df[(df['gender'] == 1)])*100, 2)
percantageOfSickMan = round(len(df[(df['gender'] == 0) & (df[target] == 1)]) / len(df[(df['gender'] == 0)])*100, 2)

print(f"Sick Woman : {percantageOfSickWoman}%, sick man: {percantageOfSickMan}%")

df['cvdPredByGender'] = np.where(df['gender'] < 1, 1, 0)

genderAccuracy = round((len(df[(df['cvdPredByGender'] == 1) & (df[target] == 1)]) + len(df[(df['cvdPredByGender'] == 0) & (df[target] == 0)])) / len(df) *100, 2)
print(f"Gender prediction accuracy: {genderAccuracy}%.")

#Age
print(f"\nDistribution of Age by CVD:")
print(df.groupby(target)['age'].value_counts())

percantageOfSickOlder = round(len(df[(df['age'] == 1) & (df[target] == 1)]) / len(df[(df['age'] == 1)])*100, 2)
percantageOfSickYounger = round(len(df[(df['age'] == 0) & (df[target] == 1)]) / len(df[(df['age'] == 0)])*100, 2)

print(f"Sick older : {percantageOfSickOlder}%, sick younger: {percantageOfSickYounger}%")

df['cvdPredByAge'] = np.where(df['age'] >= 1, 1, 0)

ageAccuracy = round((len(df[(df['cvdPredByAge'] == 1) & (df[target] == 1)]) + len(df[(df['cvdPredByAge'] == 0) & (df[target] == 0)])) / len(df) *100, 2)
print(f"Age prediction accuracy: {ageAccuracy}%.")

#SBP
print(f"\nDistribution of SBP by CVD:")
print(df.groupby(target)['ap_hi'].value_counts())

percantageOfSickAphiAbove = round(len(df[(df['ap_hi'] == 1) & (df[target] == 1)]) / len(df[(df['ap_hi'] == 1)])*100, 2)
percantageOfSickAphiNormal = round(len(df[(df['ap_hi'] == 0) & (df[target] == 1)]) / len(df[(df['ap_hi'] == 0)])*100, 2)

print(f"Sick above : {percantageOfSickAphiAbove}%, sick normal: {percantageOfSickAphiNormal}%")

df['cvdPredByAphi'] = np.where(df['ap_hi'] >= 1, 1, 0)

aphiAccuracy = round((len(df[(df['cvdPredByAphi'] == 1) & (df[target] == 1)]) + len(df[(df['cvdPredByAphi'] == 0) & (df[target] == 0)])) / len(df) *100, 2)
print(f"SBP prediction accuracy: {aphiAccuracy}%.")

#DBP
print(f"\nDistribution of DBP by CVD:")
print(df.groupby(target)['ap_lo'].value_counts())

percantageOfSickAploAbove = round(len(df[(df['ap_lo'] == 1) & (df[target] == 1)]) / len(df[(df['ap_lo'] == 1)])*100, 2)
percantageOfSickAploNormal = round(len(df[(df['ap_lo'] == 0) & (df[target] == 1)]) / len(df[(df['ap_lo'] == 0)])*100, 2)

print(f"Sick above : {percantageOfSickAploAbove}%, sick normal: {percantageOfSickAploNormal}%")

df['cvdPredByAplo'] = np.where(df['ap_lo'] >= 1, 1, 0)

aploAccuracy = round((len(df[(df['cvdPredByAplo'] == 1) & (df[target] == 1)]) + len(df[(df['cvdPredByAplo'] == 0) & (df[target] == 0)])) / len(df) *100, 2)
print(f"DBP prediction accuracy: {aploAccuracy}%.")

#Cholesterol
print(f"\nDistribution of Cholesterol by CVD:")
print(df.groupby(target)['cholesterol'].value_counts())

percantageOfSickCholAbove = round(len(df[(df['cholesterol'] == 1) & (df[target] == 1)]) / len(df[(df['cholesterol'] == 1)])*100, 2)
percantageOfSickCholNormal = round(len(df[(df['cholesterol'] == 0) & (df[target] == 1)]) / len(df[(df['cholesterol'] == 0)])*100, 2)

print(f"Sick above : {percantageOfSickCholAbove}%, sick normal: {percantageOfSickCholNormal}%")

df['cvdPredByChol'] = np.where(df['cholesterol'] >= 1, 1, 0)

cholAccuracy = round((len(df[(df['cvdPredByChol'] == 1) & (df[target] == 1)]) + len(df[(df['cvdPredByChol'] == 0) & (df[target] == 0)])) / len(df) *100, 2)
print(f"Cholesterol prediction accuracy: {cholAccuracy}%.")

#Glucose
print(f"\nDistribution of Glucose by CVD:")
print(df.groupby(target)['gluc'].value_counts())

percantageOfSickGlucAbove = round(len(df[(df['gluc'] == 1) & (df[target] == 1)]) / len(df[(df['gluc'] == 1)])*100, 2)
percantageOfSickGlucNormal = round(len(df[(df['gluc'] == 0) & (df[target] == 1)]) / len(df[(df['gluc'] == 0)])*100, 2)

print(f"Sick above : {percantageOfSickGlucAbove}%, sick normal: {percantageOfSickGlucNormal}%")

df['cvdPredByGluc'] = np.where(df['gluc'] >= 1, 1, 0)

glucAccuracy = round((len(df[(df['cvdPredByGluc'] == 1) & (df[target] == 1)]) + len(df[(df['cvdPredByGluc'] == 0) & (df[target] == 0)])) / len(df) *100, 2)
print(f"Glucose prediction accuracy: {glucAccuracy}%.")

#Smoking
print(f"\nDistribution of Smoking by CVD:")
print(df.groupby(target)['smoke'].value_counts())

percantageOfSickSmoking = round(len(df[(df['smoke'] == 1) & (df[target] == 1)]) / len(df[(df['smoke'] == 1)])*100, 2)
percantageOfSickNonSmoking = round(len(df[(df['smoke'] == 0) & (df[target] == 1)]) / len(df[(df['smoke'] == 0)])*100, 2)

print(f"Sick Smoking : {percantageOfSickSmoking}%, sick not smoking: {percantageOfSickNonSmoking}%")

df['cvdPredBySmoke'] = np.where(df['smoke'] < 1, 1, 0)

smokeAccuracy = round((len(df[(df['cvdPredBySmoke'] == 1) & (df[target] == 1)]) + len(df[(df['cvdPredBySmoke'] == 0) & (df[target] == 0)])) / len(df) *100, 2)
print(f"Smoking prediction accuracy: {smokeAccuracy}%.")

#Alcohol
print(f"\nDistribution of Alcohol by CVD:")
print(df.groupby(target)['alco'].value_counts())

percantageOfSickDrinking = round(len(df[(df['alco'] == 1) & (df[target] == 1)]) / len(df[(df['alco'] == 1)])*100, 2)
percantageOfSickNonDrinking = round(len(df[(df['alco'] == 0) & (df[target] == 1)]) / len(df[(df['alco'] == 0)])*100, 2)

print(f"Sick Drinking : {percantageOfSickDrinking}%, sick not drinking: {percantageOfSickNonDrinking}%")

df['cvdPredByAlco'] = np.where(df['alco'] < 1, 1, 0)

alcoAccuracy = round((len(df[(df['cvdPredByAlco'] == 1) & (df[target] == 1)]) + len(df[(df['cvdPredByAlco'] == 0) & (df[target] == 0)])) / len(df) *100, 2)
print(f"Drinking prediction accuracy: {alcoAccuracy}%.")

#Physical Activity
print(f"\nDistribution of Physical activity by CVD:")
print(df.groupby(target)['active'].value_counts())

percantageOfSickActive = round(len(df[(df['active'] == 1) & (df[target] == 1)]) / len(df[(df['active'] == 1)])*100, 2)
percantageOfSickNonActive = round(len(df[(df['active'] == 0) & (df[target] == 1)]) / len(df[(df['active'] == 0)])*100, 2)

print(f"Sick active : {percantageOfSickActive}%, sick not active: {percantageOfSickNonActive}%")

df['cvdPredByActive'] = np.where(df['active'] < 1, 1, 0)

activeAccuracy = round((len(df[(df['cvdPredByActive'] == 1) & (df[target] == 1)]) + len(df[(df['cvdPredByActive'] == 0) & (df[target] == 0)])) / len(df) *100, 2)
print(f"Physical activity prediction accuracy: {activeAccuracy}%.")

