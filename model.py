import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data.csv")   # make sure file name matches

# Antibiotic columns
antibiotics = ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM',
               'GEN', 'AN', 'Acide nalidixique', 'ofx',
               'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine']

# Convert wide → long
df_long = df.melt(
    id_vars=['Souches'],
    value_vars=antibiotics,
    var_name='Antibiotic',
    value_name='Result'
)

df_long.rename(columns={'Souches': 'Bacteria'}, inplace=True)

# Clean data
df_long = df_long.dropna()
df_long = df_long[df_long['Result'].isin(['R', 'S'])]

# Encode
le_bacteria = LabelEncoder()
le_antibiotic = LabelEncoder()
le_result = LabelEncoder()

df_long['Bacteria'] = le_bacteria.fit_transform(df_long['Bacteria'])
df_long['Antibiotic'] = le_antibiotic.fit_transform(df_long['Antibiotic'])
df_long['Result'] = le_result.fit_transform(df_long['Result'])

# Train model
X = df_long[['Bacteria', 'Antibiotic']]
y = df_long['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# Save everything
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_bacteria, open("bacteria_encoder.pkl", "wb"))
pickle.dump(le_antibiotic, open("antibiotic_encoder.pkl", "wb"))

print("✅ Model trained & saved!")