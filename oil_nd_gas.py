import streamlit as st
import pandas as pd
import joblib as jb
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold



st.set_page_config(
    page_title="Oil & Gas Classification",
    page_icon="🛢️",
    layout="wide"
)

st.title("Oil & Gas Classification")

st.header("Dataset")
@st.cache_data
def load_data():
    train_oil = pd.read_csv(
        "train_oil.csv",
        engine='python',
        on_bad_lines='skip',
        sep=","
    )

    train_oil = train_oil[
        train_oil["Onshore/Offshore"] != "ONSHORE-OFFSHORE"
    ]
    return train_oil
train_oil = load_data()


st.dataframe(train_oil.head(20))




st.header("Model Performance")

X = train_oil.drop("Onshore/Offshore", axis=1)

y = train_oil["Onshore/Offshore"]







cat_cols = X.select_dtypes(
    include=["object", "string", "category"]
).columns
num_cols = X.select_dtypes(include=["number"]).columns


cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])



preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, cat_cols),
        ("num", num_transformer, num_cols)
    ]
)






X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)





ran = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model = Pipeline([
    ("prep", preprocessor),
    ("model", ran)
])


model.fit(X_train, y_train)





pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

st.metric(
    "Model Accuracy",
    f"{accuracy:.2%}"
)




class_report = classification_report(y_test, pred)
st.code(class_report)



st.header("Confusion Matrix")


cm = confusion_matrix(y_test, pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Offshore","Actual Onshore"],
    columns=["Pred Offshore","Pred Onshore"]
)

st.dataframe(cm_df)




st.header("Feature Importance")

feature_names = model.named_steps["prep"].get_feature_names_out()

importances = model.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

top_features = importance_df.head(10)


fig = px.bar(

    top_features,

    x="Importance",

    y="Feature",

    orientation="h",

    text="Importance",

    title="Feature Importance"

)

fig.update_traces(texttemplate="%{text:.3f}")

fig.update_layout(yaxis=dict(autorange="reversed"))

st.plotly_chart(
    fig,
    width="stretch"
)





st.header("Cross Validation")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)


st.metric(
    "Cross Validation Accuracy",
    f"{scores.mean():.2%}"
)

categories = {
    "Country": sorted(train_oil["Country"].dropna().unique()),
    "Region": sorted(train_oil["Region"].dropna().unique()),
    "Basin name": sorted(train_oil["Basin name"].dropna().unique()),
    "Operator company": sorted(train_oil["Operator company"].dropna().unique()),
    "Hydrocarbon type": sorted(train_oil["Hydrocarbon type"].dropna().unique()),
    "Field name": sorted(train_oil["Field name"].dropna().unique()),
    "Latitude": sorted(train_oil["Latitude"].dropna().unique()),
    "Longitude": sorted(train_oil["Longitude"].dropna().unique()),
    "Reservoir unit": sorted(train_oil["Reservoir unit"].dropna().unique()),
    "Depth": sorted(train_oil["Depth"].dropna().unique()),
    "Lithology": sorted(train_oil["Depth"].dropna().unique()),
    "Porosity": sorted(train_oil["Porosity"].dropna().unique()),
    "Tectonic regime": sorted(train_oil["Tectonic regime"].dropna().unique()),
    "Reservoir status": sorted(train_oil["Reservoir status"].dropna().unique()),
    "Structural setting": sorted(train_oil["Structural setting"].dropna().unique()),
    "Reservoir period": sorted(train_oil["Reservoir period"].dropna().unique()),
    "Thickness (gross average ft)": sorted(train_oil["Thickness (gross average ft)"].dropna().unique()),
    "Thickness (net pay average ft)": sorted(train_oil["Thickness (net pay average ft)"].dropna().unique()),
    "Permeability": sorted(train_oil["Permeability"].dropna().unique())
}

# st.write(scores)

jb.dump(model, "models/oil_nd_gas.pkl")
jb.dump(categories, "models/categories.pkl")


st.sidebar.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

st.sidebar.metric(
    "CV Accuracy",
    f"{scores.mean():.2%}"
)