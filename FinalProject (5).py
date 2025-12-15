#!/usr/bin/env python
# coding: utf-8

# In[64]:


import os
import ibis

import altair as alt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import mean_squared_error
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Simplify working with large datasets in Altair
alt.data_transformers.disable_max_rows();


# In[2]:


upfit_data = pd.read_excel("NewUpfitVehiclesMoved2025.xlsx")
upfit_data


# This Dataset is a row per vehicle. Vehicles are supplied from the company Work Truck Solutions (WTS) from a multitude of dealers utilizing WTS' advertising services. The data is anonymized to protect dealer's sensitive information. 

# ## Descriptive Question

# What is the largest category of Body type? How fast do does that body type sell, i.e., what is its Average DTT (days to turn)?

# ## Exploratory Question

# Is there an association between Buyer Engagements (The amount of clicks on a vehicles web page) and Web Leads (specific clicks where an interested buyer fills out a form requesting for information on a specific vehicle)?

# ## Predictive Question

# Will a specific vehicle receive any web leads (interested buyers filling out a form for more information on the vehicles page)?

# ## Inferential Question

# Does DTT change based on Buyer Interactions for all vehicles listed using WTS advertising services?

# # Data Preparation

# In[3]:


# Rename Columns
col_map = {
    "Searches Body Type" : "searches_body_type",
    "Searches Model" : "searches_model",
    "Body Type" : "body_type",
    "Model" : "model",
    "Fuel Type" : "fuel_type",
    "DTT" : "dtt",
    "Buyer Engagements" : "buyer_engagements",
    "Web Leads" : "web_leads",
    "Final Price" : "final_price",
    "Searches Make" : "searches_make"
}
upfit_data_renamed = upfit_data.rename(columns=col_map)
upfit_data_renamed


# In[4]:


# Getting only the columns I care about
upfits_subset = upfit_data_renamed[["searches_body_type", "searches_model", "body_type", "model",
                            "fuel_type", "dtt", "buyer_engagements", "web_leads", "final_price", "searches_make"]]
upfits_subset


# ## Describing the variables in the dataset

# A Body Type Search (search_body_type) is defined as the number of searches for a specific body type (the body type that a vehicle is upfit with) in the 6 months leading up to it being sold. Meaning, two vehicles of the same body type that are sold on the same day will have the same searches by body type, regardless of how long it takes for each vehicle to be sold. 
# 
# A Model Search (search_model) is similar to a Body Type Search, but for searches for a specific model.
# 
# A Make Search (search_make) is similar to a Body Type Search, but for searches for a specific make.
# 
# Body Type (body_type) gives us the body type that the vehicle is. Same idea for Model (model).
# 
# Fuel Type (fuel_type) explains what type of fuel the vehicle receieves.
# 
# Days to turn (dtt) are the amount of days it takes from when a vehicle enters a dealer's inventory to when it leaves their inventory, also known as the vehicle being "Moved". There are a few reasons a vehicle is moved, including to be sent to an upfitter, or being sold, as well as other reasons. For simplicity, and since these are all already-upfit vehicles, we consider DTT to be the days it takes for a vehicle to be sold, since that is the majority of cases. Lower DTT is considered better, since that implies a vehicle is selling quickly.
# 
# Buyer Engagements (buyer_engagements) refer to the amount of clicks on a vehicle's webpage. Clicking through photos, clicking share or print, or interacting with any other buttons or features on the page, is considered a buyer engagement. 
# 
# Web Leads (web_leads) are a specific type of buyer engagement. On the vehicles page is an "I'm Interested" button, that, when clicked, opens up a web form that interested buyers can fill out for further information on the vehicle. When a buyer fills out the form and submits it, it is counted as 1 web lead. 
# 
# Final Price (final_price) refers to the price of the vehicle when sold. Some dealer's neglect to provide this information, whether because the vehicle wasn't sold, or because they don't follow up with us for what it sold for, and a 0 is placed in those instances. In the following line of code, those 0's are replaced will null values.

# In[5]:


# When price is input as 0, it is actually because a dealer doesn't want to give us that information. So we will replace 0 with null.
upfits_subset.loc[upfits_subset['final_price'] == 0, 'final_price'] = np.nan


# In[6]:


upfits_subset.info()


# 3,105 vehicles will null prices. (33902 - 30797). All variables are the correct type.

# In[7]:


upfits_subset["body_type"].unique()


# We see above the amount of different body types we have. We will be narrowing this analysis down to fewer categories below, specifically the top 5 largest categories and an "other" category containing the remaining body types.

# In[8]:


# Make a copy
upfits_subset = upfits_subset.copy()

# Get the top 5 most frequent categories
top5 = upfits_subset['body_type'].value_counts().nlargest(5).index

# Create a new column with only top 5, everything else is "Other"
upfits_subset.loc[:, 'body_type_top5'] = upfits_subset['body_type'].where(
    upfits_subset['body_type'].isin(top5),
    other='Other'
)


# In[9]:


upfits_subset["body_type_top5"].unique()


# Rather than looking at average amount of Web leads, let's simply consider whether a vehicle received a web lead or not. This will be informative for our analysis.

# In[10]:


upfits_subset["received_lead"] = np.where(
    upfits_subset["web_leads"] >= 1,
    "Received at least 1 Web Lead",
    "No Web Leads"
)


# In[11]:


upfits_subset["received_lead"].unique()


# In[12]:


upfits_subset["received_lead"].value_counts()


# In[13]:


upfits_subset["received_lead"].value_counts(normalize=True)


# Approximately 3/4 (76.2%, or 25,827 vehicles) of all vehicles recieve no web leads.

# In[14]:


upfits_subset.info()


# ## Descriptive Question

# What is the largest category of Body type? How fast do does that body type sell, i.e., what is its Average DTT (days to turn)?

# In[15]:


bars = alt.Chart(
    upfits_subset,
    title="Amount of each Body Type"
).mark_bar().encode(
    x=alt.X("body_type:N").title("Body Type"),
    y="count():Q"
)

text = alt.Chart(upfits_subset).mark_text(
    dy=-5
).encode(
    x="body_type:N",
    y="count():Q",
    text="count():Q"
)

upfits_bar = bars + text
upfits_bar


# The largest category is Service Trucks, with 11,210 total Service Trucks. The second highest body type are Flatbed Trucks, at 3,810. The third and fourth largest are Upfitted Cargo Vans and Box Vans, with 3,202 and 3,110, respectively. The fifth largest category is Service Utility Vans with 2,350 vehicles.

# In[16]:


# Summarize counts
summary = (
    upfits_subset.groupby("body_type_top5")
    .size()
    .reset_index(name="count")
)

# Add percentages
total = summary["count"].sum()
summary["percent"] = (summary["count"] / total) * 100


bars3 = alt.Chart(
    summary,
    title="Amount of each of the top 5 Body Types"
).mark_bar().encode(
    x=alt.X("body_type_top5:N").title("Body Type"),
    y=alt.Y("count:Q", title="Count")
)

text_count = alt.Chart(summary).mark_text(
    dy=-5
).encode(
    x="body_type_top5:N",
    y="count:Q",
    text="count:Q"
)

text_pct = alt.Chart(summary).mark_text(
    dy=-10,
    color="black"
).encode(
    x="body_type_top5:N",
    y="count():Q",
    text=alt.Text("percent:Q", format=".1f")
)

upfits_bar2 = bars3 + text_count + text_pct
upfits_bar2


# Our largest category is Service Trucks, with Service Trucks making up 33.1% (11,210) of total vehicles. The second highest body type are Flatbed Trucks, at 11.2% (3,810). The third and fourth largest are Upfitted Cargo Vans and Box Vans, with 9.4% (3,202) and 9.2% (3,110), respectively. The fifth largest category is Service Utility Vans at 6.9% (2,350) of total vehicles. The other category, of all the vehicles not in the top 5, makes up 30.1% (10,220) of total vehicles.

# In[17]:


# Percentages of each group (labeled on the bottom of the bar graph above as well)
100 * upfits_subset.groupby("body_type_top5").size() / upfits_subset.shape[0]


# In[18]:


# DTT by body type
bars2 = alt.Chart(
    upfits_subset,
    title="DTT by Body Type"
).mark_bar().encode(
    x=alt.X("body_type_top5:N").title("Body Type"),
    y=alt.Y("mean(dtt):Q").title("Days to Turn")
)

    
text2 = alt.Chart(upfits_subset).mark_text(
    dy=-5
).encode(
    x="body_type_top5:N",
    y="mean(dtt):Q",
    text=alt.Text("mean(dtt):Q", format=".0f")
) 

dtt_bars =  bars2 + text2
dtt_bars


# Recall that our largest category was Service Trucks. The average DTT for Service Trucks is 240 days, or roughly 8 months. We see that our top 5 categories range from dtt of as low as 180 days for Service Utility Vans (about 6 months) to as much as 392 days for Box Vans (about 13 months).

# ## Exploratory Question

# Is there an association between Buyer Engagements (The amount of clicks on a vehicles web page) and Web Leads (a specific click where an interested buyer fills out a form requesting for information on a specific vehicle)?

# In[19]:


# Only Rows that are Service Trucks
# service_truck = upfits_subset[upfits_subset["body_type"] == "Service Truck"]


# In[20]:


scatter = alt.Chart(upfits_subset, title="Buyer Engagements vs. Web Leads").mark_circle(size=10).encode(
    x=alt.X("buyer_engagements:Q").title("Buyer Engagements"),
    y=alt.Y("web_leads:Q").title("Web Leads"),
    color="body_type")
loess = (
    alt.Chart(upfits_subset)
    .transform_loess(
        "buyer_engagements",   
        "web_leads"           
    )
    .mark_line(size=2)
    .encode(
        x="buyer_engagements:Q",
        y="web_leads:Q",
        color=alt.value("black")   
    )
)

chart = scatter + loess
chart


# In[21]:


# Correlation Coefficient
upfits_subset['buyer_engagements'].corr(upfits_subset['web_leads'])


# With a correlation coefficient of 0.28, we have |0.28| < 0.4, implying a weak or no relationship. If there is a relationship, though weak, it would be positive, implying as buyer engagements increases, the number of web leads also increases. Form appears to be non-linear.

# In[22]:


avg_bodytype = (
    upfits_subset
    .groupby("body_type", as_index=False)
    .agg(
        avg_buyer_engagements=("buyer_engagements", "mean"),
        avg_web_leads=("web_leads", "mean")
    )
)

scatter2 = alt.Chart(avg_bodytype, title="Avg Buyer Engagements vs. Avg Web Leads by Body Type").mark_circle(size=25).encode(
    x=alt.X("avg_buyer_engagements:Q").title("Avg Buyer Engagements"),
    y=alt.Y("avg_web_leads:Q").title("Avg Web Leads"),
    color="body_type",
    tooltip=["body_type", "avg_buyer_engagements", "avg_web_leads"] 
)

loess2 = (
    alt.Chart(avg_bodytype, title="Avg Buyer Engagements vs. Avg Web Leads by Body Type")
    .transform_loess(
        "avg_buyer_engagements",   
        "avg_web_leads"           
    )
    .mark_line(size=2)
    .encode(
        x="avg_buyer_engagements:Q",
        y="avg_web_leads:Q",
        color=alt.value("black")   
    )
)


scatter2 + loess2


# In[23]:


# Correlation Coefficient
avg_bodytype['avg_buyer_engagements'].corr(avg_bodytype['avg_web_leads'])


# I thought that perhaps aggregating over body type would help me understand the relationship between average buyer engagements and average web leads for each body type. The correlation coefficient here is 0.48, and since 0.4 < |.48| < 0.6, we have a moderate-strength positive association between average buyer engagements and average web leads for each body type. This means as average buyer engagements increase, so do average web leads.

# In[24]:


upfits_subset['received_lead'].dtype


# In[25]:


lead_bar1 = alt.Chart(upfits_subset, title="Buyer Engagements vs. Web Leads")

bars1 = (
    alt.Chart(upfits_subset, title="Total Vehicles Receiving at Least 1 Web Lead").mark_bar()
    .encode(
        x=alt.X("received_lead:N", title="Web Leads"),
        y=alt.Y("count():Q", title="Count")
    )
)

labels1 = (
    lead_bar1.mark_text(dy=-5, fontSize=12)
    .encode(
        x=alt.X("received_lead:N"),
        y="count():Q",
        text=alt.Text("count(received_lead):Q")   # count labels
    )
)

(bars1 + labels1)


# I noticed that a lot of vehicles get no web leads at all. On each bar is the count of vehicles that received either no web leads or at least 1 web lead. 25,827 vehicles received no web leads, and 8,075 received at least 1. 

# In[26]:


lead_bar = alt.Chart(upfits_subset, title="Average Buyer Engagements by Received Lead")

bars = (
    lead_bar.mark_bar()
    .encode(
        x=alt.X("received_lead:N", title="Web Leads"),
        y=alt.Y("mean(buyer_engagements):Q", title="Avg Buyer Engagements")
    )
)

labels = (
    lead_bar.mark_text(dy=-5, fontSize=12)
    .encode(
        x=alt.X("received_lead:N"),
        y="mean(buyer_engagements):Q",
        text=alt.Text("mean(buyer_engagements):Q", format=".1f")   # count labels
    )
)

(bars + labels)


# Of the only 8,075 vehicles receiving at least 1 web lead, these vehicles received, on average, approximately 494 buyer engagements. Of the majority 25,827 vehicles that received no web leads, they received about 189 buyer engagements on average. 

# In[27]:


lead_bar3 = alt.Chart(upfits_subset, title="median Buyer Engagements by Received Lead")

bars3 = (
    lead_bar.mark_bar()
    .encode(
        x=alt.X("received_lead:N", title="Web Leads"),
        y=alt.Y("median(buyer_engagements):Q", title="Buyer Engagements")
    )
)

labels3 = (
    lead_bar.mark_text(dy=-5, fontSize=12)
    .encode(
        x=alt.X("received_lead:N"),
        y="median(buyer_engagements):Q",
        text=alt.Text("median(buyer_engagements):Q", format=".1f")   # count labels
    )
)

(bars3 + labels3)


# Above, the median buyer engagements is plotted. So while there must be some outliers in each category skewing the average to the right, the overall trend is still the same; that is, vehicles that received at least 1 web lead have higher buyer engagements than those receiving none.

# ## Predictive Question

# Will a specific vehicle of one of the top 5 body types receive any web leads (interested buyers filling out a form for more information on the vehicles page)?

# Since the goal is to make a categorical prediction on whether a specific vehicle is likely to receive any web leads, we use classification with K-nearest neighbors.

# In[28]:


#limiting to the top 5 body types (no other category)
upfits_subset2 = upfits_subset.copy()

top5 = upfits_subset2['body_type'].value_counts().nlargest(5).index

upfits_subset2['body_type_top5'] = upfits_subset2['body_type'].where(
    upfits_subset2['body_type'].isin(top5)
)

# filtering out the "other" body types
upfits_top5_only = upfits_subset2.dropna(subset=['body_type_top5'])


# In[29]:


upfits_top5_only["body_type"].unique()


# In[30]:


set_config(transform_output="pandas")


# In[31]:


# Set a random seed
np.random.seed(1)


# Setting a random seed allows randomly produced numbers to be randomized but our result to be reproducible to anyone who uses the same random seed. 

# See how the amount of vehicles of one of the top 5 body types that received any web leads are distributed:

# In[32]:


upfits_top5_only["received_lead"].value_counts()


# In[33]:


upfits_top5_only["received_lead"].value_counts(normalize=True)


# Approximately 77% of the top 5 body type receive no web leads.

# Consider the possible associaton between Buyer engagements (clicks on a vehicle's webpage) and Body type searches (the amount of searches for a a vehicle's body type in the 6 months leading up to it being sold).

# In[34]:


buyer_searches = alt.Chart(upfits_top5_only).mark_circle().encode(
    x=alt.X("buyer_engagements").title("Buyer Engagements"),
    y=alt.Y("searches_body_type").title("Body Type Searches"),
    color=alt.Color("received_lead").title("received_lead")
)
buyer_searches


# Visually inspecting the scatter plot above, it appears that the more buyer engagements there are, the more vehicles seem to receive at least one web lead.

# ### K-Nearest Neighbors

# In[35]:


# Creating the training and testing sets
upfits_train, upfits_test = train_test_split(
    upfits_top5_only, train_size=0.75, stratify=upfits_top5_only["received_lead"]
)
upfits_train.info()


# In[36]:


upfits_test.info()


# In[75]:


# prepare the preprocessor
upfits_preprocessor = make_column_transformer(
    (StandardScaler(), ["buyer_engagements", "dtt"]),
)


# In[83]:


#Train the classifier
knn = KNeighborsClassifier(n_neighbors=88)

X = upfits_train[["buyer_engagements", "dtt"]]
y = upfits_train["received_lead"]

knn_pipeline = make_pipeline(upfits_preprocessor, knn)
knn_pipeline.fit(X, y)

knn_pipeline


# Creating our model:

# In[84]:


# make a prediction about a new observation
new_obs = pd.DataFrame({"buyer_engagements": [1000], "dtt": [100]})
knn_pipeline.predict(new_obs)


# In[85]:


# make a prediction about a new observation
new_obs2 = pd.DataFrame({"buyer_engagements": [250], "dtt": [100]})
knn_pipeline.predict(new_obs2)


# For example, the model predicts that a vehicle with 1000 clicks on its page and 100 days to turn by the vehicles body type will receive at least 1 web lead, while a vehicle with 250 clicks and the same amount of dtt will not receive any web leads.

# In[86]:


upfits_test["predicted"] = knn_pipeline.predict(upfits_test[["buyer_engagements", "dtt"]])
upfits_test[["received_lead", "predicted"]]


# #### Evaluating Performance of the model

# In[87]:


knn_pipeline.score(
    upfits_test[["buyer_engagements", "dtt"]],
    upfits_test["received_lead"]
)


# This shows that the estimated accuracy of our model on the test data is 74%.

# In[43]:


precision_score(
    y_true=upfits_test["received_lead"],
    y_pred=upfits_test["predicted"],
    pos_label="Received at least 1 Web Lead"
)


# In[44]:


recall_score(
    y_true=upfits_test["received_lead"],
    y_pred=upfits_test["predicted"],
    pos_label="Received at least 1 Web Lead"
)


# This shows that the estimated precision and recall of whether or not a web lead was received on the test data was 44& and 34%.

# In[45]:


pd.crosstab(
    upfits_test["received_lead"],
    upfits_test["predicted"]
)


# The confusion matrix shows that 404 observations were correctly predicted as receiving at least 1 web lead, and 3,953 were correctly predicted as receiving none. 935 were predicted as receiving no web leads, when they actually received one, a false negative, and 629 were predicted to have received at least 1 when they received none, a false positive.

# Since 76% of our dataset received no web leads, the majority classifier would predict all tested vehicles as receiving no web leads. Thus, the estimated accuracy of a majority estimator would be about 76%. Since our model has an accuracy of 74%, it performs slightly worse. Let's try again with more predictors.

# #### Cross-validation - 5 fold validation

# In[151]:


# Use all numeric columns except the target
feature_cols = [
    "buyer_engagements", "dtt"
]
X_train = upfits_train[feature_cols]
y_train = upfits_train["received_lead"]

X_test = upfits_test[feature_cols]
y_test = upfits_test["received_lead"]

# Preprocessor
vehicle_preprocessor = make_column_transformer(
    (StandardScaler(), feature_cols),
    remainder="drop"
)


# In[152]:


knn = KNeighborsClassifier(n_neighbors=73)
vehicle_pipe = make_pipeline(upfits_preprocessor, knn)
cv_5_df = pd.DataFrame(
    cross_validate(
        estimator=vehicle_pipe,
        cv=5,
        X=X,
        y=y
    )
)

cv_5_df


# In[153]:


# Fit
vehicle_pipe.fit(X_train, y_train)

# Evaluate
y_pred = vehicle_pipe.predict(X_test)

vehicle_pipe

#Create a model object, combine the model object and preprocessor into a pipeline, then use fit method to build classifier


# In[154]:


upfits_test["predicted"] = vehicle_pipe.predict(upfits_test[feature_cols])
upfits_test[["received_lead", "predicted"]]
#use the k-nearest neighbor classifier object to predict the class labels for our test set and augment the original test data with a column  of predictions
#class variable contains diagnosis while predicted is the predicted diagnosis


# In[155]:


precision_score(
    y_true=upfits_test["received_lead"],
    y_pred=upfits_test["predicted"],
    pos_label="Received at least 1 Web Lead"
)


# Estimated precision of our classifier's performance on the dataset is 61%.

# In[156]:


recall_score(
    y_true=upfits_test["received_lead"],
    y_pred=upfits_test["predicted"],
    pos_label="Received at least 1 Web Lead"
)


# Estimated recall of our classifier's performance on the dataset is 20%.

# In[157]:


vehicle_pipe.score(
    upfits_test[feature_cols],
    upfits_test["received_lead"]
)
#evaluating performance


# ### Confusion Matrix

# In[158]:


pd.crosstab(
    upfits_test["received_lead"],
    upfits_test["predicted"]
)


# The confusion matrix shows that 267 observations were correctly predicted as receiving at least 1 web lead, and 4411 were correctly predicted as receiving none. 1072 were predicted as receiving no web leads, when they actually received one, a false negative, and 171 were predicted to have received at least 1 when they received none, a false positive.

# Since 77% of our dataset received no web leads, the majority classifier would predict all tested vehicles as receiving no web leads. Thus, the estimated accuracy of a majority estimator would be about 77%. Since our model has an accuracy of 79%, it performs slightly better than simply guessing.

# In[127]:


vehicle_pipe.get_params()


# In[159]:


parameter_grid = {
    "kneighborsclassifier__n_neighbors":range(1,101,2),
}


# In[160]:


upfits_tune_grid = GridSearchCV(
    estimator=vehicle_pipe,
    param_grid=parameter_grid,
    cv=5
)


# In[161]:


upfits_tune_grid.fit(
    upfits_train[feature_cols],
    upfits_train["received_lead"]
)
accuracies_grid = pd.DataFrame(upfits_tune_grid.cv_results_)
accuracies_grid.info()


# In[162]:


accuracies_grid["sem_test_score"] = accuracies_grid["std_test_score"] / 10**(1/2)
accuracies_grid = (
    accuracies_grid[[
        "param_kneighborsclassifier__n_neighbors",
        "mean_test_score",
        "sem_test_score"
    ]]
    .rename(columns={"param_kneighborsclassifier__n_neighbors": "n_neighbors"})
)
accuracies_grid


# In[163]:


accuracy_vs_k = alt.Chart(accuracies_grid).mark_line(point=True).encode(
    x=alt.X("n_neighbors").title("Neighbors"),
    y=alt.Y("mean_test_score")
        .scale(zero=False)
        .title("Accuracy estimate")
)

accuracy_vs_k


# In[164]:


upfits_tune_grid.best_params_


# Thus, we choose knn of 73.

# In[165]:


upfits_subset1 = upfits_subset[
    ["buyer_engagements", "dtt", "searches_body_type",
    "searches_model", "final_price",
    "searches_make", "received_lead"]
]

names = list(upfits_subset1.drop(
    columns=["received_lead"]
).columns.values)

upfits_subset1


# In[166]:


names = list(upfits_subset1.drop(columns=["received_lead"]).columns.values)
cols_for_model = names + ["received_lead"]
upfits_clean = upfits_subset1.dropna(subset=cols_for_model)

X = upfits_clean[names]
y = upfits_clean["received_lead"]



# In[167]:


preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include="number"))
)


# In[170]:


param_grid = {
    "kneighborsclassifier__n_neighbors": range(1, 100, 4)  # 1 through 101
}


tune_pipe = make_pipeline(
    preprocessor,
    KNeighborsClassifier()
)

# For storing results
accuracy_dict = {"size": [], "selected_predictors": [], "accuracy": []}

selected = []
remaining = names.copy()


for i in range(1, len(names) + 1):
    scores = []

    for feat in remaining:
        X = upfits_clean[selected + [feat]]

        grid = GridSearchCV(
            estimator=tune_pipe,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1
        )

        grid.fit(X, y)

        best_acc = grid.best_score_
        scores.append(best_acc)

    # Select best new feature
    best_idx = int(np.argmax(scores))
    best_feat = remaining[best_idx]
    best_score = scores[best_idx]

    selected.append(best_feat)

    accuracy_dict["size"].append(i)
    accuracy_dict["selected_predictors"].append(", ".join(selected))
    accuracy_dict["accuracy"].append(best_score)

    del remaining[best_idx]

# Convert to DataFrame
accuracies = pd.DataFrame(accuracy_dict)
print(accuracies)


# It appears Buyer engagements and days to turn are the best predictors for this model. 

# In[131]:


knn_pipe = make_pipeline(
    preprocessor,
    KNeighborsClassifier(n_neighbors=73)
)

accuracy_dict = {"size": [], "selected_predictors": [], "accuracy": []}
selected = []
remaining = names.copy()

for i in range(1, len(names) + 1):
    scores = []
    for feat in remaining:
        X = upfits_subset1[selected + [feat]]

        cv_score = cross_val_score(
            knn_pipe,
            X, y,
            cv=5,
            n_jobs=-1
        ).mean()

        scores.append(cv_score)

    best_idx = int(np.argmax(scores))
    best_feat = remaining[best_idx]
    best_score = scores[best_idx]

    selected.append(best_feat)
    accuracy_dict["size"].append(i)
    accuracy_dict["selected_predictors"].append(", ".join(selected))
    accuracy_dict["accuracy"].append(best_score)

    del remaining[best_idx]

accuracies = pd.DataFrame(accuracy_dict)
print(accuracies)

It appears Buyer engagements and days to turn are the best predictors for this model. 
# ## Inferential Question

# Does DTT change based on Buyer Interactions for all vehicles listed using WTS advertising services?

# Buyer Interactions include: Buyer engagements (clicks on the vehicle's page), Web Leads (Interested Buyers filling out a form for more information online), Searches by body type/model.
# 
# DTT refers to the Days to Turn of the vehicle, or how long it took for the vehicle to leave the dealer's inventory once it entered.

# In[ ]:




