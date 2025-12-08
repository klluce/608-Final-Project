#!/usr/bin/env python
# coding: utf-8

# In[5]:


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

# Simplify working with large datasets in Altair
alt.data_transformers.disable_max_rows();


# In[ ]:


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

# Does DTT change based on Buyer Interactions on a vehicles web site page?

# # Data Preparation

# In[8]:


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
}
upfit_data_renamed = upfit_data.rename(columns=col_map)
upfit_data_renamed


# In[9]:


# Getting only the columns I care about
upfits_subset = upfit_data_renamed[["searches_body_type", "searches_model", "body_type", "model",
                            "fuel_type", "dtt", "buyer_engagements", "web_leads", "final_price"]]
upfits_subset


# ## Describing the variables in the dataset

# A Body Type Search (search_body_type) is defined as the number of searches for a specific body type (the body type that a vehicle is upfit with) in the 6 months leading up to it being sold. Meaning, two vehicles of the same body type that are sold on the same day will have the same searches by body type, regardless of how long it takes for each vehicle to be sold. 
# 
# A Model Search (search_model) is similar to a Body Type Search, but for searches for a specific model.
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

# In[10]:


# When price is input as 0, it is actually because a dealer doesn't want to give us that information. So we will replace 0 with null.
upfits_subset.loc[upfits_subset['final_price'] == 0, 'final_price'] = np.nan


# In[13]:


upfits_subset.info()


# 3,105 vehicles will null prices. (33902 - 30797). All variables are the correct type.

# In[20]:


upfits_subset["body_type"].unique()


# We see above the amount of different body types we have. We will be narrowing this analysis down to fewer categories below, specifically the top 5 largest categories and an "other" category containing the remaining body types.

# In[24]:


# Make a copy
upfits_subset = upfits_subset.copy()

# Get the top 5 most frequent categories
top5 = upfits_subset['body_type'].value_counts().nlargest(5).index

# Create a new column with only top 5, everything else is "Other"
upfits_subset.loc[:, 'body_type_top5'] = upfits_subset['body_type'].where(
    upfits_subset['body_type'].isin(top5),
    other='Other'
)


# In[27]:


upfits_subset["body_type_top5"].unique()


# Rather than looking at average amount of Web leads, let's simply consider whether a vehicle received a web lead or not. This will be informative for our analysis.

# In[45]:


upfits_subset["received_lead"] = np.where(
    upfits_subset["web_leads"] >= 1,
    "Received at least 1 Web Lead",
    "No Web Leads"
)


# In[42]:


upfits_subset["received_lead"].unique()


# In[43]:


upfits_subset["received_lead"].value_counts()


# In[44]:


upfits_subset["received_lead"].value_counts(normalize=True)


# Approximately 3/4 (76.2%, or 25,827 vehicles) of all vehicles recieve no web leads.

# In[40]:


upfits_subset.info()


# ## Descriptive Question

# What is the largest category of Body type? How fast do does that body type sell, i.e., what is its Average DTT (days to turn)?

# In[22]:


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

# In[23]:


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

# In[31]:


# Percentages of each group (labeled on the bottom of the bar graph above as well)
100 * upfits_subset.groupby("body_type_top5").size() / upfits_subset.shape[0]


# In[9]:


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

# In[12]:


# Only Rows that are Service Trucks
# service_truck = upfits_subset[upfits_subset["body_type"] == "Service Truck"]


# In[34]:


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


# In[35]:


# Correlation Coefficient
upfits_subset['buyer_engagements'].corr(upfits_subset['web_leads'])


# With a correlation coefficient of 0.28, we have |0.28| < 0.4, implying a weak or no relationship. If there is a relationship, though weak, it would be positive, implying as buyer engagements increases, the number of web leads also increases. Form appears to be non-linear.

# In[36]:


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


# In[56]:


# Correlation Coefficient
avg_bodytype['avg_buyer_engagements'].corr(avg_bodytype['avg_web_leads'])


# I thought that perhaps aggregating over body type would help me understand the relationship between average buyer engagements and average web leads for each body type. The correlation coefficient here is 0.48, and since 0.4 < |.48| < 0.6, we have a moderate-strength positive association between average buyer engagements and average web leads for each body type. This means as average buyer engagements increase, so do average web leads.

# In[38]:


upfits_subset['received_lead'].dtype


# In[105]:


lead_bar1 = alt.Chart(upfits_subset, title="Buyer Engagements vs. Web Leads")

bars1 = (
    alt.Chart(upfits_subset, title="Total Vehicles Receiving at Least 1 Web Lead").mark_bar()
    .encode(
        x=alt.X("received_lead:N", title="Web Leads"),
        y=alt.Y("count():Q", title="Count")
    )
)

labels1 = (
    lead_bar.mark_text(dy=-5, fontSize=12)
    .encode(
        x=alt.X("received_lead:N"),
        y="count():Q",
        text=alt.Text("count(received_lead):Q")   # count labels
    )
)

(bars1 + labels1)


# I noticed that a lot of vehicles get no web leads at all. On each bar is the count of vehicles that received either no web leads or at least 1 web lead. 25,827 vehicles received no web leads, and 8,075 received at least 1. 

# In[93]:


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

# In[104]:


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

# Will a specific vehicle receive any web leads (interested buyers filling out a form for more information on the vehicles page)?

# In[6]:


set_config(transform_output="pandas")


# In[4]:


# Set a random seed
np.random.seed(1)


# Setting a random seed allows randomly produced numbers to be randomized but our result to be reproducible to anyone who uses the same random seed. 

# Recall how the amount of vehicles that received any web leads are distributed:

# In[60]:


upfits_subset["received_lead"].value_counts()


# In[61]:


upfits_subset["received_lead"].value_counts(normalize=True)


# Consider the possible associaton between Buyer engagements (clicks on a vehicle's webpage) and DTT (Days to turn/how long the vehicle takes to sell)

# In[63]:


buyer_searches = alt.Chart(upfits_subset).mark_circle().encode(
    x=alt.X("buyer_engagements").title("Buyer Engagements"),
    y=alt.Y("searches_body_type").title("Body Type Searches"),
    color=alt.Color("received_lead").title("received_lead")
)
buyer_searches


# ### K-Nearest Neighbors

# In[65]:


# Building Training Set
upfits_train = upfits_subset[["received_lead", "buyer_engagements", "searches_body_type"]]
upfits_train


# Creating our model:

# In[66]:


knn = KNeighborsClassifier(n_neighbors=5)
knn


# In[68]:


# fit our training set
knn.fit(X=upfits_train[["buyer_engagements", "searches_body_type"]], 
        y=upfits_train["received_lead"]);


# In[83]:


# make a prediction about a new observation
new_obs = pd.DataFrame({"buyer_engagements": [1000], "searches_body_type": [100000]})
knn.predict(new_obs)


# In[81]:


# make a prediction about a new observation
new_obs2 = pd.DataFrame({"buyer_engagements": [250], "searches_body_type": [100000]})
knn.predict(new_obs2)


# For example, our model predicts that a vehicle with 1000 clicks on its page and 100,000 searches by the vehicles body type will receive at least 1 web lead, while a vehicle with 250 clicks and the same amount of searches will not receive any web leads.

# In[ ]:




