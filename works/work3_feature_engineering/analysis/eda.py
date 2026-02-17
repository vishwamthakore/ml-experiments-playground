#%%

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True, as_frame=True)
target_name = y.name if y.name else "target"
X[target_name] = y
df = X
df.head()

#%%

# Target Column Bar Plot

import plotly.express as px

target_counts = df[target_name].value_counts().reset_index()
target_counts.columns = ["class", "count"]

fig = px.bar(
    target_counts,
    x="class",
    y="count",
    title="Target Class Distribution"
)

fig.show()

# %%

# Target Column Distribution

col_name = df.columns[0]

fig = px.histogram(
    data_frame=df,
    x=col_name,
    nbins=30,
    title=f"Distribution of {col_name}",
    facet_row_spacing=1,
)

fig.show()

# %%

correlations = df.corr()
correlations

#%%

fig = px.imshow(
    correlations,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Feature Correlation Matrix"
)

fig.show()

# %%
