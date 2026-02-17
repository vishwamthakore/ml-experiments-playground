import plotly.express as px


def get_bar_plot(df, column_name):
    target_counts = df[column_name].value_counts().reset_index()
    target_counts.columns = ["class", "count"]
    fig = px.bar(
        target_counts,
        x="class",
        y="count",
        color="class",
        title="Target Class Distribution",
    )
    return fig


def get_histogram(df, column_name, nbins=30):
    fig = px.histogram(
        df,
        x=column_name,
        marginal="box",  # VERY useful
        nbins=30,
        title=f"Distribution of {column_name}",
    )
    return fig

def get_correlation_heatmap(correlation_matrix):
    fig = px.imshow(
        img=correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix"
    )
    return fig