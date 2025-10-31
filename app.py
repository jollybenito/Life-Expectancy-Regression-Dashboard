# Plotly's Figure Friday challenge. See more info here: https://community.plotly.com/t/figure-friday-2024-week-32/86401
import dash
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback, Patch
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import custom_functions as cf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


df = pd.read_csv('https://raw.githubusercontent.com/jollybenito/Life-Expectancy-Regression-Dashboard/refs/heads/main/train.csv', index_col=0)
#df = pd.read_csv('train.csv', index_col=0)

nation_dropdown = html.Div(
    [
        dbc.Label("Select Filter", html_for="date-checklist"),
        dbc.RadioItems(
            options=["No cleaning", "Inflation cleaning", "All cleaning"],
            value="No cleaning",
            id="Nation-dropdown",
        ),
    ],
    className="mb-4",
)


women_parliament_dropdown = html.Div(
    [
        dbc.Label("Select Filter 2", html_for="date-checklist"),
        dbc.RadioItems(
            options=["Coconut", "Chocolate"],
            value="Coconut",
            id="Women-parliament-dropdown",
        ),
    ], className="mb-4",
)

control_panel = dbc.Card(
    dbc.CardBody(
        [women_parliament_dropdown, nation_dropdown ],
        className="bg-light",
    ),
    className="mb-4"
)

heading = html.H1("Life expectancy Regression",className="bg-secondary text-white p-2 mb-4")

about_card = dcc.Markdown(
    """
    ASAP Life expectancy regressor model."
    """)

data_card = dcc.Markdown(
    """
    HackerRank Data Scientist Hiring Test: Predict Life Expectancy
    I will add the link on the next update but you can find them here on Github!

    Ipsem lorum ipse lorum
    """
)

info = dbc.Accordion([
    dbc.AccordionItem(about_card, title="About This Dashboard", ),
    dbc.AccordionItem(data_card, title="About The Data Source")
],  start_collapsed=True)

missing_data_card = dcc.Markdown(
    """
    There is a high amount of missing data in our dataset.
    However there is a key thing to analyze here. That is the fact that the inflation columns contain most of the missing data.
    And upon deeper analysis that the the missing data between these 3 variables is mutually exclusive.
    Therefore by using financial formulas to get the inflation into either yearly monthly or weekly, a singular variable without missing data can be created.
    The result of this can be analyzed in the graph below.
    After that we will move on to use an imputer to deal with the remaining missing data.
    Which will lead us to a zero missing data
    """
)

barchart_target_card = dcc.Markdown(
    """
    After plotting the histogram we try to think of possible distributions that fit this behavior
    "Skewed to the left"
    """
)

linearity_target_card = dcc.Markdown(
    """
    We also try to determine wether there are linear relations from just our simple variables and the target variable.
    """
)


app.layout = dbc.Container(
    [
        dcc.Store(id="store-selected", data={}),
        heading,
        dbc.Row([
            dbc.Col([control_panel, info], md=3),
            dbc.Col( html.Div(id="missing-data-selector", className="mt-4")),          
            dbc.Col(
                [
                    dcc.Markdown(id="title"),
                    dbc.Col(html.Div(id="paygap-card")),
                    dbc.Col(dbc.Card(
                        dbc.CardBody(
                            [missing_data_card],
                            className="bg-light",
                        ),className="mb-4")
                    ),
                    dbc.Col(html.Div(id="after-paygap-card")),
                ],  md=9
            ),
        ]),
        dbc.Row(dbc.Col( html.Div(id="bar-chart-card", className="mt-4"))), 
        dbc.Col(dbc.Card(dbc.CardBody([barchart_target_card],
                                      className="bg-light"),
                         className="mb-4")),        
        dbc.Row(dbc.Col( html.Div(id="linear-relations" ))),
        dbc.Col(dbc.Card(dbc.CardBody([linearity_target_card],
                                      className="bg-light"),
                                    className="mb-4")),
        dbc.Row(dbc.Col( html.Div(id="feature-importances"))),
        dbc.Row(dbc.Col( html.Div(id="gridcv-model-comparison" )))
    ],
    fluid=True,
)


@callback(
    Output("store-selected", "data"),
    Input("Nation-dropdown", "value"),
    Input("Women-parliament-dropdown", "value"),
)
def pin_selected_report(company, yr):
    records = df.to_dict("records")
    return records


@callback(
    Output("bar-chart-card", "children"),
    Input("store-selected", "data"),
    Input("Nation-dropdown", "value")
)
def make_bar_chart(data, company):
    df = pd.DataFrame(data)
    if (company=="Inflation cleaning"):
        df_data = cf.inflation_cleaning(df)
    elif (company=="All cleaning"):  
        df_data = cf.preprocess_missing_data(df)
    else:
        df_data = df
    # Separate the data for male and female
    plot = go.Figure(data=[go.Histogram(
        x = df_data["life_expectancy"], nbinsx = 12,
        histnorm='probability density')])            

    return dbc.Card([
        dbc.CardHeader(html.H2("Histogram of life expectancy"), className="text-center"),
        dcc.Graph(figure=plot, style={"height":350}, config={'displayModeBar': False})
    ])


@callback(
    Output("title", "children"),
    Input("store-selected", "data")
)
def make_title(data):
    data=data[0]
    title = f"""
    ##  Life expectancy Data Science Walkthrough 
    """
    return title



@callback(
    Output("missing-data-selector", "children"),
    Input("store-selected", "data"),
    Input("Nation-dropdown", "value")
)
def missing_data_selector(data, company):
    df = pd.DataFrame(data)
    # "3D" effect parameters
    dx = 0.5  # horizontal depth
    dy = 1.1     # vertical projection (for shadow offset)
    bar_width = 0.9
    bar_spacing = 1.2  # increase spacing between bars    
    if (company=="All cleaning"):  
        df_data = cf.preprocess_missing_data(df)
        df_missing = df_data.isnull().sum().sort_values(ascending=False).copy()
        # data variables
        categories = df_missing.index
        values = df_missing
        colors = ['#d13c3c'] * len(categories)
        # Compute x positions with extra spacing
        x_positions = np.arange(len(categories)) * bar_spacing        
        fig2 = go.Figure()
        # Just annotations (x-axis labels)
        for x, cat in zip(x_positions, categories):
            fig2.add_annotation(
                x=x,
                y=0,
                text=cat[:20],
                showarrow=False,
                font=dict(size=12, color='darkred'),
                textangle=85,
                xshift=25,
                yshift=-55
            )

        # Style / layout
        fig2.update_xaxes(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[x_positions[0] - 1, x_positions[-1] + 1.5]
        )
        fig2.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=350,
            width=600,
            margin=dict(l=50, r=50, t=80, b=80)
        )        
    else:
        if (company=="Inflation cleaning"):
            df_data = cf.inflation_cleaning(df)
        else:
            df_data = df
        df_missing = df_data.isnull().sum().sort_values(ascending=False).copy()

        # data variables
        categories = df_missing.index
        values = df_missing
        colors = ['#d13c3c'] * len(categories)
        # Compute x positions with extra spacing
        x_positions = np.arange(len(categories)) * bar_spacing

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=x_positions,
            y=values,
            width=bar_width,
            marker_color=colors,
            hovertemplate="<b>%Missing data in variable</b><br>Value: %{y}<extra></extra>",
            name='',
            showlegend=False
        ))        

        # Draw bars with 3D illusion
        for x, cat, val, color in zip(x_positions, categories, values, colors):
            # Front face
            fig2.add_shape(
                type='rect',
                x0=x - bar_width/2, x1=x + bar_width/2,
                y0=0, y1=val,
                fillcolor=color,
                line=dict(width=0)
            )
            # Right face
            fig2.add_shape(
                type='path',
                path=f'M {x + bar_width/2},{val} '
                    f'L {x + bar_width/2 + dx},{val + dy} '
                    f'L {x + bar_width/2 + dx},{dy} '
                    f'L {x + bar_width/2},{0} Z',
                fillcolor='rgba(0,0,0,0.9)',
                line=dict(width=0)
            )
            # Top face
            fig2.add_shape(
                type='path',
                path=f'M {x - bar_width/2},{val} '
                    f'L {x + bar_width/2},{val} '
                    f'L {x + bar_width/2 + dx},{val + dy} '
                    f'L {x - bar_width/2 + dx},{val + dy} Z',
                fillcolor='rgba(0,0,0,0.9)',
                line=dict(width=0)
            )

        # ADD a loop to create ANNOTATIONS for each label
        for x, cat in zip(x_positions, categories):
            fig2.add_annotation(
                x=x,
                y=0,  # Same vertical position as before
                text=cat,
                showarrow=False,
                font=dict(size=12, color='darkred'),
                textangle=85,  # ✅ This is the correct property for rotation
                xshift=-5,      # Horizontal shift (optional, can help centering)
                yshift= -len(cat)*3.5,
                xanchor='left', # ✅ anchor the text from its left edge
                align='left'    # ✅ align multi-line text (if any) to the left                
            )

        # Adjust axes
        fig2.update_xaxes(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[x_positions[0] - 1, x_positions[-1] + 1.5]
        )
        fig2.update_yaxes(
            range=[0, max(values) * 1.1],
            title=None,
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=14, color='darkred')
        )

        # Layout
        fig2.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            width=900,
            margin=dict(l=50, r=50, t=80, b=120),
        )
    return dbc.Card([
        dbc.CardHeader(html.H2("Data source selector"), className="text-center"),
        dcc.Graph(figure=fig2, style={"height":450})
    ])




@callback(
    Output("linear-relations", "children"),
    Input("store-selected", "data"),
    Input("Nation-dropdown", "value")
)
def linear_relationships(data, company):
    df = pd.DataFrame(data)
    if (company=="Inflation cleaning"):
        df_data = cf.inflation_cleaning(df)
    elif (company=="All cleaning"):  
        df_data = cf.preprocess_missing_data(df)
    else:
        df_data = df
    # List of features to plot against Life expectancy
    features = df_data.columns[df_data.dtypes == float]
    df_numerical = df_data[features]

    # Create subplot grid (1 column, N rows)
    fig = make_subplots(
        rows=len(features),
        cols=2,
        shared_xaxes=False,
        subplot_titles=[f"{feature} vs Life Expectancy" for feature in features]
    )

    # Add each scatter plot as a trace
    k = 0
    for i, feature in enumerate(features, start=1):
        if (i % 2 == 0):
            j=2
        else:
            j=1          
            k = k + 1
        fig.add_trace(
            go.Scatter(
                x=df_numerical[feature],
                y=df_numerical['life_expectancy'],
                mode='markers',
                marker=dict(color="#ea5454", size=6, opacity=0.7),
                name=""
            ),
            row=k, col=j  
        )

    # Layout formatting
    fig.update_layout(
        height=150 * len(features),
        width=1200,
        showlegend=False,
        template='plotly_white'
    )

    return dbc.Card([
        dbc.CardHeader(html.H2("Linear relationships analysis"), className="text-center"),
        dcc.Graph(figure=fig, style={"height":90 * len(features)})
    ])

@callback(
    Output("gridcv-model-comparison", "children"),
    Input("store-selected", "data")
)
def model_comparison(data):
    df = pd.DataFrame(data)
    # List of features to plot against Life expectancy
    x_cols = [c for c in df.columns if c != 'life_expectancy']
    X = df[x_cols]
    y = df['life_expectancy'].astype('float64')
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
    # Cleaning
    X_train = cf.preprocess_missing_data(X_train)
    X_test = cf.preprocess_missing_data(X_test)
    just_scores_df, best_grid_df = cf.model_comparer(X_train, y_train)
    # Create figure
    fig = go.Figure()
    # Loop df columns and plot columns to the figure
    for i in just_scores_df['model'].unique():
        df_temp = just_scores_df[just_scores_df['model'] == i]
        fig.add_trace(go.Scatter(x=df_temp['CrossValidation'],
                        y=df_temp['RMSE Score'], 
                        name=i,
                        mode='lines+markers',
                        ))
    return dbc.Card([
        dbc.CardHeader(html.H2("5 Best models from GridSearchCV"), className="text-center"),
        dcc.Graph(figure=fig, style={"height":90 * len(just_scores_df['model'].unique())})
    ])

@callback(
    Output("feature-importances", "children"),
    Input("store-selected", "data")
)
def feature_importances(data):
    df = pd.DataFrame(data)
    # List of features to plot against Life expectancy
    x_cols = [c for c in df.columns if c != 'life_expectancy']
    X = df[x_cols]
    y = df['life_expectancy'].astype('float64')
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
    X_train = cf.preprocess_missing_data(X_train)
    X_test = cf.preprocess_missing_data(X_test)

    just_scores_df, best_grid_df = cf.model_comparer(X_train, y_train)
    best_params = best_grid_df.sort_values("mean_test_score", ascending=False).iloc[0,1]
    best_model_name =  list(best_grid_df.sort_values("mean_test_score", ascending=False).index)[0]    
    if best_model_name == "rf":
        model = RandomForestRegressor(**best_params)
        model.fit(X_train, y_train)
    elif best_model_name == "xgb":
        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        importances = importances.sort_values(ascending=True) 
    fig = go.Figure(go.Bar(
        x=importances.values,
        y=importances.index,
        orientation='h',
        marker=dict(color="#e52d2d")
    ))
    return dbc.Card([
        dbc.CardHeader(html.H2("Feature importances"), className="text-center"),
        dcc.Graph(figure=fig, style={"height":30 * len(X_test.columns)})
    ])



if __name__ == "__main__":
app.run_server(host="0.0.0.0", port=8050, debug=True)