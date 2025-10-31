import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer

def inflation_cleaning(df):
    # Get number of rows
    num_rows = len(df)
    # First we will identify the columns that have missing values
    # And we will clasify it by their sizes in 2 tiers
    # Tier 1 we can be sure to use imputing
    # Tier we will analyze to determine if there is something to rescue
    # or if its better to scrape
    Columns_sorted_by_nulls = df.isnull().sum().sort_values(ascending=False)
    Just_Columns_With_Nulls = Columns_sorted_by_nulls[Columns_sorted_by_nulls>0]
    Columns_with_Alot_Nulls = Just_Columns_With_Nulls[Just_Columns_With_Nulls >= num_rows/3].index
    # But on a different time scale. So we'll join the data into a singular annual time frame
    # Note we previously checked the non-missing data DOES NOT intersect
    # The nan variables will be set to 0 due to our transformations nature, but we will
    # restore them later
    df['annualized_inflation'] = 0
    # These formulas are from financial math formulas
    df['annualized_inflation'] = df['inflation_annual'].fillna(0) + pd.Series((1 + df["inflation_weekly"].fillna(0))**52 - 1)
    df['annualized_inflation'] = df['annualized_inflation'] + pd.Series((1 + df["inflation_monthly"].fillna(0))**12 - 1)
    # We restore the nan values back to nan
    df['annualized_inflation'] = df['annualized_inflation'].replace(0, np.nan)

    # We have now created a new variable that merges data from the 3 other cases
    # We will proceed from this point on with just this one and drop the others
    df1 = df.drop(Columns_with_Alot_Nulls, axis=1)

    return df1

def preprocess_missing_data(df):    
    # Get number of rows
    num_rows = len(df)
    # First we will identify the columns that have missing values
    # And we will clasify it by their sizes in 2 tiers
    # Tier 1 we can be sure to use imputing
    # Tier we will analyze to determine if there is something to rescue
    # or if its better to scrape
    Columns_sorted_by_nulls = df.isnull().sum().sort_values(ascending=False)
    Just_Columns_With_Nulls = Columns_sorted_by_nulls[Columns_sorted_by_nulls>0]
    Columns_with_Alot_Nulls = Just_Columns_With_Nulls[Just_Columns_With_Nulls >= num_rows/3].index
    # We see that these 3 variables are describing similar things
    # But on a different time scale. So we'll join the data into a singular annual time frame
    # Note we previously checked the non-missing data DOES NOT intersect
    # The nan variables will be set to 0 due to our transformations nature, but we will
    # restore them later
    df['annualized_inflation'] = 0
    print(df.columns)
    # These formulas are from financial math formulas
    df['annualized_inflation'] = df['inflation_annual'].fillna(0) + pd.Series((1 + df["inflation_weekly"].fillna(0))**52 - 1)
    df['annualized_inflation'] = df['annualized_inflation'] + pd.Series((1 + df["inflation_monthly"].fillna(0))**12 - 1)
    # We restore the nan values back to nan
    df['annualized_inflation'] = df['annualized_inflation'].replace(0, np.nan)

    # We have now created a new variable that merges data from the 3 other cases
    # We will proceed from this point on with just this one and drop the others
    df.drop(Columns_with_Alot_Nulls, axis=1, inplace=True)
    # That reduced our missing data related to these variables
    # For the remainder variables and our new variable, we will use a MICE method
    # But first we need to transform the objects into numerical or ranked data if possible
    df[['surface_area', 'agricultural_land', 'forest_area', 'armed_forces_total', 
    'urban_pop_major_cities', 'urban_pop_minor_cities',
    'secure_internet_servers_total', 'annualized_inflation']] = df[['surface_area', 'agricultural_land', 'forest_area', 'armed_forces_total', 
    'urban_pop_major_cities', 'urban_pop_minor_cities', 'secure_internet_servers_total',
    'annualized_inflation']].astype('float64')
    # We notice that national_income and improved_sanitation can be assigned ranks and they have the same general
    # string patterns so we will encode them together
    df.loc[df['national_income'].isin(['medium low']),'national_income'] = 3
    df.loc[df['national_income'].isin(['medium high']),'national_income'] = 4
    df.loc[df['national_income'].isin(['very high']),'national_income'] = 6
    df.loc[df['national_income'].isin(['high']),'national_income'] = 5
    df.loc[df['national_income'].isin(['very low']),'national_income'] = 1
    df.loc[df['national_income'].isin(['low']),'national_income'] = 2
    df['national_income'] = pd.to_numeric(df['national_income'], errors='coerce')
    df.loc[df['improved_sanitation'].isin(['medium access']),'improved_sanitation'] = 3
    df.loc[df['improved_sanitation'].isin(['very high access']),'improved_sanitation'] = 5
    df.loc[df['improved_sanitation'].isin(['high access']),'improved_sanitation'] = 4
    df.loc[df['improved_sanitation'].isin(['very low access']),'improved_sanitation'] = 1
    df.loc[df['improved_sanitation'].isin(['low access']),'improved_sanitation'] = 2
    df['improved_sanitation'] = pd.to_numeric(df['improved_sanitation'], errors='coerce')
    # For 'internet_users' we see we can create a float by splitting the string and dividing by 1000 or 100
    # respectively
    df.loc[:,'internet_users'] = df.loc[:,'internet_users'].replace('unknown', '-1 per 100 people')
    df.loc[:,'internet_users_0'] = df['internet_users'].str.split(" per ", expand=True)[0]
    df.loc[:,'internet_users_1']  = df['internet_users'].str.split(" per ", expand=True)[1].str.split(" people", expand=True)[0]
    # Fill nas with 100 to avoid errors in division later
    df.loc[:,'internet_users_1'] = df.loc[:,'internet_users_1'].fillna(100)
    df['internet_users_numeric'] = df['internet_users_0'].astype(int)/df['internet_users_1'].astype(int)
    df.loc[:,'internet_users_numeric'] = df.loc[:,'internet_users_numeric'].replace(-0.01, np.nan)
    df.drop(['internet_users','internet_users_0','internet_users_1'], axis=1, inplace=True) 
    # Convert to categorical type and then get the codes
    df.loc[:,'women_parliament_seats_rate'] = df['women_parliament_seats_rate'].replace('unknown', np.nan)
    df.loc[:,'women_parliament_seats_rate_numeric'] = df['women_parliament_seats_rate'].astype('category').cat.codes
    df.loc[:,'women_parliament_seats_rate_numeric'] = df['women_parliament_seats_rate_numeric'].replace(-1, np.nan)
    df.drop(['women_parliament_seats_rate'], axis=1, inplace=True)
    #######
    df.loc[:,'mobile_subscriptions'] = df['mobile_subscriptions'].replace('unknown', np.nan)
    df.loc[:,'mobile_subscriptions_numeric'] = df['mobile_subscriptions'].astype('category').cat.codes
    df.loc[:,'mobile_subscriptions_numeric'] = df['mobile_subscriptions_numeric'].replace(-1, np.nan)
    df.drop(['mobile_subscriptions'], axis=1, inplace=True)
    # Now we have finished with our data cleaning
    # Now that we only have numerical values we can use an imputer for filling in the missing values
    # We just have to make sure that we separate the processes for categorical values and
    # real values.
    # For the categorical values we will use a mode replacement
    df_simple = df.copy()
    categorical_data = ['women_parliament_seats_rate_numeric', 'mobile_subscriptions_numeric']
    simple_imputer = SimpleImputer(strategy='most_frequent')
    df_simple[categorical_data] = simple_imputer.fit_transform(df_simple[categorical_data])
    #####
    df_mice = df.drop(['women_parliament_seats_rate_numeric', 'mobile_subscriptions_numeric'], axis=1)
    df_mice_columns = df_mice.columns
    mice_imputer = IterativeImputer(random_state=0)
    df_mice = mice_imputer.fit_transform(df_mice)
    df_temp1 = pd.DataFrame(df_mice, columns=df_mice_columns)
    df_imputeddata = pd.concat([df_temp1, df_simple[categorical_data].reset_index(drop=True)], axis=1)
    return df_imputeddata

# Compare models
def model_comparer(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
    rf_params = {"n_estimators": [50, 100], 
                "max_depth": [5, 10],
                "max_features": ["sqrt", "log2"]}
    xgb_params = {"n_estimators": [50, 100], 
                "max_depth": [3, 5],
                "learning_rate": [0.1, 0.01]}

    # GridSearch for RF and XGB
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring=scorer)
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring=scorer)
    rf_grid.fit(X_train, y_train)
    xgb_grid.fit(X_train, y_train)

    # Save scores into dataframes
    all_results = []
    best_grid_model_results = []
    grids = {"rf": rf_grid, "xgb": xgb_grid }
    for name, grid in grids.items():  # 
        df = pd.DataFrame(grid.cv_results_)
        df["model"] = name + "-" + df["param_n_estimators"].astype(str) + "-" + df["param_max_depth"].astype(str)
        if name == "rf":
            df["model"] = df["model"] + "-" + df["param_max_features"].astype(str)
        elif name == "xgb":
            df["model"] = df["model"] + "-" + df["param_learning_rate"].astype(str)
        all_results.append(df)
        best_grid_model = pd.Series(pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")[["mean_test_score", 
                                        "params"]].iloc[0,:])
        best_grid_model_results.append(best_grid_model)
    
    best_grid_df = pd.concat(best_grid_model_results, axis=1)
    best_grid_df.columns = list(grids.keys())
    best_grid_df = best_grid_df.T
    best_grid_df = best_grid_df.sort_values("mean_test_score", ascending=False)
    #
    all_results_df = pd.concat(all_results, ignore_index=True)
    # Now you can analyze across models
    all_results_df.sort_values(by="mean_test_score", ascending=False, inplace=True)
    just_scores_df = all_results_df[["model", "split0_test_score", 
                    "split1_test_score","split2_test_score",
                    "split3_test_score","split4_test_score"]].head(5).copy()
    just_scores_df.columns = ["model", "1", "2", "3", "4", "5"]
    just_scores_df = just_scores_df.melt(id_vars=["model"],
                        var_name="CrossValidation",
                        value_name="RMSE Score")

    return just_scores_df, best_grid_df