#Import all required libraries
import pandas as pd
import matplotlib
import math
import os
import seaborn as sbn
from altair import Chart, X, Y, Color, Scale
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import xlrd

SDG2022 = pd.read_excel(r'C:\Users\cooperd\PycharmProjects\GSTR410\SDR-2022-database.xlsx', sheet_name="SDR2022 Data")
Backdate = pd.read_excel(r'C:\Users\cooperd\PycharmProjects\GSTR410\SDR-2022-database.xlsx', sheet_name="Backdated SDG Index")
#new = SDG2022[['Country Code ISO3', 'Country', 'Regions used for the SDR', '2022 SDG Index Score', 'Population in 2021', 'Goal 7 Score', 'Goal 13 Score', 'Goal 7 Regional Score', 'Goal 13 Regional Score']].copy()

# Drop any country with a non recorded value
SDG2022.rename(columns={'Country Code ISO3':'iso_code'}, inplace=True)
SDG2022.rename(columns={'2022 SDG Index Score':'SDG Index Score'}, inplace=True)
SDG2022.rename(columns={'Regions used for the SDR':'Region'}, inplace=True)
Backdate.rename(columns={'Country Code ISO3':'iso_code'}, inplace=True)
SDG2022.columns = SDG2022.columns.str.strip()
Backdate.columns = Backdate.columns.str.strip()
SDG2022cols = SDG2022.columns.to_list()
Backdatecols = Backdate.columns.to_list()
print(SDG2022cols)
print(Backdatecols)

# Data set and code used in this cell was recovered from a project
# I completed for CSC 328 (Data Analytics) as a group alongside Bryar Frank and Anthony Wafula
covid = pd.read_csv(r'C:\Users\cooperd\PycharmProjects\GSTR410\owid-covid-data (1).csv')


covid['date'] = pd.to_datetime(covid['date'])

#covid = covid.drop_duplicates('location', keep='last') # drops every row except most recent listing

def Covidiso(iso_code, covid):
    """
    This function takes the covid dataframe and deletes all the useless columns and rows of each country we do not wish to see.
    It groups the data based on year and returns the dataframe only containing the most recent entry for each year
    for the country we wish to see.
    :param iso_code: This variable represents the iso code of the country we wish to use.
    It is a user generated variable created through input.
    :param covid: Covid is the covid dataframe generated before the function and main.
    It must be passed to be recognized within the function.
    :return: Edited dataframe, will be of size (3 rows, 4 columns)
    """

    rows_to_drop = ['World', 'International', 'Hong Kong']

    cols_to_drop = ['new_cases', 'new_cases_smoothed', 'new_deaths', 'new_deaths_smoothed',
                    'new_cases_per_million', 'new_cases_smoothed_per_million', 'new_deaths_per_million',
                    'new_deaths_smoothed_per_million', 'new_tests', 'new_tests_per_thousand',
                    'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_units',
                    'stringency_index', 'aged_65_older', 'aged_70_older', 'cardiovasc_death_rate',
                    'diabetes_prevalence', 'female_smokers', 'male_smokers', 'country', 'pop2020', 'positive_rate',
                    'reproduction_rate', 'icu_patients', 'gdp_per_capita', 'extreme_poverty', 'handwashing_facilities',
                    'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index',
                    'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative', 'excess_mortality',
                    'excess_mortality_cumulative_per_million', 'continent', 'location',
                    'icu_patients_per_million', 'hosp_patients', 'hosp_patients_per_million', 'weekly_icu_admissions',
                    'weekly_icu_admissions_per_million', 'total_vaccinations_per_hundred',
                    'people_vaccinated_per_hundred',
                    'people_fully_vaccinated_per_hundred', 'new_people_vaccinated_smoothed',
                    'population_density	median_age', 'total_cases_per_million',
                    'total_deaths_per_million	weekly_hosp_admissions',
                    'weekly_hosp_admissions_per_million', 'total_tests	total_tests_per_thousand', 'tests_per_case',
                    'people_fully_vaccinated', 'total_boosters', 'new_vaccinations_smoothed',
                    'total_boosters_per_hundred', 'new_vaccinations_smoothed_per_million', 'population_density',
                    'median_age',
                    'total_deaths_per_million', 'weekly_hosp_admissions', 'total_tests	total_tests_per_thousand',
                    'total_vaccinations	people_vaccinated', 'total_boosters', 'new_vaccinations',
                    'new_people_vaccinated_smoothed_per_hundred', 'population',
                    'total_tests', 'total_tests_per_thousand', 'total_vaccinations', 'people_vaccinated']

    covid = covid[covid.iso_code == iso_code]
    covid.date = pd.DatetimeIndex(covid.date).year
    covid = covid.drop_duplicates('date', keep='last')
    covid.date = covid.date.astype(float)
    #print(covid.dtypes)

    for location in rows_to_drop:                           # this is where we dropped unneccesary rows stated from before
      covid = covid[covid.location != '{}'.format(location)]

    for col in cols_to_drop:                                # drops every column named in the list above.
      if col in covid.columns:
        covid = covid.drop(col, axis=1)

    print(covid.shape)
    covid.rename(columns={'date':'Year'}, inplace=True)
    covid.head(10)
    return covid

#covid = Covidiso('IRN', covid)

def Countryset(iso_code, df, GDP=False):
    """
    This function adds the 2022 SDG data to the existing backdated data for the country
    specified through iso code variable.
    :param iso_code: User specified iso code corresponding with country they wish to use.
    :param df: Dataframe holding backdated data.
    :return: Edited dataframe with new row for year 2022 and cols3 which is simply list of columns in dataframe.
    """
    if GDP != False:
        GDP = pd.read_excel(r"C:\Users\cooperd\PycharmProjects\GSTR410\API_NY.GDP.MKTP.CD_DS2_en_excel_v2_4251142.xls", sheet_name="Data")
        GDP.rename(columns={'Country Code': 'iso_code'}, inplace=True)
        GDP = GDP[GDP.iso_code == iso_code]
        listcols = GDP.columns.tolist()
        listvals = GDP.values.tolist()
        listvals = listvals[0]
        listvals.pop(0)
        listcols.pop(0)
        #print(listvals)
        new_df = pd.DataFrame(columns=["GDP"], index=listcols, data=listvals, dtype=float)
        new_df["Year"] = new_df.index
        new_df = new_df.astype(float)
        print(new_df.GDP)
        # transposed_GDP = GDP.transpose(copy=True)
        # print(transposed_GDP.columns)

    Country = df[df.iso_code == iso_code]
    Country.columns

    Iran2022 = SDG2022[SDG2022.iso_code == iso_code]
    next = pd.concat([Country, Iran2022], axis=0)
    next['Year'] = next['Year'].fillna(2022.0)
    next['Population'] = next['Population'].fillna(85028760.0)
    next.dropna(axis=1, inplace=True)
    #next.tail(5)
    #print(next.dtypes)
    next = next.merge(new_df, on="Year", how="outer")
    #print(next.columns)
    cols3 = next.columns
    return next, cols3

#next, cols3 = Countryset('IRN', Backdate)

def Corrgoals(df, cols, iso):
    """
    Create a correlation matrix between all columns in df
    :param df: Dataframe holding country and SDG data
    :param cols: list of columns in df (in our case cols3)
    :return: N/A
    """
    plt.figure(figsize=[16,8])
    sbn.heatmap(df[cols].corr(), annot=True, cmap = 'viridis')
    #plt.show()
    filepath = r"C:\Users\cooperd\PycharmProjects\GSTR410\FigImages\ " + str(iso) + "corr"
    #print(filepath)
    plt.savefig(filepath)

#Corrgoals(next, cols3)

def Plotgoals(df, iso):
    """
    Plot a line chart, one line for each goal in data with year on X axis.
    :param df: Dataframe holding country and SDG data
    :return: N/A
    """
    goal = 1
    colo = ["", "xkcd:purple", "xkcd:green", "xkcd:blue", "xkcd:pink", "xkcd:brown", "xkcd:red", "xkcd:light blue", "xkcd:teal", "xkcd:orange", "xkcd:light green", "xkcd:magenta",
            "xkcd:yellow", "xkcd:grey", "xkcd:lime green", "xkcd:light purple", "xkcd:dark green", "xkcd:dark gold"]


    plt.figure(figsize=[16,8])
    for i in df.columns:
      if 'Goal' in i:
        current = 'Goal ' + str(goal)
        plt.plot(df['Year'], df[i], label=current, color=colo[goal])
        goal += 1


    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 20,
            }

    plt.xlabel("Years", fontdict=font)
    plt.ylabel("Goal Score", fontdict=font)
    plt.legend()
    filepath = r"C:\Users\cooperd\PycharmProjects\GSTR410\FigImages\ " + str(iso) + "goals"
    plt.savefig(filepath)
    #plt.show()

#Plotgoals(next)
#next.describe()

def Plotindependent(df, iso):
    """
    Plot the goal scores for each year in their own independent subplot.
    :param df: Dataframe holding country and SDG data
    :return: N/A
    """
    fig, axs = plt.subplots(3, 5, figsize=(8, 5), constrained_layout=True,
                            sharex=True, sharey=True)
    column = []
    goals = []
    for i in df.columns:
      if 'Goal' in i:
        goals.append(i)

    for nn, ax in enumerate(axs.flat):
        ax.set_xlim(2000.0, 2022.0)
        column = goals[nn]
        column_rec_name = column.replace('\n', '_').replace(' ', '_')

        line, = ax.plot('Year', column, data=df, lw=2.5)
        ax.set_title(column, fontsize='small', loc='left')
        ax.set_ylim([0, 100])
        ax.grid()

    # fig.supxlabel('Year')
    # fig.supylabel('Goal Score')
    filepath = r"C:\Users\cooperd\PycharmProjects\GSTR410\FigImages\ " + str(iso) + "independent"
    # print(filepath)
    plt.savefig(filepath)
    #plt.show()


#Plotindependent(next)
#print(covid)
def Plotcov(next, covid, iso):
    """
    Plot the same graph as Plotgoals with the addition of the covid data.
    :param next: Dataframe containing country and SDG data.
    :param covid: Dataframe containing covid data for country.
    :return: N/A
    """
    next = next.merge(covid, on='Year', how='outer')
    next.head(23)

    goal = 1
    colo = ["", "xkcd:purple", "xkcd:green", "xkcd:blue", "xkcd:pink", "xkcd:brown", "xkcd:red", "xkcd:light blue",
            "xkcd:teal", "xkcd:orange", "xkcd:light green", "xkcd:magenta",
            "xkcd:yellow", "xkcd:grey", "xkcd:lime green", "xkcd:light purple", "xkcd:dark green", "xkcd:dark gold"]

    plt.figure(figsize=[16, 8])
    next['total_cases'] = next['total_cases'].fillna(0)

    next['total_cases'] = (next['total_cases'] - next['total_cases'].min()) / (
                next['total_cases'].max() - next['total_cases'].min()) * 100
    print(next['total_cases'])
    # next['total_cases'] = (next.total_cases–mincases) / (maxcases–mincases) * 100
    plt.plot(next['Year'], next['total_cases'], label='Cases', color="xkcd:black")

    for i in next.columns:
        if 'Goal' in i:
            current = 'Goal ' + str(goal)
            plt.plot(next['Year'], next[i], label=current, color=colo[goal])
            goal += 1
            # plt.plot(next['Year'], next['total_cases'], label="Cases", color="xkcd:black")

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 20,
            }
    plt.xticks(np.arange(min(next["Year"]), max(next["Year"]) + 1, 1.0))
    plt.xlabel("Years", fontdict=font)
    plt.ylabel("Goal Score", fontdict=font)
    plt.grid()
    plt.legend()
    filepath = r"C:\Users\cooperd\PycharmProjects\GSTR410\FigImages\ " + str(iso) + "covid"
    # print(filepath)
    plt.savefig(filepath)
    #plt.show()

def main(cov, Backdate):
    iso = input('Please enter the 3 letter iso code of your country')
    iso = iso.upper()
    print(iso)
    if len(iso) != 3:
        print("The iso code you entered is invalid")
        main()
    else:
        covid = Covidiso(iso, cov)
        next, cols3 = Countryset(iso, Backdate, GDP=True)
        Corrgoals(next, cols3, iso)
        Plotgoals(next, iso)
        Plotindependent(next, iso)
        Plotcov(next, covid, iso)



if __name__ == "__main__":
    main(covid, Backdate)