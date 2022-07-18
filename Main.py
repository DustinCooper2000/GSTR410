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

Covidiso('IRN', covid)

def Countryset(iso_code, df):
    Country = df[df.iso_code == 'IRN']
    Country.columns

    Iran2022 = SDG2022[SDG2022.iso_code == 'IRN']
    next = pd.concat([Country, Iran2022], axis=0)
    next['Year'] = next['Year'].fillna(2022.0)
    next['Population'] = next['Population'].fillna(85028760.0)
    next.dropna(axis=1, inplace=True)
    next.tail(5)

    cols3 = next.columns
    return next, cols3

next, cols3 = Countryset('IRN', Backdate)

def Corrgoals(df, cols):
    plt.figure(figsize=[16,8])
    sbn.heatmap(df[cols].corr(), annot=True, cmap = 'viridis')

#Corrgoals(next, cols3)

def Plotgoals(df):
    goal = 1
    colo = ["", "xkcd:purple", "xkcd:green", "xkcd:blue", "xkcd:pink", "xkcd:brown", "xkcd:red", "xkcd:light blue", "xkcd:teal", "xkcd:orange", "xkcd:light green", "xkcd:magenta",
            "xkcd:yellow", "xkcd:grey", "xkcd:lime green", "xkcd:light purple", "xkcd:dark green", "xkcd:dark gold"]


    plt.figure(figsize=[16,8])
    for i in next.columns:
      if 'Goal' in i:
        current = 'Goal ' + str(goal)
        plt.plot(df['Year'], next[i], label=current, color=colo[goal])
        goal += 1


    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 20,
            }

    plt.xlabel("Years", fontdict=font)
    plt.ylabel("Goal Score", fontdict=font)
    plt.legend()
    plt.show()

#Plotgoals(next)
next.describe()

def Plotindependent(df):
    fig, axs = plt.subplots(3, 5, figsize=(16, 10), constrained_layout=True,
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

        line, = ax.plot('Year', column, data=next, lw=2.5)
        ax.set_title(column, fontsize='small', loc='left')
        ax.set_ylim([0, 100])
        ax.grid()

    # fig.supxlabel('Year')
    # fig.supylabel('Goal Score')

    plt.show()


#Plotindependent(next)
print(covid)

# next = next.merge(covid, on='Year', how='outer')
# next.head(23)
#
# goal = 1
# colo = ["", "xkcd:purple", "xkcd:green", "xkcd:blue", "xkcd:pink", "xkcd:brown", "xkcd:red", "xkcd:light blue",
#         "xkcd:teal", "xkcd:orange", "xkcd:light green", "xkcd:magenta",
#         "xkcd:yellow", "xkcd:grey", "xkcd:lime green", "xkcd:light purple", "xkcd:dark green", "xkcd:dark gold"]
#
# plt.figure(figsize=[16, 8])
# next['total_cases'] = next['total_cases'].fillna(0)
#
# next['total_cases'] = (next['total_cases'] - next['total_cases'].min()) / (
#             next['total_cases'].max() - next['total_cases'].min()) * 100
# print(next['total_cases'])
# # next['total_cases'] = (next.total_cases–mincases) / (maxcases–mincases) * 100
# plt.plot(next['Year'], next['total_cases'], label='Cases', color="xkcd:black")
#
# for i in next.columns:
#     if 'Goal' in i:
#         current = 'Goal ' + str(goal)
#         plt.plot(next['Year'], next[i], label=current, color=colo[goal])
#         goal += 1
#         # plt.plot(next['Year'], next['total_cases'], label="Cases", color="xkcd:black")
#
# font = {'family': 'serif',
#         'color': 'darkred',
#         'weight': 'normal',
#         'size': 20,
#         }
# plt.xticks(np.arange(min(next["Year"]), max(next["Year"]) + 1, 1.0))
# plt.xlabel("Years", fontdict=font)
# plt.ylabel("Goal Score", fontdict=font)
# plt.grid()
# plt.legend()
# plt.show()


