# SDG Data Analysis and Visualization Tool
Created by: Dustin Cooper  
For: GSTR 410  
Berea College

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This repository uses the databases from the United Nations 
Sustainability Goals, Worldbank database, and Our World Covid database.
The purpose of this repo is twofold. It serves as my final project
for the course and cleans and visualizes the data mentioned above.
More specifically, I remove all columns from the UN data that are not
the year, country code, or a total goal score.
From the Covid data I isolate the last entry for each year
for the country code you enter. One current flaw (7/25/22)
is there is currently no error handling around the user inputted
country code. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The visualizations created
are mainly correlation matrices and line graphs. The correlation
matrices correlate each goal with each other goal while the line
charts plot the goal for each score over the lifespan of the 
database (2000-2022). 