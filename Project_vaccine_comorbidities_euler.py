#### Import packages and data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import random
random.seed(13)

# import data
# need to import this file in slightly different way because of different delimiters
clinical = pd.read_csv('data/covid19_clinical_sample.txt',sep='\t|,', engine = 'python')
GPic10 = pd.read_csv('data/GP_ic10_lookup.csv') 
mortality = pd.read_csv('data/patients_data_death_sample.csv')
mortality_coding = pd.read_csv('data/patients_data_death_decoding.csv')

#### Data Cleaning
# drop unnecessary columns
clinical.drop(columns = ['code_type', 'value'], inplace = True)

# remove duplicate entries 
clinical.drop_duplicates(inplace = True) 
mortality.drop_duplicates (inplace = True)

#### Make 'event_date' in 'clinical' a datetime type
# convert entire event_dt in datetime format
clinical['event_dt'] = pd.to_datetime(clinical['event_dt'])

#### Adding column with IC10 categories to clinical dataset for simpler analysis later on
# create a reduced dataframe for merging with the clinical dataframe
GPic10_red =  GPic10[['read_code', 'icd10_code']]
# rename to merge on the ctv3 column with the clinical dataset
GPic10_red = GPic10_red.rename(columns = {'read_code':'code'})

clinical_ic10 = pd.merge(clinical, GPic10_red, how='inner', on='code')

# remove not considered clinical events in the clinical dataset (e.g. external factors)
# remove O-Z, but leave R as adverse event & Q as underlying disease.

# tuple of categories we do not consider as adverse event or underlying disease:
exclude = ('O','P','S','T','U','V','W','X','Y','Z')

# broadcasting the clinical dataset for all icd codes except those we exclude
clinical_ic10 = clinical_ic10[~clinical_ic10['icd10_code'].str.startswith(exclude)]

# NEW OPTION of filtering data by vaccination status
clinical_vaccines_only = clinical[clinical['code']=='Y29e7'] # first dose

clinical_vaccines_only['issue_dt'] =  clinical_vaccines_only['event_dt']

# finding eids in the clinical dataset that had first covid-19 dose with code 'Y29e7'
id_vac = clinical[clinical['code']=='Y29e7']['eid'].unique()

#### Rename columns of mortality dataframe

# make a dictionary out of the mortality_coding dataframe 
dict_columns = {}
for index, value in enumerate(mortality_coding['Unnamed: 0']):
  dict_columns[value] = mortality_coding.loc[index, "description"]

mortality.rename(columns = dict_columns, inplace = True) # columns = a dictionary will automatically rename the columns

#### Add deaths to clinical_ic10 df
# subset mortality df to the eids with a deathdate that is not NaN
mortality_sub = mortality.dropna(subset=['DateDeath'])
# drop all the unnecessary columns (also drop age and sex, perhaps no time to analyse this), keep eid, deathdate, cause
mortality_sub.drop(columns = ['Sex', 'YearBirth', 'MonthBirth', 'DateRecruit', 'BMI', 'AgeDeath', 'AgeRecruit', 'cause of death 1', 'cause of death 2'], inplace = True)
# rename column 'PatientID' to 'eid' and 'DateDeath' to 'event_dt'
mortality_sub.rename(columns = {'PatientID':'eid', 'DateDeath':'event_dt'}, inplace = True) 

# add column 'icd10_code' to mortality, filled with string 'death'
mortality_sub['icd10_code'] = pd.Series(["death" for x in range(len(mortality.index))])

# make 'DateDeath' datetime type
mortality_sub['event_dt'] = pd.to_datetime(mortality_sub['event_dt'])

# now join them
clinical_ic10= pd.concat([clinical_ic10, mortality_sub], join = 'outer')

#### Add column with the date of vaccination to clinical_ic10
# merge clinical_ic10 and clinical_vaccines_only to add the issue_date (date of vaccine) to the df
clinical_vac = pd.merge(clinical_ic10, clinical_vaccines_only, how ='outer', on =['eid', 'code', 'event_dt'])

#### Add column in clinical_vac that says whether a patient is vaccinated or not
mask_vaccinated = clinical_vac['eid'].isin(id_vac)
clinical_vac['vaccinated'] = mask_vaccinated

#### Add random issue date to events of unvaccinated patients to 'issue_date_format' in clinical_vac
# make list from issue_date in clinical_vaccines_only
issue_dates = clinical_vaccines_only['issue_dt'].tolist()

# 'eid' of unvaccinated people
id_nvac = clinical_vac[clinical_vac['vaccinated']==False]['eid'].unique()

# create list of new dates
issue_dt_random = random.choices(issue_dates, k = len(id_nvac))

# make dictionary with random issue dates and unvaccinated eid
dict_nvac = dict(zip(id_nvac,issue_dt_random))

# fill na with mapping from the dictionary
clinical_vac['issue_dt'] = clinical_vac['issue_dt'].fillna(clinical_vac['eid'].map(dict_nvac))

#### Add column where it says if the event (event_date_format) was before or after the vaccination (issue_date)
before_after_vaccine = []
for i in range (len(clinical_vac)):
    if clinical_vac['event_dt'].iloc[i] >= clinical_vac['issue_dt'].iloc[i]:
        before_after_vaccine.append('after')
    else:
        before_after_vaccine.append('before')
clinical_vac['before_after_vaccine'] = before_after_vaccine


#### Analysis 1) Effect of the vaccination on adverse events ####

### 1a) Comparison of occurence of adverse events after vaccination of vaccinated and unvaccinated people
# initialize index names
dict_vacORnot = {'Vaccinated':1, 'Not vaccinated':0} #vaccination status$

## Plot number of AEs per thousands
def incidence(df_event, id_vac, id_nvac):
    """
    Calculates the number of events (AEs/deaths) per thousand individuals in a given population.
    
    Input:
    - df_event: a dataframe containing all events after the issue date considered in the analysis (AEs/deaths)
    - id_vac: series containing ID's of each individual (with each individual listed once) within the given population. // only vaccinated!
    - id_nvac: same as id_vac but with all non-vaccinated individuals
    
    Note: this is not exactly the incidence (would be number of events in specified time, which would be nice to analyse in the future)
    
    """
    
    # initialize df for putting in the values
    df_sum = pd.DataFrame(columns = ['Number of events', 'Number of individuals', 'Number of events per thousand individuals'], index = dict_vacORnot)
    
    for v in dict_vacORnot: # loop through vaccinated vs. not
        
        # put in the number of events (AEs) by subsetting the df_event into vaccinated/not vaccinated
        df_sum['Number of events'][v] = len(df_event[df_event['vaccinated'] == dict_vacORnot[v]])
    
    # put number of vaccinated/not vaccinated people in the population into the summary df
    df_sum['Number of individuals']['Vaccinated'] = len(id_vac) # vaccinated (row)
    df_sum['Number of individuals']['Not vaccinated'] = len(id_nvac) # not vaccinated (row)
    
    # calculate AEs/individual
    df_sum['Number of events per thousand individuals'] = df_sum['Number of events']/df_sum['Number of individuals']*1000
    
    return df_sum

# Make a df with only AEs (defined medical events w. ICD10 code after the (artificial) vaccination date)
# reminder: leave R as adverse event & Q as underlying disease.

# subset of clinical dataset for adverse events (=after vaccine)
df_AEs = clinical_vac[clinical_vac['before_after_vaccine']== 'after']
# drop NaN (from lines with vaccine)
df_AEs.dropna(subset = ['icd10_code'], inplace = True)

# remove Q (congenital diseases), which are not considered as adverse events
df_AEs = df_AEs[~df_AEs['icd10_code'].str.startswith('Q')]

# remove deaths, which are not considered adverse events
df_AEs = df_AEs[df_AEs['icd10_code'] != 'death'] 

# get number of people of the chosen population
# list of all IDs from the clinical dataset
id_clin = clinical['eid'].unique()

# list of all IDs from the mortality dataset
id_mor = mortality['PatientID'].unique()

# concatenate both lists to get all IDs in one list
id_all = np.concatenate((id_clin, id_mor))

# leave only unique IDs in the list to get the number of IDs using len()
id_all = np.unique(id_all)

# get array with IDs without vaccine by deleting the IDs with vaccine out of the array with all IDs
id_nvac = np.delete(id_all, np.isin(id_all,id_vac))

# use the function to create the summary table
df_sum_all = incidence(df_AEs, id_vac, id_nvac)

# save 
df_sum_all.to_csv('analysis/1a_total_AE_table.csv')

# Plot how oftern AEs occur in vaccinated vs. not vaccinated people for all individuals in the dataset.
plt.figure(figsize = (5,5))
plt.bar(df_sum_all.index, df_sum_all['Number of events per thousand individuals'])
plt.ylabel('Number of AEs per thousand individuals')
plt.title('Adverse events in the whole population of the dataset')

# save figure in vector format, bbox_inches='tight' so that the plot is not cropped
plt.savefig('analysis/1a_total_AE_barplot.svg', bbox_inches='tight')

# Make contingency table (not using pd.crosstab to avoid another huge dataset with all individuals and because we need to count unique IDs)
def contingency(df_event, id_vac, id_nvac, index = dict_vacORnot):
    """
    Makes a 2x2 contingency table, which can be used for the chi^2 test and calculating the risk ratio.
    
    Input:
    - df_event: a dataframe containing all events (AEs/deaths) considered in the analysis
    - id_vac: series containing ID's of each individual (with each individual listed once) within the given population.
    - id_nvac: same as id_vac but with all non-vaccinated individuals
    
    """

    # initialize contingency table
    crosstab = pd.DataFrame(columns = ['event', 'no event'], index = index)
    
    # fill contingency table with values in the subset

    for v in dict_vacORnot: # loop throug vaccinated vs. not

        
        # with events

        # number of nonvaccinated/vaccinated people with event = number of individuals with AEs
        crosstab.loc[v,'event'] = len(df_event[df_event['vaccinated']==dict_vacORnot[v]]['eid'].unique())

            
        # without events

        if v == 'Vaccinated': # for vaccinated people

            # total number of vaccinated individuals - those with events
            crosstab.loc[v,'no event'] = len(id_vac) - crosstab.loc[v,'event']

        else: # for not vaccinated people
            
            # total number of unvaccinated individuals - those with events
            crosstab.loc[v,'no event'] = len(id_nvac) - crosstab.loc[v,'event']
            
    
    # adding margins to the contingency table (=total value as row and column)
    
    # vaccinated vs. not
    crosstab.loc['Total',:] = np.sum(crosstab, axis = 0)
    # event vs. no event
    crosstab.loc[:,'Total'] = np.sum(crosstab, axis = 1)

    return crosstab

crosstab_all = contingency(df_AEs, id_vac, id_nvac)

# save
crosstab_all.to_csv('analysis/1a_contigency_table.csv')

### Chi-squared test
# define significance treshold
alpha = 0.05

# perform chi-squared test:
stat_1a, p_1a, dof_1a, expected_1a = sp.stats.chi2_contingency(crosstab_all.iloc[0:2,0:2], 1)

#save 
chi_square_1a = pd.DataFrame([[stat_1a, p_1a, dof_1a, expected_1a]],columns=['stat_1a', 'p_1a', 'dof_1a', 'expected_1a'])
chi_square_1a.to_csv('analysis/1a_chi_square.csv')

## risk ratio
# Done similar to Barda et al., but se and CI calculated as described on https://en.wikipedia.org/wiki/Relative_risk for simplicity. 
def riskratio(crosstab, z = 1.96):
    """
    Calculates the risk ratio.
    
    Returns the risk ratio with the confidence interval (CI) as a df and the adjusted contingency table:
    (RR, crosstab)
    
    Important: the input (crosstab) needs to have the following format:
    - 2x2
    - column 0 = events (e.g. AEs, deaths), column 1 = non-events
    - row 0 = intervention (e.g. vaccinated), column 1 = non-vaccinated
    """
    
    # calculating the risk ratio

    # risk of getting AE/ dying when vaccinated
    R_vac = crosstab.loc['Vaccinated','event']/crosstab.loc['Vaccinated','Total']
    # risk of getting "AE" or dying when not vaccinated
    R_nvac = crosstab.loc['Not vaccinated', 'event']/crosstab.loc['Not vaccinated', 'Total']

    # risk ratio
    RR = R_vac / R_nvac
    
    # natural log of the RR     
    logRR = np.log(RR) 

    # standard error
    se = np.sqrt(crosstab.loc['Vaccinated','no event']/ \
                 (crosstab.loc['Vaccinated','event']*crosstab.loc['Vaccinated','Total']) + \
                crosstab.loc['Not vaccinated','no event']/ \
                (crosstab.loc['Not vaccinated','event']*crosstab.loc['Not vaccinated','Total']))

    # Assuming normal distribution, 95% lie around 1.96 standard deviations of the mean.
    # as 95% CI is common, the z-score (z) 1.96 is put as the default value.

    # 1-alpha confidence interval (CI) 
    CI_lower = np.exp(np.log(RR) - se * z) # lower bound
    CI_upper = np.exp(np.log(RR) + se * z) # upper bound    
    
    # put the result in a series
    RR_result = pd.Series(data = [RR, CI_lower, CI_upper], index = ['RR', 'CI_lower', 'CI_upper'])
    
    return RR_result

RR_all = riskratio(crosstab_all) # will later be saved after 1b 


### 1b) Same comparison in subset of subjects with underlying medical condition
# Subset df to eids with AEs and underlying disease
# get dataset with only AEs AND people with underlying disease 
# = adverse events (events after vaccination) of people with underlying disease (= that had an event before vaccination)

# subset of the df with all AEs (df_AEs) with only people with underlying diseases (df_AEs_ud)
# reminder: leave R as adverse event (and general underlying disease) & Q as underlying disease.

# subset of clinical_vac df with only medical events before
df_ud= clinical_vac[clinical_vac['before_after_vaccine']== 'before']

# drop rows without icd10_code
df_ud.dropna(subset = ['icd10_code'], inplace = True)

# remove deaths to not consider people that died before
df_ud = df_ud[df_ud['icd10_code'] != 'death'] 

# ICD code R is kept in this analysis of general underlying diseases

# get id's of people with an underlying disease
id_ud = df_ud['eid'].unique()

# subset of the df with AEs with only people with an underlying disease
df_AEs_ud = df_AEs[df_AEs['eid'].isin(id_ud)]

# individuals in subset with only people with underlying disease AND the vaccine
id_ud_vac = df_ud[df_ud['vaccinated']==True]['eid'].unique()

# individuals in subset with only people with underlying disease AND NOT the vaccine
id_ud_nvac = df_ud[df_ud['vaccinated']==False]['eid'].unique()

## "incidence"
df_sum_ud = incidence(df_AEs_ud, id_ud_vac, id_ud_nvac)

# save
df_sum_ud.to_csv('analysis/1b_total_AE_table.csv')

plt.figure(figsize = (5,5))
plt.bar(df_sum_ud.index, df_sum_ud['Number of events per thousand individuals'])
plt.ylabel('Number of AEs per thousand individuals')
plt.title('Adverse events in subjects with underlying diseases')

# save figure in vector format, bbox_inches='tight' so that the plot is not cropped
plt.savefig('analysis/1b_total_AE_barplot.svg', bbox_inches='tight')

## Chi-squared test
# contingency table
crosstab_ud = contingency(df_AEs_ud, id_ud_vac, id_ud_nvac, index = dict_vacORnot)
crosstab_ud.to_csv('analysis/1b_contigency_table.csv')

# define significance treshold
alpha = 0.05

# perform chi-squared test:
stat_1b, p_1b, dof_1b, expected_1b = sp.stats.chi2_contingency(crosstab_ud.iloc[0:2,0:2], 1)

# save
chi_square_1b = pd.DataFrame([[stat_1b, p_1b, dof_1b, expected_1b]],columns=['stat_1b', 'p_1b', 'dof_1b', 'expected_1b'])
chi_square_1b.to_csv('analysis/1b_chi_square.csv')

## Risk ratio
RR_ud = riskratio(crosstab_ud)

# create a df for collecting the results of the risk ratio to compare the populations

# initialize
RR_result = pd.DataFrame(index = ['whole population', 'people with underlying diseases'], columns = ['RR', 'CI_lower', 'CI_upper'])

# fill it
# RR with data from all individuals in the mortality & clinical dataset
RR_result.loc['whole population',:] = RR_all
# RR with data from all individuals with underlying diseases
RR_result.loc['people with underlying diseases',:] = RR_ud

RR_result

# save 
pd.DataFrame(RR_result).to_csv('analysis/1a_b_RR_table.csv')

# plot the difference in RR with CI

# general figure configuration
plt.figure(figsize = (6,6))
plt.xscale("log")
plt.axvline(1, ls='--', linewidth=1, color='black')

# draw the RR
# RR results
x = RR_result['RR'].values
# error bar with same size in both directions
x_error = RR_result['RR']-RR_result['CI_lower']
# for equal spacing of the results
y = np.arange(len(RR_result))

# the plot
plt.errorbar(x, y, xerr = x_error, marker = "o", markersize = 10, color = 'b', ls='none')

# labelling
plt.xlabel('Risk Ratio (log scale)')
plt.title('Risk ratio of vaccination in people with underlying diseases or the general population') # renee: Risk ratio of vaccination on an adverse event in subjects with underlying diseases and the entire population
# labelling with ICD10 categories
plt.yticks(ticks = y, labels = RR_result.index) 
plt.ylim(-0.5,1.5)

# save figure in vector format, bbox_inches='tight' so that the plot is not cropped
plt.savefig('analysis/1a_b_RR_plot.svg', bbox_inches='tight')

### 1c) Same comparison in subset of subjects with specific underlying medical condition

## Loop through the ICD-10 categories
# dictionary of disease categories to analyse, from https://www.icd10data.com/ICD10CM/Codes
dict_cat = {('A','B'):'Certain infectious and parasitic diseases',
           ('C','D0','D1','D2','D3','D4'):'Neoplasms',
           ('D5','D6','D7','D8'):'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
           'E':'Endocrine, nutritional and metabolic diseases',
           'F':'Mental, Behavioral and Neurodevelopmental disorders',
           'G':'Diseases of the nervous system',
           ('H0','H1','H2','H3','H4','H5'):'Diseases of the eye and adnexa',
           ('H6','H7','H8', 'H9'): 'Diseases of the ear and mastoid process',
           'I':'Diseases of the circulatory system',
           'J':'Diseases of the respiratory system',
           'K':'Diseases of the digestive system',
           'L':'Diseases of the skin and subcutaneous tissue',
           'M':'Diseases of the musculoskeletal system and connective tissue',
           'N':'Diseases of the genitourinary system',
           'Q':'Congenital malformations, deformations and chromosomal abnormalities'}

# As mentioned by Barda et al., adjustment for multiple comparisons is not commonly done in studies regarding safety.
alpha = 0.05

# initialize 
# dataframe for collecting number of AEs per thousands with/without vaccinated people for all categories
df_sum_cat_all = pd.DataFrame(columns = dict_vacORnot, index=dict_cat.values()) 

# df for saving categories & their p-values
#df_chi2_cat = pd.DataFrame(columns = ['p-value', 'significant'], index=dict_cat.values())

# df for saving the risk ratio and CIs for the plot
RR_result_cat = pd.DataFrame(index = dict_cat.values(), columns = ['RR', 'CI_lower', 'CI_upper'])

# df for saving categories and their p-value, significance, RR and CI
df_combined = pd.DataFrame(columns = ['P-value', 'Significant', 'Risk Ratio', 'Confidence Interval'], index = dict_cat.values())



for i, ic in enumerate(dict_cat): # loop over ICD10 codes of each disease category
    
    # get dataset with only AEs AND people with specific underlying disease
    # subset of the df with all AEs (df_AEs) with only people with specific underlying diseases
    # reminder: leave R as adverse event & Q as underlying disease.
    # df_ud = subset of clinical_vac df with only diagnoses before vaccine as defined further above

    # subset with only underlying diseases within the specific category
    df_cat = df_ud[df_ud['icd10_code'].str.startswith(ic)]

    # IDs with underlying disease in the specific category
    id_cat = df_cat['eid'].unique()

    # subset of the df with AEs with only people with underlying disease in the specific category
    df_AEs_cat = df_AEs[df_AEs['eid'].isin(id_cat)]

    # get IDs of individuals in this subset with vs without vaccine 
    id_cat_vac = df_cat[df_cat['vaccinated']==True]['eid'].unique() # with vaccine
    id_cat_nvac = df_cat[df_cat['vaccinated']==False]['eid'].unique() # without vaccine

    ################################################################################
    # calculate the number of AEs per thousand people and save the result in a table
    df_sum_cat = incidence(df_AEs_cat, id_cat_vac, id_cat_nvac)
    
    # put in the values from the summary incidence table
    df_sum_cat_all.loc[dict_cat[ic],:] = df_sum_cat.loc[:, 'Number of events per thousand individuals']
    
    
    ################################################################################    
    # contingency table
    crosstab_cat = contingency(df_AEs_cat, id_cat_vac, id_cat_nvac, index = dict_vacORnot)
    
    ###############################################################################
    # perform chi-squared test:
    stat, p, dof, expected = sp.stats.chi2_contingency(crosstab_cat.iloc[0:2,0:2], 1)     ## changed crosstab_ud to crosstab_cat

    # save the p-value in the df
    df_combined.loc[dict_cat[ic],'P-value'] = p
    
    # mark in which categories there is a significant difference between vaccinated and not vaccinated
    if p <= alpha/2: # divided by two because of two-sided test (number of AEs could be higher or lower in vaccinated)
        df_combined.loc[dict_cat[ic],'Significant'] = 'Yes'
    else:
        df_combined.loc[dict_cat[ic],'Significant'] = 'No'
       
    
    ###############################################################################
    # risk ratio
    RR_cat = riskratio(crosstab_cat)

    # fill the df with the RR results for the plot
    RR_result_cat.loc[dict_cat[ic],:] = RR_cat
    
    CI_rounded = (round(RR_cat[1], 2), round(RR_cat[2], 2)) # round to one more decimal place than the original data has
    
    # new fill df with RR
    df_combined.loc[dict_cat[ic], ['Risk Ratio', 'Confidence Interval']] = round(RR_cat[0],2) , CI_rounded   
    
# total number of AEs in different categories
df_sum_cat_all.sort_values(by = 'Vaccinated', ascending = False, inplace = True) # sorting

# save
df_sum_cat_all.to_csv('analysis/1c_total_AE_table.csv')

# Outcomes of Chi squared test and RR
# df with p-value, significance, RR and CI
df_combined.to_csv('analysis/1c_chi_square_RR_table.csv')

plt.figure(figsize = (8,8))

# plot results for vaccinated and not on top of each other
for v in dict_vacORnot:
    plt.barh(df_sum_cat_all.index, df_sum_cat_all[v], alpha = 0.5, label = v)

# labelling
plt.title('Adverse events in people with diseases in specific ICD10 categories')
plt.xlabel('Number of AEs per thousand individuals')
plt.legend(loc = "upper right")

# save figure in vector format, bbox_inches='tight' so that the plot is not cropped
plt.savefig('analysis/1c_AE_barplot.svg', bbox_inches='tight')

# plot the RR with CI

# general figure configuration
plt.figure(figsize = (6,6))
plt.xscale("log")
plt.axvline(1, ls='--', linewidth=1, color='black')

# draw the RR
# RR results
x = RR_result_cat['RR'].values
# error bar with same size in both directions
x_error = RR_result_cat['RR']-RR_result_cat['CI_lower']
# for equal spacing of the results
y = np.arange(len(RR_result_cat))

# the plot
plt.errorbar(x, y, xerr = x_error, marker = "o", markersize = 10, color = 'b', ls='none')

# labelling
plt.xlabel('Risk Ratio (log scale)')
plt.title('Risk ratio of vaccination in people with underlying diseases in different ICD10 categories')
# labelling with ICD10 categories
plt.yticks(ticks = y, labels = RR_result_cat.index) 

# save figure in vector format, bbox_inches='tight' so that the plot is not cropped
plt.savefig('analysis/1c_AE_RR.svg', bbox_inches='tight')


#### Analysis 2) Effect of the vaccination on mortality ####

### 2a) Comparison of mortality rate after vaccination of vaccinated and unvaccinated people

# subset of clinical_vac dataset for adverse events (=after vaccine)
df_deaths = clinical_vac[clinical_vac['before_after_vaccine']== 'after']

# make dataframe with only deaths 
df_deaths = df_deaths[df_deaths['icd10_code'] == 'death'] 

crosstab_all_deaths = contingency(df_deaths, id_vac, id_nvac)

# save
crosstab_all_deaths.to_csv('analysis/2a_contigency_table.csv')

## Chi-squared test
# perform chi-squared test and name variavles differently for each test
# e.g. 1a, 1b, etc.
stat_2a, p_2a, dof_2a, expected_2a = sp.stats.chi2_contingency(crosstab_all_deaths.iloc[0:2,0:2], 1)

# save
chi_square_2a = pd.DataFrame([[stat_2a, p_2a, dof_2a, expected_2a]],columns=['stat_2a', 'p_2a', 'dof_2a', 'expected_2a'])
chi_square_2a.to_csv('analysis/2a_chi_square.csv')

## Risk ratio
RR_all_deaths = riskratio(crosstab_all_deaths)

# save together with 2b

### 2b) Same comparison in subset of subjects with underlying medical condition

# subset of the df with deaths with only people with an underlying disease - non in this dataset
df_deaths_ud = df_deaths[df_deaths['eid'].isin(id_ud)]

df_sum_ud = incidence(df_deaths_ud, id_ud_vac, id_ud_nvac)

plt.figure(figsize = (5,5))
plt.bar(df_sum_ud.index, df_sum_ud['Number of events per thousand individuals'])
plt.ylabel('Number of deaths per thousand individuals')
plt.title('Deaths in people with underlying diseases')

## Chi-squared test
# contingency table
crosstab_ud_deaths = contingency(df_deaths_ud, id_ud_vac, id_ud_nvac, index = dict_vacORnot)

# save
crosstab_ud_deaths.to_csv('analysis/2b_contigency_table.csv')

# perform chi-squared test:
stat_2b, p_2b, dof_2b, expected_2b = sp.stats.chi2_contingency(crosstab_ud_deaths.iloc[0:2,0:2], 1)

# save
chi_square_2b = pd.DataFrame([[stat_2a, p_2a, dof_2a, expected_2a]],columns=['stat_2a', 'p_2a', 'dof_2a', 'expected_2a'])
chi_square_2b.to_csv('analysis/2b_chi_square.csv')

## Risk ratio
RR_ud_deaths = riskratio(crosstab_ud_deaths)

# create a df for collecting the results of the risk ratio to compare the populations

# initialize
RR_result_deaths = pd.DataFrame(index = ['Whole population', 'People with underlying diseases'], columns = ['RR', 'CI_lower', 'CI_upper'])

# fill it
# RR with data from all individuals in the mortality & clinical dataset
RR_result_deaths.loc['Whole population',:] = RR_all_deaths
# RR with data from all individuals with underlying diseases
RR_result_deaths.loc['People with underlying diseases',:] = RR_ud_deaths

# save
pd.DataFrame(RR_result_deaths).to_csv('analysis/1a_b_RR.csv')

# plot the difference in RR with CI

# general figure configuration
plt.figure(figsize = (6,6))
plt.xscale("log")
plt.axvline(1, ls='--', linewidth=1, color='black')

# draw the RR
# RR results
x = RR_result_deaths['RR'].values
# error bar with same size in both directions
x_error = RR_result_deaths['RR']-RR_result_deaths['CI_lower']
# for equal spacing of the results
y = np.arange(len(RR_result_deaths))

# the plot
plt.errorbar(x, y, xerr = x_error, marker = "o", markersize = 10, color = 'b', ls='none')

# labelling
plt.xlabel('Risk Ratio (log scale)')
plt.title('Risk ratio of vaccination on death in subjects with underlying diseases or the general population') 
# labelling with ICD10 categories
plt.yticks(ticks = y, labels = RR_result_deaths.index) 
plt.ylim(-0.5,1.5)

# save figure in vector format, bbox_inches='tight' so that the plot is not cropped
plt.savefig('analysis/2a_b_RR_plot.svg', bbox_inches='tight')

### 2c) Same comparison in subset of subjects with specific underlying medical condition

# initialize 

# dataframe for collecting number of AEs per thousands with/without vaccinated people for all categories
df_sum_cat_all_deaths = pd.DataFrame(columns = dict_vacORnot, index=dict_cat.values()) 

# df for saving the risk ratio and CI
RR_result_cat_deaths = pd.DataFrame(index = dict_cat.values(), columns = ['RR', 'CI_lower', 'CI_upper'])

# df for saving categories and their p-value, significance, RR and CI
df_combined_deaths = pd.DataFrame(columns = ['P-value', 'Significant', 'Risk Ratio', 'Confidence Interval'], index = dict_cat.values())


for i, ic in enumerate(dict_cat): # loop over ICD10 codes of each disease category
    
    # get dataset with only deaths AND people with specific underlying disease
    # subset of the df with all deaths (df_deaths) with only people with specific underlying diseases
    # reminder: leave R as adverse event & Q as underlying disease.
    # df_ud = subset of clinical_vac df with only diagnoses before vaccine as defined further above

    # subset with only underlying diseases within the specific category (=ic)
    df_cat = df_ud[df_ud['icd10_code'].str.startswith(ic)]

    # IDs with underlying disease in the specific category
    id_cat = df_cat['eid'].unique()

    # subset of the df with deaths after vaccination only in people with underlying disease in the specific category
    df_deaths_cat = df_deaths[df_deaths['eid'].isin(id_cat)]

    # get IDs of individuals in this subset with vs without vaccine 
    id_cat_vac = df_cat[df_cat['vaccinated']==True]['eid'].unique() # with vaccine
    id_cat_nvac = df_cat[df_cat['vaccinated']==False]['eid'].unique() # without vaccine

    ################################################################################
    # calculate the number of deaths per thousand people and save the result in a table
    df_sum_cat_deaths = incidence(df_deaths_cat, id_cat_vac, id_cat_nvac)
    
    # put in the values from the summary table
    df_sum_cat_all_deaths.loc[dict_cat[ic],:] = df_sum_cat_deaths.loc[:,'Number of events per thousand individuals']
    
    
    ################################################################################    
    # contingency table
    crosstab_cat = contingency(df_deaths_cat, id_cat_vac, id_cat_nvac, index = dict_vacORnot)
    
    ###############################################################################
    # perform chi-squared test:
    stat, p, dof, expected = sp.stats.chi2_contingency(crosstab_cat.iloc[0:2,0:2], 1)       # changed to crosstab_cat from crosstab_ud
    
    # save the p-value in the df
    df_combined_deaths.loc[dict_cat[ic],'P-value'] = p
    
    # mark in which categories there is a significant difference between vaccinated and not vaccinated
    if p <= alpha/2: # divided by two because of two-sided test (number of AEs could be higher or lower in vaccinated)
        df_combined_deaths.loc[dict_cat[ic],'Significant'] = 'Yes'
    else:
        df_combined_deaths.loc[dict_cat[ic],'Significant'] = 'No'
       
    ###############################################################################
    # risk ratio
    RR_cat = riskratio(crosstab_cat)

    # fill the df with the RR results for the plot later
    RR_result_cat_deaths.loc[dict_cat[ic],:] = RR_cat
    
    CI_rounded = (round(RR_cat[1], 2), round(RR_cat[2], 2)) # round to one more decimal place than the original data has
    
    # fill df with RR
    df_combined_deaths.loc[dict_cat[ic], ['Risk Ratio', 'Confidence Interval']] = round(RR_cat[0],2) , CI_rounded
    

df_sum_cat_all_deaths.sort_values(by = 'Vaccinated', ascending = False, inplace = True) # sorting

# save
df_sum_cat_all_deaths.to_csv('analysis/2c_total_mortality_table.csv')

# df with p-value, significance, RR and CI
df_combined_deaths.to_csv('analysis/2c_chi_square_RR_table.csv')

plt.figure(figsize = (8,8))

# plot results for vaccinated and not on top of each other
for v in dict_vacORnot:
    plt.barh(df_sum_cat_all_deaths.index, df_sum_cat_all_deaths[v], alpha = 0.5, label = v)

# labelling
plt.title('Deaths in subjects with underlying diseases in specific ICD10 categories')
plt.xlabel('Number of deaths per thousand individuals')
plt.legend(loc = "upper right")

# save figure in vector format, bbox_inches='tight' so that the plot is not cropped
plt.savefig('analysis/2c_deaths_barplot.svg', bbox_inches='tight')

# plot the RR with CI

# general figure configuration
plt.figure(figsize = (6,6))
plt.xscale("log")
plt.axvline(1, ls='--', linewidth=1, color='black')

# draw the RR
# RR results
x = RR_result_cat_deaths['RR'].values
# error bar with same size in both directions
x_error = RR_result_cat_deaths['RR']-RR_result_cat_deaths['CI_lower']
# for equal spacing of the results
y = np.arange(len(RR_result_cat_deaths))

# the plot
plt.errorbar(x, y, xerr = x_error, marker = "o", markersize = 10, color = 'b', ls='none')

# labelling
plt.xlabel('Risk Ratio (log scale)')
plt.title('Risk ratio of vaccination in people with underlying diseases in different ICD10 categories') # renee: Risk ratio of vaccination on an adverse event in subjects with underlying diseases in specific ICD-10 categories
# labelling with ICD10 categories
plt.yticks(ticks = y, labels = RR_result_cat_deaths.index) 

# save figure in vector format, bbox_inches='tight' so that the plot is not cropped
plt.savefig('analysis/2c_deaths_RR.svg', bbox_inches='tight')












