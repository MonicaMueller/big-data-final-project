### IMPORT PACKAGES ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from datetime import datetime, timedelta
import random
random.seed(13)

### IMPORT DATA ###

clinical = pd.read_csv('data/covid19_clinical_sample.txt',sep='\t|,', engine = 'python')
GPic10 = pd.read_csv('data/GP_ic10_lookup.csv') 
scripts = pd.read_csv('data/covid19_prescriptions_sample.txt', sep='\t|,', engine = 'python')
mortality = pd.read_csv('data/patients_data_death_sample.csv')
mortality_coding = pd.read_csv('data/patients_data_death_decoding.csv')

### PREPARE DATA ###

# add IC-10 to clinical, exclude irrelevant columns
GPic10_red =  GPic10[['read_code', 'icd10_code']]
GPic10_red = GPic10_red.rename(columns = {'read_code':'code'})
clinical_ic10 = pd.merge(clinical, GPic10_red, how='inner', on='code')
clinical_ic10.drop(columns = ['code_type', 'value', 'Unnamed: 0'], inplace = True)
clinical_ic10.drop_duplicates(inplace = True) 

exclude = ('O','P','S','T','U','V','W','X','Y','Z')
clinical_ic10 = clinical_ic10[~clinical_ic10['icd10_code'].str.startswith(exclude)]

# prepare mortality df
dict_columns = {}
for index, value in enumerate(mortality_coding['Unnamed: 0']):
  dict_columns[value] = mortality_coding.loc[index, "description"]
mortality.rename(columns = dict_columns, inplace = True) # columns = a dictionary will automatically rename the columns

# find vaccination status
dict_vac = {39326911000001101: 'Moderna', 39230211000001104: 'Janssen', \
            39826711000001101:'Medicago', 39473011000001103: 'Baxter', \
            39114911000001105: 'AstraZeneca', 39115611000001103: 'Pfizer',\
            39373511000001104: 'Valneva'}

for dmd_code in dict_vac:
    mask_vac = scripts['dmd_code']== dmd_code

mask_vac = scripts['dmd_code'].isin(dict_vac)

id_vac = scripts[mask_vac]['eid'] 

# change to datetime format, subset scripts to vaccinations
clinical_ic10['event_date_format'] = pd.to_datetime(clinical_ic10['event_dt'])
scripts.drop(columns = ['Unnamed: 0'], inplace = True)
mask_scripts_vac = scripts['dmd_code'].isin(dict_vac)
scripts_vaccines_only = scripts[mask_scripts_vac] 
scripts_vaccines_only['issue_date_format'] = pd.to_datetime(scripts_vaccines_only['issue_date'])

# add deaths to clinical_ic10 df
mortality_sub = mortality.dropna(subset=['DateDeath'])
mortality_sub.drop(columns = ['Unnamed: 0', 'Sex', 'YearBirth', 'MonthBirth', 'DateRecruit', 'BMI', 'AgeDeath', 'AgeRecruit', 'cause of death 1', 'cause of death 2'], inplace = True)
mortality_sub.rename(columns = {'PatientID':'eid', 'DateDeath':'event_date_format'}, inplace = True) 

mortality_sub['icd10_code'] = pd.Series(["death" for x in range(len(mortality.index))])
mortality_sub['event_date_format'] = pd.to_datetime(mortality_sub['event_date_format'])

clinical_ic10= pd.concat([clinical_ic10, mortality_sub], join = 'outer')

# add vaccination date and whether a subject is vaccinated
clinical_vac = pd.merge(clinical_ic10, scripts_vaccines_only, how ='outer', on ='eid')
clinical_vac.drop(columns = ['event_dt', 'issue_date'], inplace = True)

mask_vaccinated = clinical_vac['eid'].isin(id_vac)
clinical_vac['vaccinated'] = mask_vaccinated

# add random issue date to unvaccinated subjects
issue_dates = scripts_vaccines_only['issue_date_format'].tolist()
id_nvac = clinical_vac[clinical_vac['vaccinated']==False]['eid'].unique()
issue_dt_random = random.choices(issue_dates, k = len(id_nvac))
dict_nvac = dict(zip(id_nvac,issue_dt_random))
clinical_vac['issue_date_format'] = clinical_vac['issue_date_format'].fillna(clinical_vac['eid'].map(dict_nvac))

# add before/after vaccine column
before_after_vaccine = []
for i in range (len(clinical_vac)):
    if clinical_vac['event_date_format'].iloc[i] >= clinical_vac['issue_date_format'].iloc[i]:
        before_after_vaccine.append('after')
    else:
        before_after_vaccine.append('before')
clinical_vac['before_after_vaccine'] = before_after_vaccine

### ANALYSIS 1) EFFECTS OF VACCINATION ON ADVERSE EVENTS ###

## 1a) Comparison of occurence of adverse events after vaccination of vaccinated and unvaccinated people

dict_vacORnot = {'Vaccinated':1, 'Not vaccinated':0} #vaccination status

# define function 'incidence'
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
    
        # if we have no event (happens in the subset), add 0.5 for testing the plot and analysis further below
        # -> still visible in the output that it was added
    
    # put number of vaccinated/not vaccinated people in the population into the summary df
    df_sum['Number of individuals']['Vaccinated'] = len(id_vac) # vaccinated (row)
    df_sum['Number of individuals']['Not vaccinated'] = len(id_nvac) # not vaccinated (row)
    
    # calculate AEs/individual
    df_sum['Number of events per thousand individuals'] = df_sum['Number of events']/df_sum['Number of individuals']*1000
    
    return df_sum

# subset of clinical dataset for adverse events (=after vaccine)
df_AEs = clinical_vac[clinical_vac['before_after_vaccine']== 'after']
df_AEs = df_AEs[~df_AEs['icd10_code'].str.startswith('Q')]
df_AEs = df_AEs[df_AEs['icd10_code'] != 'death'] 

id_clin = clinical['eid'].unique()
id_scr = scripts['eid'].unique()
id_all = np.concatenate((id_clin, id_scr))
id_all = np.unique(id_all)
id_nvac = np.delete(id_all, np.isin(id_all,id_vac))

#  incidence of all subjects
df_sum_all = incidence(df_AEs, id_vac, id_nvac)
df_sum_all.to_csv('analysis/1a_total_AE_table.csv')

# barplot total AE
plt.figure(figsize = (5,5))
plt.bar(df_sum_all.index, df_sum_all['Number of events per thousand individuals'])
plt.ylabel('Number of AEs per thousand individuals')
plt.title('Adverse events in the whole population of the dataset')
plt.savefig('analysis/1a_total_AE_barplot.svg', bbox_inches='tight')

# define function 'contingency'
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

# contigency table
crosstab_all = contingency(df_AEs, id_vac, id_nvac)
crosstab_all.to_csv('analysis/1a_contigency_table.csv')

# chi-squared test
alpha = 0.05
stat_1a, p_1a, dof_1a, expected_1a = sp.stats.chi2_contingency(crosstab_all.iloc[0:2,0:2], 1)
chi_square_1a = pd.DataFrame([[stat_1a, p_1a, dof_1a, expected_1a]],columns=['stat_1a', 'p_1a', 'dof_1a', 'expected_1a'])
chi_square_1a.to_csv('analysis/1a_chi_square.csv')

# define function 'riskratio'
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

# risk ratio
RR_all = riskratio(crosstab_all)

## 1b) Same comparison in subset of subjects with underlying medical condition

# subset to subjects with underlying disease
df_ud= clinical_vac[clinical_vac['before_after_vaccine']== 'before']
df_ud.dropna(subset = ['icd10_code'], inplace = True)
df_ud = df_ud[df_ud['icd10_code'] != 'death'] 
id_ud = df_ud['eid'].unique()
df_AEs_ud = df_AEs[df_AEs['eid'].isin(id_ud)]
df_AEs_ud = df_AEs_ud[df_AEs_ud['icd10_code'] != 'death'] 

# subset of eids with underlying disease AND the vaccine / not the vaccine
id_ud_vac = df_ud[df_ud['vaccinated']==True]['eid'].unique()
id_ud_nvac = df_ud[df_ud['vaccinated']==False]['eid'].unique()

# incidence of AE in subjects with underlying disease
df_sum_ud = incidence(df_AEs_ud, id_ud_vac, id_ud_nvac)
df_sum_ud.to_csv('analysis/1b_total_AE_table.csv')

# plot
plt.figure(figsize = (5,5))
plt.bar(df_sum_ud.index, df_sum_ud['Number of events per thousand individuals'])
plt.ylabel('Number of AEs per thousand individuals')
plt.title('Adverse events in subjects with underlying diseases')
plt.savefig('analysis/1b_total_AE_barplot.svg', bbox_inches='tight')

# contigency
crosstab_ud = contingency(df_AEs_ud, id_ud_vac, id_ud_nvac, index = dict_vacORnot)
crosstab_ud.to_csv('analysis/1b_contigency_table.csv')

# chi-squared
stat_1b, p_1b, dof_1b, expected_1b = sp.stats.chi2_contingency(crosstab_ud.iloc[0:2,0:2], 1)
chi_square_1b = pd.DataFrame([[stat_1b, p_1b, dof_1b, expected_1b]],columns=['stat_1b', 'p_1b', 'dof_1b', 'expected_1b'])
chi_square_1b.to_csv('analysis/1b_chi_square.csv')

# risk ratio, compared to 1a)
RR_ud = riskratio(crosstab_ud)
RR_result = pd.DataFrame(index = ['whole population', 'people with underlying diseases'], columns = ['RR', 'CI_lower', 'CI_upper'])
RR_result.loc['whole population',:] = RR_all
RR_result.loc['people with underlying diseases',:] = RR_ud
pd.DataFrame(RR_result).to_csv('analysis/1a_b_RR_table.csv')

# plot 
plt.figure(figsize = (6,6))
plt.xscale("log")
plt.axvline(1, ls='--', linewidth=1, color='black')

x = RR_result['RR'].values
x_error = RR_result['RR']-RR_result['CI_lower']
y = np.arange(len(RR_result))

plt.errorbar(x, y, xerr = x_error, marker = "o", markersize = 10, color = 'b', ls='none')
plt.xlabel('Risk Ratio (log scale)')
plt.title('Risk ratio of vaccination in people with underlying diseases or the general population') # renée: Risk ratio of vaccination on an adverse event in subjects with underlying diseases and the entire population
plt.yticks(ticks = y, labels = RR_result.index) 
plt.ylim(-0.5,1.5)
plt.savefig('analysis/1a_b_RR_plot.svg', bbox_inches='tight')

## 1c) Same comparison in subset of subjects with specific underlying medical condition 

# repeat and loop through the IC-10 categories 

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

# adjustment for multiple comparisons is not commonly done in studies regarding safety
alpha = 0.05

# initialize 
df_sum_cat_all = pd.DataFrame(columns = dict_vacORnot, index=dict_cat.values()) 
RR_result_cat = pd.DataFrame(index = dict_cat.values(), columns = ['RR', 'CI_lower', 'CI_upper'])
df_combined = pd.DataFrame(columns = ['P-value', 'Significant', 'Risk Ratio', 'Confidence Interval'], index = dict_cat.values())

for i, ic in enumerate(dict_cat): 
    # subset with only underlying diseases within the specific category
    df_cat = df_ud[df_ud['icd10_code'].str.startswith(ic)]

    # IDs with underlying disease in the specific category
    id_cat = df_cat['eid'].unique()

    # subset of the df with AEs with only people with underlying disease in the specific category
    df_AEs_cat = df_AEs[df_AEs['eid'].isin(id_cat)]

    # get IDs of individuals in this subset with vs without vaccine 
    id_cat_vac = df_cat[df_cat['vaccinated']==True]['eid'].unique() # with vaccine
    id_cat_nvac = df_cat[df_cat['vaccinated']==False]['eid'].unique() # without vaccine

    # calculate the number of AEs per thousand people and save the result in a table
    df_sum_cat = incidence(df_AEs_cat, id_cat_vac, id_cat_nvac)
    
    # put in the values from the summary incidence table
    df_sum_cat_all.loc[dict_cat[ic],:] = df_sum_cat.loc[:, 'Number of events per thousand individuals']
        
    # contingency table
    crosstab_cat = contingency(df_AEs_cat, id_cat_vac, id_cat_nvac, index = dict_vacORnot)
    
    # chi-squared test:
    stat, p, dof, expected = sp.stats.chi2_contingency(crosstab_cat.iloc[0:2,0:2], 1)     ## changed crosstab_ud to crosstab_cat

    # save the p-value in the df
    df_combined.loc[dict_cat[ic],'P-value'] = p
    
    # mark in which categories there is a significant difference between vaccinated and not vaccinated
    if p <= alpha/2: # divided by two because of two-sided test (number of AEs could be higher or lower in vaccinated)
        df_combined.loc[dict_cat[ic],'Significant'] = 'Yes'
    else:
        df_combined.loc[dict_cat[ic],'Significant'] = 'No'
       
    
    # risk ratio
    RR_cat = riskratio(crosstab_cat)

    # fill the df with the RR results for the plot
    RR_result_cat.loc[dict_cat[ic],:] = RR_cat
    
    CI_rounded = (round(RR_cat[1], 2), round(RR_cat[2], 2)) # round to one more decimal place than the original data has
    
    # new fill df with RR
    df_combined.loc[dict_cat[ic], ['Risk Ratio', 'Confidence Interval']] = round(RR_cat[0],2) , CI_rounded
    
# total number of AEs in IC-10 categories
df_sum_cat_all.sort_values(by = 'Vaccinated', ascending = False, inplace = True) # sorting
df_sum_cat_all.to_csv('analysis/1c_total_AE_table.csv')

# chi-squared and RR
df_combined.to_csv('analysis/1c_chi_square_RR_table.csv')

# plot
plt.figure(figsize = (8,8))

for v in dict_vacORnot:
    plt.barh(df_sum_cat_all.index, df_sum_cat_all[v], alpha = 0.5, label = v)

plt.title('Adverse events in people with diseases in specific ICD10 categories')
plt.xlabel('Number of AEs per thousand individuals')
plt.legend(loc = "upper right")
plt.savefig('analysis/1c_AE_barplot.svg', bbox_inches='tight')

# plot RR with CI

plt.figure(figsize = (6,6))
plt.xscale("log")
plt.axvline(1, ls='--', linewidth=1, color='black')

x = RR_result_cat['RR'].values
x_error = RR_result_cat['RR']-RR_result_cat['CI_lower']
y = np.arange(len(RR_result_cat))

plt.errorbar(x, y, xerr = x_error, marker = "o", markersize = 10, color = 'b', ls='none')
plt.xlabel('Risk Ratio (log scale)')
plt.title('Risk ratio of vaccination in people with underlying diseases in different ICD10 categories')
plt.yticks(ticks = y, labels = RR_result_cat.index) 
plt.savefig('analysis/1c_AE_RR.svg', bbox_inches='tight')

### ANALYSIS 2) EFFECT OF VACCINATION ON MORTALITY ###

## 2a) Comparison of mortality rate after vaccination of vaccinated and unvaccinated people

# subset clinical to only deaths entries
df_deaths = clinical_vac[clinical_vac['before_after_vaccine']== 'after']
df_deaths = df_deaths[df_deaths['icd10_code'] == 'death'] 
ls_death = ['Deaths', 'No deaths'] # deaths

# mortality in entire study population (contigency)
crosstab_all_deaths = contingency(df_deaths, id_vac, id_nvac)
crosstab_all_deaths.to_csv('analysis/2a_contigency_table.csv')

# chi-squared 
stat_2a, p_2a, dof_2a, expected_2a = sp.stats.chi2_contingency(crosstab_all_deaths.iloc[0:2,0:2], 1)
chi_square_2a = pd.DataFrame([[stat_2a, p_2a, dof_2a, expected_2a]],columns=['stat_2a', 'p_2a', 'dof_2a', 'expected_2a'])
chi_square_2a.to_csv('analysis/2a_chi_square.csv')

# RR
RR_all_deaths = riskratio(crosstab_all_deaths)

## 2b) Same comparison in subset of subjects with underlying medical condition

# subset to subjects with underlying disease
df_deaths_ud = df_deaths[df_deaths['eid'].isin(id_ud)]

# chi-squared
crosstab_ud_deaths = contingency(df_deaths_ud, id_ud_vac, id_ud_nvac, index = dict_vacORnot)
crosstab_ud_deaths.to_csv('analysis/2b_contigency_table.csv')

stat_2b, p_2b, dof_2b, expected_2b = sp.stats.chi2_contingency(crosstab_ud_deaths.iloc[0:2,0:2], 1)
chi_square_2b = pd.DataFrame([[stat_2a, p_2a, dof_2a, expected_2a]],columns=['stat_2a', 'p_2a', 'dof_2a', 'expected_2a'])
chi_square_2b.to_csv('analysis/2b_chi_square.csv')

# RR
RR_ud_deaths = riskratio(crosstab_ud_deaths)

# plot
RR_result_deaths = pd.DataFrame(index = ['Whole population', 'People with underlying diseases'], columns = ['RR', 'CI_lower', 'CI_upper'])
RR_result_deaths.loc['Whole population',:] = RR_all_deaths
RR_result_deaths.loc['People with underlying diseases',:] = RR_ud_deaths
pd.DataFrame(RR_result_deaths).to_csv('analysis/1a_b_RR.csv')

# plot difference 2a) and 2b)
plt.figure(figsize = (6,6))
plt.xscale("log")
plt.axvline(1, ls='--', linewidth=1, color='black')

x = RR_result_deaths['RR'].values
x_error = RR_result_deaths['RR']-RR_result_deaths['CI_lower']
y = np.arange(len(RR_result_deaths))

plt.errorbar(x, y, xerr = x_error, marker = "o", markersize = 10, color = 'b', ls='none')
plt.xlabel('Risk Ratio (log scale)')
plt.title('Risk ratio of vaccination on death in subjects with underlying diseases or the general population') 
plt.yticks(ticks = y, labels = RR_result_deaths.index) 
plt.ylim(-0.5,1.5)
plt.savefig('analysis/2a_b_RR_plot.svg', bbox_inches='tight')

## 2c) Same comparison in subset of subjects with specific underlying medical condition 

#initialize
df_sum_cat_all_deaths = pd.DataFrame(columns = dict_vacORnot, index=dict_cat.values()) 
RR_result_cat_deaths = pd.DataFrame(index = dict_cat.values(), columns = ['RR', 'CI_lower', 'CI_upper'])
df_combined_deaths = pd.DataFrame(columns = ['P-value', 'Significant', 'Risk Ratio', 'Confidence Interval'], index = dict_cat.values())

# loop through IC-10 categories
for i, ic in enumerate(dict_cat): # loop over ICD10 codes of each disease category
    
     # subset with only underlying diseases within the specific category (=ic)
    df_cat = df_ud[df_ud['icd10_code'].str.startswith(ic)]

    # IDs with underlying disease in the specific category
    id_cat = df_cat['eid'].unique()

    # subset of the df with deaths after vaccination only in people with underlying disease in the specific category
    df_deaths_cat = df_deaths[df_deaths['eid'].isin(id_cat)]

    # get IDs of individuals in this subset with vs without vaccine 
    id_cat_vac = df_cat[df_cat['vaccinated']==True]['eid'].unique() # with vaccine
    id_cat_nvac = df_cat[df_cat['vaccinated']==False]['eid'].unique() # without vaccine

    # calculate the number of deaths per thousand people and save the result in a table
    df_sum_cat_deaths = incidence(df_deaths_cat, id_cat_vac, id_cat_nvac)
    
    # put in the values from the summary table
    df_sum_cat_all_deaths.loc[dict_cat[ic],:] = df_sum_cat_deaths.loc[:,'Number of events per thousand individuals']
    
    # contingency table
    crosstab_cat = contingency(df_deaths_cat, id_cat_vac, id_cat_nvac, index = dict_vacORnot)
    
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
df_sum_cat_all_deaths.to_csv('analysis/2c_total_mortality_table.csv')

# df with p-value, significance, RR and CI
df_combined_deaths.to_csv('analysis/2c_chi_square_RR_table.csv')

# plot
plt.figure(figsize = (8,8))

for v in dict_vacORnot:
    plt.barh(df_sum_cat_all_deaths.index, df_sum_cat_all_deaths[v], alpha = 0.5, label = v)

plt.title('Deaths in subjects with underlying diseases in specific ICD10 categories')
plt.xlabel('Number of deaths per thousand individuals')
plt.legend(loc = "upper right")
plt.savefig('analysis/2c_deaths_barplot.svg', bbox_inches='tight')


# plot the RR with CI

plt.figure(figsize = (6,6))
plt.xscale("log")
plt.axvline(1, ls='--', linewidth=1, color='black')

x = RR_result_cat_deaths['RR'].values
x_error = RR_result_cat_deaths['RR']-RR_result_cat_deaths['CI_lower']
y = np.arange(len(RR_result_cat_deaths))

plt.errorbar(x, y, xerr = x_error, marker = "o", markersize = 10, color = 'b', ls='none')

plt.xlabel('Risk Ratio (log scale)')
plt.title('Risk ratio of vaccination in people with underlying diseases in different ICD10 categories') # renée: Risk ratio of vaccination on an adverse event in subjects with underlying diseases in specific ICD-10 categories
plt.yticks(ticks = y, labels = RR_result_cat_deaths.index) 
plt.savefig('analysis/2c_deaths_RR.svg', bbox_inches='tight')


