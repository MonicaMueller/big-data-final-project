{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Introduction\n",
    "In hope of ending the global COVID-19 pandemic, multiple vaccines have been rapidly developed around the world. However, vaccine hesitancy due to safety concerns is hindering its usage. Phase 2 and 3 clinical trials showed promising results regarding both efficacy and adverse effects, also in people with underlying medical conditions [Choi, 2021].  \n",
    "In this project, the effect of two different vaccines, namely Pfizer-BioNTech and Oxford/AstraZeneca, on people in the UK with different underlying diseases will be analysed, aiming to shed more light on their safety. Specifically, the following questions will be answered: \n",
    "* How common are adverse events? Are they more common in people with (specific categories of) underlying medical conditions? Medical events with an ICD10 code after the second vaccine will be considered and compared to unvaccinated people in a similar time frame. \n",
    "* Is the mortality the same in vaccinated vs. non-vaccinated people? Is it the same when comparing vaccinated and non-vaccinated people with the same underlying diseases?\n",
    "\n",
    "Both analyses start more generally with the full population. The population to be analysed is then reduced to only people with underlying medical conditions and then to specific categories of underlying diseases according to ICD10.  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Methods"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "BnOF_K5b9eXs"
   },
   "source": [
    "## Import packages and data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import scipy as sp\n",
    "from scipy import stats\n",
    "import random\n",
    "random.seed(13)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Import data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "ED2_-Iv0DbIJ"
   },
   "outputs": [],
   "source": [
    "# need to import this file in slightly different way because of different delimiters\n",
    "clinical = pd.read_csv('/cluster/scratch/earaldi/BigData/Clinical_and_prescription_data/COVID_Vaccine/covid19_clinical.txt',sep='\\t|,', engine = 'python')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "zyYSNBpnypUf"
   },
   "outputs": [],
   "source": [
    "GPic10 = pd.read_csv('/cluster/scratch/bmonica/covid_analysis/GP_ic10_lookup.csv') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mortality = pd.read_csv('/cluster/scratch/earaldi/BigData/Clinical_and_prescription_data/COVID_Vaccine/patients_data_death.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mortality_coding = pd.read_csv('/cluster/scratch/earaldi/BigData/Clinical_and_prescription_data/COVID_Vaccine/patients_data_death_decoding.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Have a look at the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 424
    },
    "executionInfo": {
     "elapsed": 319,
     "status": "ok",
     "timestamp": 1635338237249,
     "user": {
      "displayName": "Monica Müller",
      "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64",
      "userId": "03925055125343928828"
     },
     "user_tz": -120
    },
    "id": "7svXJiX1F5PV",
    "outputId": "d5106cec-154c-46ed-8a33-e4764cd1b78d",
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# delete all of these ouputs \n",
    "\n",
    "clinical\n",
    "# GP dataset from patients in the database\n",
    "# each line corresponds to one incident with a date (event_dt), including vaccinations\n",
    "# eid: patient number \n",
    "# code_type: ?\n",
    "# value: measured value of a test or something similar"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 476
    },
    "executionInfo": {
     "elapsed": 11,
     "status": "ok",
     "timestamp": 1635338239808,
     "user": {
      "displayName": "Monica Müller",
      "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64",
      "userId": "03925055125343928828"
     },
     "user_tz": -120
    },
    "id": "QZo7OndFyvd_",
    "outputId": "78c836ec-1ff8-427f-baa7-52298ae33508"
   },
   "outputs": [],
   "source": [
    "GPic10\n",
    "#translates CTV3 codes (read_code) to ICD10 (icd10_code)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mortality\n",
    "# contains death dates and other info such as age and cause of death"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mortality_coding\n",
    "# coding of column names"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Preparing data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# drop unnecessary columns\n",
    "clinical.drop(columns = ['code_type', 'value'], inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# delete \n",
    "\n",
    "pd.isnull(clinical).sum()\n",
    "# no NaN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove duplicate entries \n",
    "clinical.drop_duplicates(inplace = True) \n",
    "mortality.drop_duplicates (inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Make 'event_date' in 'clinical' a datetime type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# convert entire event_dt in datetime format\n",
    "clinical['event_dt'] = pd.to_datetime(clinical['event_dt'])\n",
    "clinical"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Adding column with IC10 categories to clinical dataset for simpler analysis later on"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create a reduced dataframe for merging with the clinical dataframe\n",
    "GPic10_red =  GPic10[['read_code', 'icd10_code']]\n",
    "# rename to merge on the ctv3 column with the clinical dataset\n",
    "GPic10_red = GPic10_red.rename(columns = {'read_code':'code'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clinical_ic10 = pd.merge(clinical, GPic10_red, how='inner', on='code')\n",
    "clinical_ic10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove not considered clinical events in the clinical dataset (e.g. external factors)\n",
    "# remove O-Z, but leave R as adverse event & Q as underlying disease.\n",
    "\n",
    "# tuple of categories we do not consider as adverse event or underlying disease:\n",
    "exclude = ('O','P','S','T','U','V','W','X','Y','Z')\n",
    "\n",
    "# broadcasting the clinical dataset for all icd codes except those we exclude\n",
    "clinical_ic10 = clinical_ic10[~clinical_ic10['icd10_code'].str.startswith(exclude)]\n",
    "\n",
    "# delete\n",
    "clinical_ic10"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### NEW OPTION of filtering data by vaccination status"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# codes for covid vaccinations in the clinical dataset as found in the file 8708.tsv\n",
    "clinical[clinical['code']=='Y29e7'] # first dose\n",
    "clinical[clinical['code']=='Y29e8'] # second dose\n",
    "clinical[clinical['code']=='Y2a0e'] # SARS-2 Coronavirus vaccine\n",
    "clinical[clinical['code']=='Y210d'] # Severe acute respiratory syndrome coronavirus 2 vaccination (procedure)\n",
    "clinical[clinical['code']=='Y2eb5'] # COVID-19 Vaccine Novavax (adj) 5mcg/0.5ml dose susp for inj multidose vials (Baxter Oncology GmbH) part 1\n",
    "clinical[clinical['code']=='Y2eb5'] # COVID-19 Vaccine Novavax (adj) 5mcg/0.5ml dose susp for inj multidose vials (Baxter Oncology GmbH) part 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clinical_vaccines_only = clinical[clinical['code']=='Y29e7'] # first dose"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clinical_vaccines_only['issue_dt'] =  clinical_vaccines_only['event_dt']\n",
    "clinical_vaccines_only"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# finding eids in the clinical dataset that had first covid-19 dose with code 'Y29e7'\n",
    "id_vac = clinical[clinical['code']=='Y29e7']['eid'].unique()\n",
    "len(id_vac) # 961 eids with first dose"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "len(clinical['eid'].unique()) # 1277 of total eids in the dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Rename columns of mortality dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# make a dictionary out of the mortality_coding dataframe \n",
    "dict_columns = {}\n",
    "for index, value in enumerate(mortality_coding['Unnamed: 0']):\n",
    "  dict_columns[value] = mortality_coding.loc[index, \"description\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mortality.rename(columns = dict_columns, inplace = True) # columns = a dictionary will automatically rename the columns\n",
    "\n",
    "# delete\n",
    "mortality['DateDeath'].isna().sum()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# delete\n",
    "mortality"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Add deaths to clinical_ic10 df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# subset mortality df to the eids with a deathdate that is not NaN\n",
    "mortality_sub = mortality.dropna(subset=['DateDeath'])\n",
    "# drop all the unnecessary columns (also drop age and sex, perhaps no time to analyse this), keep eid, deathdate, cause\n",
    "mortality_sub.drop(columns = ['Sex', 'YearBirth', 'MonthBirth', 'DateRecruit', 'BMI', 'AgeDeath', 'AgeRecruit', 'cause of death 1', 'cause of death 2'], inplace = True)\n",
    "# rename column 'PatientID' to 'eid' and 'DateDeath' to 'event_dt'\n",
    "mortality_sub.rename(columns = {'PatientID':'eid', 'DateDeath':'event_dt'}, inplace = True) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# add column 'icd10_code' to mortality, filled with string 'death'\n",
    "mortality_sub['icd10_code'] = pd.Series([\"death\" for x in range(len(mortality.index))])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# make 'DateDeath' datetime type\n",
    "mortality_sub['event_dt'] = pd.to_datetime(mortality_sub['event_dt'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# delete\n",
    "mortality_sub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# now join them\n",
    "clinical_ic10= pd.concat([clinical_ic10, mortality_sub], join = 'outer')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# delete\n",
    "clinical_ic10 # 133687 events"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Add column with the date of vaccination to clinical_ic10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# drop event_dt in in clinical_vaccines only so that the issue date of vaccine and event date of other medical event is on the same row\n",
    "clinical_vaccines_only.drop(columns = ['event_dt'], inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# drop event_dt in in clinical_vaccines only so that the issue date of vaccine and event date of other medical event is on the same row\n",
    "clinical_vaccines_only.drop(columns = ['code'], inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# merge clinical_ic10 and clinical_vaccines_only to add the issue_date (date of vaccine) to the df\n",
    "clinical_vac = pd.merge(clinical_ic10, clinical_vaccines_only, how ='outer', on =['eid'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clinical_vaccines_only # 962 vaccinations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clinical_vac # 134649 events now in total = 133687 ic10 events + 962 vaccinations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "len(clinical_vac) \n",
    "# total amounts of events: vaccinations and medical events (incl. deaths) from clinical_ic10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "clinical_vac['issue_dt'].isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Add column in clinical_vac that says whether a patient is vaccinated or not"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mask_vaccinated = clinical_vac['eid'].isin(id_vac)\n",
    "clinical_vac['vaccinated'] = mask_vaccinated\n",
    "\n",
    "# check how many events of vaccinated people there are\n",
    "clinical_vac['vaccinated'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# delete\n",
    "clinical_vac \n",
    "# rows with icd10_code=NaN: vaccinated eids without medical events"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Add random issue date to events of unvaccinated patients to 'issue_date_format'  in clinical_vac "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# make list from issue_date in clinical_vaccines_only\n",
    "issue_dates = clinical_vaccines_only['issue_dt'].tolist()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 'eid' of unvaccinated people\n",
    "id_nvac = clinical_vac[clinical_vac['vaccinated']==False]['eid'].unique()\n",
    "\n",
    "# delete\n",
    "len(id_nvac)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create list of new dates\n",
    "issue_dt_random = random.choices(issue_dates, k = len(id_nvac))\n",
    "\n",
    "# delete\n",
    "len(issue_dt_random)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# make dictionary with random issue dates and unvaccinated eid\n",
    "dict_nvac = dict(zip(id_nvac,issue_dt_random))\n",
    "\n",
    "# delete\n",
    "len(dict_nvac)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clinical_vac['issue_dt']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# fill na with mapping from the dictionary\n",
    "clinical_vac['issue_dt'] = clinical_vac['issue_dt'].fillna(clinical_vac['eid'].map(dict_nvac))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# delete\n",
    "\n",
    "clinical_vac['issue_dt'][clinical_vac['vaccinated']==True].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# delete\n",
    "\n",
    "clinical_vac['issue_dt'][clinical_vac['vaccinated']==False].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# delete\n",
    "\n",
    "clinical_vac[clinical_vac['vaccinated'] == False]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# delete \n",
    "clinical_vac['issue_dt'].isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Add column where it says if the event (event_date_format) was before or after the vaccination (issue_date)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clinical_vac['days_after_vaccine'] = clinical_vac['event_dt']-clinical_vac['issue_dt']\n",
    "clinical_vac"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "conditions = [clinical_vac['event_dt']>=clinical_vac['issue_dt'], clinical_vac['event_dt']<clinical_vac['issue_dt']]\n",
    "choices = ['after', 'before']\n",
    "clinical_vac['before_after_vaccine'] = np.select(conditions, choices, default = 'before')\n",
    "clinical_vac"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# old way\n",
    "# before_after_vaccine = []\n",
    "# for i in range (len(clinical_vac)):\n",
    "#    if clinical_vac['event_dt'].iloc[i] >= clinical_vac['issue_dt'].iloc[i]:\n",
    "#        before_after_vaccine.append('after')\n",
    "#    else:\n",
    "#        before_after_vaccine.append('before')\n",
    "#clinical_vac['before_after_vaccine'] = before_after_vaccine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# delete\n",
    "clinical_vac"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# delete\n",
    "\n",
    "# checking how many events before and after\n",
    "clinical_vac['before_after_vaccine'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1) Effect of the vaccination on adverse events"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "a) Comparison of occurence of adverse events after vaccination of vaccinated and unvaccinated people\n",
    "\n",
    "b) Same comparison in subset of subjects with underlying medical condition\n",
    "\n",
    "c) Same comparison in subset of subjects with specific underlying medical condition "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1a) Comparison of occurence of adverse events after vaccination of vaccinated and unvaccinated people"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# initialize index names\n",
    "dict_vacORnot = {'Vaccinated':1, 'Not vaccinated':0} #vaccination status"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Total number of adverse events \n",
    "Adverse events = medical events with ICD10 code after the (artificial) vaccination date\n",
    "\n",
    "Total number of adverse events: (multiple per patient in some cases) vaccinated vs. non-vaccinated"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def incidence(df_event, id_vac, id_nvac):\n",
    "    \"\"\"\n",
    "    Calculates the number of events (AEs/deaths) per thousand individuals in a given population.\n",
    "    \n",
    "    Input:\n",
    "    - df_event: a dataframe containing all events after the issue date considered in the analysis (AEs/deaths)\n",
    "    - id_vac: series containing ID's of each individual (with each individual listed once) within the given population. // only vaccinated!\n",
    "    - id_nvac: same as id_vac but with all non-vaccinated individuals\n",
    "    \n",
    "    Note: this is not exactly the incidence (would be number of events in specified time, which would be nice to analyse in the future)\n",
    "    \n",
    "    \"\"\"\n",
    "    \n",
    "    # initialize df for putting in the values\n",
    "    df_sum = pd.DataFrame(columns = ['Number of events', 'Number of individuals', 'Number of events per thousand individuals'], index = dict_vacORnot)\n",
    "    \n",
    "    for v in dict_vacORnot: # loop through vaccinated vs. not\n",
    "        \n",
    "        # put in the number of events (AEs) by subsetting the df_event into vaccinated/not vaccinated\n",
    "        df_sum['Number of events'][v] = len(df_event[df_event['vaccinated'] == dict_vacORnot[v]])\n",
    "    \n",
    "    \n",
    "    # put number of vaccinated/not vaccinated people in the population into the summary df\n",
    "    df_sum['Number of individuals']['Vaccinated'] = len(id_vac) # vaccinated (row)\n",
    "    df_sum['Number of individuals']['Not vaccinated'] = len(id_nvac) # not vaccinated (row)\n",
    "    \n",
    "    # calculate AEs/individual\n",
    "    df_sum['Number of events per thousand individuals'] = df_sum['Number of events']/df_sum['Number of individuals']*1000\n",
    "    \n",
    "    return df_sum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make a df with only AEs (defined medical events w. ICD10 code after the (artificial) vaccination date)\n",
    "# reminder: leave R as adverse event & Q as underlying disease.\n",
    "\n",
    "# subset of clinical dataset for adverse events (=after vaccine)\n",
    "df_AEs = clinical_vac[clinical_vac['before_after_vaccine']== 'after']\n",
    "# drop NaN (from lines with vaccine)\n",
    "df_AEs.dropna(subset = ['icd10_code'], inplace = True)\n",
    "\n",
    "# remove Q (congenital diseases), which are not considered as adverse events\n",
    "df_AEs = df_AEs[~df_AEs['icd10_code'].str.startswith('Q')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove deaths, which are not considered adverse events\n",
    "df_AEs = df_AEs[df_AEs['icd10_code'] != 'death'] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# get number of people of the chosen population\n",
    "# list of all IDs from the clinical dataset\n",
    "id_clin = clinical['eid'].unique()\n",
    "\n",
    "# list of all IDs from the mortality dataset\n",
    "id_mor = mortality['PatientID'].unique()\n",
    "\n",
    "# concatenate both lists to get all IDs in one list\n",
    "id_all = np.concatenate((id_clin, id_mor))\n",
    "\n",
    "# leave only unique IDs in the list to get the number of IDs using len()\n",
    "id_all = np.unique(id_all)\n",
    "\n",
    "# get array with IDs without vaccine by deleting the IDs with vaccine out of the array with all IDs\n",
    "id_nvac = np.delete(id_all, np.isin(id_all,id_vac))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# use the function to create the summary table\n",
    "df_sum_all = incidence(df_AEs, id_vac, id_nvac)\n",
    "\n",
    "# save \n",
    "df_sum_all.to_csv('analysis/1a_total_AE_table.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Plot how oftern AEs occur in vaccinated vs. not vaccinated people for all individuals in the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize = (5,5))\n",
    "plt.bar(df_sum_all.index, df_sum_all['Number of events per thousand individuals'])\n",
    "plt.ylabel('Number of AEs per thousand individuals')\n",
    "plt.title('Adverse events in the whole population of the dataset')\n",
    "\n",
    "# save figure in vector format, bbox_inches='tight' so that the plot is not cropped\n",
    "plt.savefig('analysis/1a_total_AE_barplot.svg', bbox_inches='tight')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Incidence of AEs in all subjects\n",
    "In further analysis: number of eids with AE is analysed (not total number of AEs occurring)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Make contingency table (not using pd.crosstab to avoid another huge dataset with all individuals and because we need to count unique IDs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def contingency(df_event, id_vac, id_nvac, index = dict_vacORnot):\n",
    "    \"\"\"\n",
    "    Makes a 2x2 contingency table, which can be used for the chi^2 test and calculating the risk ratio.\n",
    "    \n",
    "    Input:\n",
    "    - df_event: a dataframe containing all events (AEs/deaths) considered in the analysis\n",
    "    - id_vac: series containing ID's of each individual (with each individual listed once) within the given population.\n",
    "    - id_nvac: same as id_vac but with all non-vaccinated individuals\n",
    "    \n",
    "    \"\"\"\n",
    "\n",
    "    # initialize contingency table\n",
    "    crosstab = pd.DataFrame(columns = ['event', 'no event'], index = index)\n",
    "    \n",
    "    # fill contingency table with values in the subset\n",
    "\n",
    "    for v in dict_vacORnot: # loop throug vaccinated vs. not\n",
    "\n",
    "        \n",
    "        # with events\n",
    "\n",
    "        # number of nonvaccinated/vaccinated people with event = number of individuals with AEs\n",
    "        crosstab.loc[v,'event'] = len(df_event[df_event['vaccinated']==dict_vacORnot[v]]['eid'].unique())\n",
    "\n",
    "            \n",
    "        # without events\n",
    "\n",
    "        if v == 'Vaccinated': # for vaccinated people\n",
    "\n",
    "            # total number of vaccinated individuals - those with events\n",
    "            crosstab.loc[v,'no event'] = len(id_vac) - crosstab.loc[v,'event']\n",
    "\n",
    "        else: # for not vaccinated people\n",
    "            \n",
    "            # total number of unvaccinated individuals - those with events\n",
    "            crosstab.loc[v,'no event'] = len(id_nvac) - crosstab.loc[v,'event']\n",
    "            \n",
    "    \n",
    "    # adding margins to the contingency table (=total value as row and column)\n",
    "    \n",
    "    # vaccinated vs. not\n",
    "    crosstab.loc['Total',:] = np.sum(crosstab, axis = 0)\n",
    "    # event vs. no event\n",
    "    crosstab.loc[:,'Total'] = np.sum(crosstab, axis = 1)\n",
    "\n",
    "    return crosstab"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "crosstab_all = contingency(df_AEs, id_vac, id_nvac)\n",
    "\n",
    "# save\n",
    "crosstab_all.to_csv('analysis/1a_contigency_table.csv')\n",
    "\n",
    "crosstab_all"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Chi-squared test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# define significance treshold\n",
    "alpha = 0.05\n",
    "\n",
    "# perform chi-squared test:\n",
    "stat_1a, p_1a, dof_1a, expected_1a = sp.stats.chi2_contingency(crosstab_all.iloc[0:2,0:2], 1)\n",
    "\n",
    "#save \n",
    "chi_square_1a = pd.DataFrame([[stat_1a, p_1a, dof_1a, expected_1a]],columns=['stat_1a', 'p_1a', 'dof_1a', 'expected_1a'])\n",
    "chi_square_1a.to_csv('analysis/1a_chi_square.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Risk ratio \n",
    "Done similar to Barda et al., but se and CI calculated as described on https://en.wikipedia.org/wiki/Relative_risk for simplicity. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def riskratio(crosstab, z = 1.96):\n",
    "    \"\"\"\n",
    "    Calculates the risk ratio.\n",
    "    \n",
    "    Returns the risk ratio with the confidence interval (CI) as a df and the adjusted contingency table:\n",
    "    (RR, crosstab)\n",
    "    \n",
    "    Important: the input (crosstab) needs to have the following format:\n",
    "    - 2x2\n",
    "    - column 0 = events (e.g. AEs, deaths), column 1 = non-events\n",
    "    - row 0 = intervention (e.g. vaccinated), column 1 = non-vaccinated\n",
    "    \"\"\"\n",
    "    \n",
    "    # calculating the risk ratio\n",
    "\n",
    "    # risk of getting AE/ dying when vaccinated\n",
    "    R_vac = crosstab.loc['Vaccinated','event']/crosstab.loc['Vaccinated','Total']\n",
    "    # risk of getting \"AE\" or dying when not vaccinated\n",
    "    R_nvac = crosstab.loc['Not vaccinated', 'event']/crosstab.loc['Not vaccinated', 'Total']\n",
    "\n",
    "    # risk ratio\n",
    "    RR = R_vac / R_nvac\n",
    "    \n",
    "    # natural log of the RR     \n",
    "    logRR = np.log(RR) \n",
    "\n",
    "    # standard error\n",
    "    se = np.sqrt(crosstab.loc['Vaccinated','no event']/ \\\n",
    "                 (crosstab.loc['Vaccinated','event']*crosstab.loc['Vaccinated','Total']) + \\\n",
    "                crosstab.loc['Not vaccinated','no event']/ \\\n",
    "                (crosstab.loc['Not vaccinated','event']*crosstab.loc['Not vaccinated','Total']))\n",
    "\n",
    "    # Assuming normal distribution, 95% lie around 1.96 standard deviations of the mean.\n",
    "    # as 95% CI is common, the z-score (z) 1.96 is put as the default value.\n",
    "\n",
    "    # 1-alpha confidence interval (CI) \n",
    "    CI_lower = np.exp(np.log(RR) - se * z) # lower bound\n",
    "    CI_upper = np.exp(np.log(RR) + se * z) # upper bound    \n",
    "    \n",
    "    # put the result in a series\n",
    "    RR_result = pd.Series(data = [RR, CI_lower, CI_upper], index = ['RR', 'CI_lower', 'CI_upper'])\n",
    "    \n",
    "    return RR_result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "RR_all = riskratio(crosstab_all)\n",
    "\n",
    "# will later be saved after 1b "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1b) Same comparison in subset of subjects with underlying medical condition\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Subset df to eids with AEs and underlying disease"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# get dataset with only AEs AND people with underlying disease \n",
    "# = adverse events (events after vaccination) of people with underlying disease (= that had an event before vaccination)\n",
    "\n",
    "# subset of the df with all AEs (df_AEs) with only people with underlying diseases (df_AEs_ud)\n",
    "# reminder: leave R as adverse event (and general underlying disease) & Q as underlying disease.\n",
    "\n",
    "# subset of clinical_vac df with only medical events before\n",
    "df_ud= clinical_vac[clinical_vac['before_after_vaccine']== 'before']\n",
    "\n",
    "# drop rows without icd10_code\n",
    "df_ud.dropna(subset = ['icd10_code'], inplace = True)\n",
    "\n",
    "# remove deaths to not consider people that died before\n",
    "df_ud = df_ud[df_ud['icd10_code'] != 'death'] \n",
    "\n",
    "# ICD code R is kept in this analysis of general underlying diseases\n",
    "\n",
    "# get id's of people with an underlying disease\n",
    "id_ud = df_ud['eid'].unique()\n",
    "\n",
    "# subset of the df with AEs with only people with an underlying disease\n",
    "df_AEs_ud = df_AEs[df_AEs['eid'].isin(id_ud)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# individuals in subset with only people with underlying disease AND the vaccine\n",
    "id_ud_vac = df_ud[df_ud['vaccinated']==True]['eid'].unique()\n",
    "\n",
    "# individuals in subset with only people with underlying disease AND NOT the vaccine\n",
    "id_ud_nvac = df_ud[df_ud['vaccinated']==False]['eid'].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Incidence of AE in subjects with underlying disease"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_sum_ud = incidence(df_AEs_ud, id_ud_vac, id_ud_nvac)\n",
    "\n",
    "# save\n",
    "df_sum_ud.to_csv('analysis/1b_total_AE_table.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize = (5,5))\n",
    "plt.bar(df_sum_ud.index, df_sum_ud['Number of events per thousand individuals'])\n",
    "plt.ylabel('Number of AEs per thousand individuals')\n",
    "plt.title('Adverse events in subjects with underlying diseases')\n",
    "\n",
    "# save figure in vector format, bbox_inches='tight' so that the plot is not cropped\n",
    "plt.savefig('analysis/1b_total_AE_barplot.svg', bbox_inches='tight')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Chi-squared test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# contingency table\n",
    "crosstab_ud = contingency(df_AEs_ud, id_ud_vac, id_ud_nvac, index = dict_vacORnot)\n",
    "crosstab_ud.to_csv('analysis/1b_contigency_table.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# define significance treshold\n",
    "alpha = 0.05\n",
    "\n",
    "# perform chi-squared test:\n",
    "stat_1b, p_1b, dof_1b, expected_1b = sp.stats.chi2_contingency(crosstab_ud.iloc[0:2,0:2], 1)\n",
    "\n",
    "# save\n",
    "chi_square_1b = pd.DataFrame([[stat_1b, p_1b, dof_1b, expected_1b]],columns=['stat_1b', 'p_1b', 'dof_1b', 'expected_1b'])\n",
    "chi_square_1b.to_csv('analysis/1b_chi_square.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Risk ratio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "RR_ud = riskratio(crosstab_ud)\n",
    "RR_ud\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create a df for collecting the results of the risk ratio to compare the populations\n",
    "\n",
    "# initialize\n",
    "RR_result = pd.DataFrame(index = ['whole population', 'people with underlying diseases'], columns = ['RR', 'CI_lower', 'CI_upper'])\n",
    "\n",
    "# fill it\n",
    "# RR with data from all individuals in the mortality & clinical dataset\n",
    "RR_result.loc['whole population',:] = RR_all\n",
    "# RR with data from all individuals with underlying diseases\n",
    "RR_result.loc['people with underlying diseases',:] = RR_ud\n",
    "\n",
    "RR_result\n",
    "\n",
    "# save \n",
    "pd.DataFrame(RR_result).to_csv('analysis/1a_b_RR_table.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# plot the difference in RR with CI\n",
    "\n",
    "# general figure configuration\n",
    "plt.figure(figsize = (6,6))\n",
    "plt.xscale(\"log\")\n",
    "plt.axvline(1, ls='--', linewidth=1, color='black')\n",
    "\n",
    "# draw the RR\n",
    "# RR results\n",
    "x = RR_result['RR'].values\n",
    "# error bar with same size in both directions\n",
    "x_error = RR_result['RR']-RR_result['CI_lower']\n",
    "# for equal spacing of the results\n",
    "y = np.arange(len(RR_result))\n",
    "\n",
    "# the plot\n",
    "plt.errorbar(x, y, xerr = x_error, marker = \"o\", markersize = 10, color = 'b', ls='none')\n",
    "\n",
    "# labelling\n",
    "plt.xlabel('Risk Ratio (log scale)')\n",
    "plt.title('Risk ratio of vaccination in people with underlying diseases or the general population') # renee: Risk ratio of vaccination on an adverse event in subjects with underlying diseases and the entire population\n",
    "# labelling with ICD10 categories\n",
    "plt.yticks(ticks = y, labels = RR_result.index) \n",
    "plt.ylim(-0.5,1.5)\n",
    "\n",
    "# save figure in vector format, bbox_inches='tight' so that the plot is not cropped\n",
    "plt.savefig('analysis/1a_b_RR_plot.svg', bbox_inches='tight')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1c) Same comparison in subset of subjects with specific underlying medical condition "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Loop through the ICD-10 categories"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7Bw4hTHD2Rg8"
   },
   "source": [
    "ICD10 codes are structured. The following website was used for looking up the meanings of the categories (e.g. respiratory diseases):\n",
    "https://www.icd10data.com/ICD10CM/Codes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# dictionary of disease categories to analyse, from https://www.icd10data.com/ICD10CM/Codes\n",
    "dict_cat = {('A','B'):'Certain infectious and parasitic diseases',\n",
    "           ('C','D0','D1','D2','D3','D4'):'Neoplasms',\n",
    "           ('D5','D6','D7','D8'):'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',\n",
    "           'E':'Endocrine, nutritional and metabolic diseases',\n",
    "           'F':'Mental, Behavioral and Neurodevelopmental disorders',\n",
    "           'G':'Diseases of the nervous system',\n",
    "           ('H0','H1','H2','H3','H4','H5'):'Diseases of the eye and adnexa',\n",
    "           ('H6','H7','H8', 'H9'): 'Diseases of the ear and mastoid process',\n",
    "           'I':'Diseases of the circulatory system',\n",
    "           'J':'Diseases of the respiratory system',\n",
    "           'K':'Diseases of the digestive system',\n",
    "           'L':'Diseases of the skin and subcutaneous tissue',\n",
    "           'M':'Diseases of the musculoskeletal system and connective tissue',\n",
    "           'N':'Diseases of the genitourinary system',\n",
    "           'Q':'Congenital malformations, deformations and chromosomal abnormalities'}\n",
    "\n",
    "# As mentioned by Barda et al., adjustment for multiple comparisons is not commonly done in studies regarding safety.\n",
    "alpha = 0.05\n",
    "\n",
    "# initialize \n",
    "# dataframe for collecting number of AEs per thousands with/without vaccinated people for all categories\n",
    "df_sum_cat_all = pd.DataFrame(columns = dict_vacORnot, index=dict_cat.values()) \n",
    "\n",
    "# df for saving categories & their p-values\n",
    "#df_chi2_cat = pd.DataFrame(columns = ['p-value', 'significant'], index=dict_cat.values())\n",
    "\n",
    "# df for saving the risk ratio and CIs for the plot\n",
    "RR_result_cat = pd.DataFrame(index = dict_cat.values(), columns = ['RR', 'CI_lower', 'CI_upper'])\n",
    "\n",
    "# df for saving categories and their p-value, significance, RR and CI\n",
    "df_combined = pd.DataFrame(columns = ['P-value', 'Significant', 'Risk Ratio', 'Confidence Interval'], index = dict_cat.values())\n",
    "\n",
    "\n",
    "\n",
    "for i, ic in enumerate(dict_cat): # loop over ICD10 codes of each disease category\n",
    "    \n",
    "    # get dataset with only AEs AND people with specific underlying disease\n",
    "    # subset of the df with all AEs (df_AEs) with only people with specific underlying diseases\n",
    "    # reminder: leave R as adverse event & Q as underlying disease.\n",
    "    # df_ud = subset of clinical_vac df with only diagnoses before vaccine as defined further above\n",
    "\n",
    "    # subset with only underlying diseases within the specific category\n",
    "    df_cat = df_ud[df_ud['icd10_code'].str.startswith(ic)]\n",
    "\n",
    "    # IDs with underlying disease in the specific category\n",
    "    id_cat = df_cat['eid'].unique()\n",
    "\n",
    "    # subset of the df with AEs with only people with underlying disease in the specific category\n",
    "    df_AEs_cat = df_AEs[df_AEs['eid'].isin(id_cat)]\n",
    "\n",
    "    # get IDs of individuals in this subset with vs without vaccine \n",
    "    id_cat_vac = df_cat[df_cat['vaccinated']==True]['eid'].unique() # with vaccine\n",
    "    id_cat_nvac = df_cat[df_cat['vaccinated']==False]['eid'].unique() # without vaccine\n",
    "\n",
    "    ################################################################################\n",
    "    # calculate the number of AEs per thousand people and save the result in a table\n",
    "    df_sum_cat = incidence(df_AEs_cat, id_cat_vac, id_cat_nvac)\n",
    "    \n",
    "    # put in the values from the summary incidence table\n",
    "    df_sum_cat_all.loc[dict_cat[ic],:] = df_sum_cat.loc[:, 'Number of events per thousand individuals']\n",
    "    \n",
    "    \n",
    "    ################################################################################    \n",
    "    # contingency table\n",
    "    crosstab_cat = contingency(df_AEs_cat, id_cat_vac, id_cat_nvac, index = dict_vacORnot)\n",
    "    \n",
    "    ###############################################################################\n",
    "    # perform chi-squared test:\n",
    "    stat, p, dof, expected = sp.stats.chi2_contingency(crosstab_cat.iloc[0:2,0:2], 1)     ## changed crosstab_ud to crosstab_cat\n",
    "\n",
    "    # save the p-value in the df\n",
    "    df_combined.loc[dict_cat[ic],'P-value'] = p\n",
    "    \n",
    "    # mark in which categories there is a significant difference between vaccinated and not vaccinated\n",
    "    if p <= alpha/2: # divided by two because of two-sided test (number of AEs could be higher or lower in vaccinated)\n",
    "        df_combined.loc[dict_cat[ic],'Significant'] = 'Yes'\n",
    "    else:\n",
    "        df_combined.loc[dict_cat[ic],'Significant'] = 'No'\n",
    "       \n",
    "    \n",
    "    ###############################################################################\n",
    "    # risk ratio\n",
    "    RR_cat = riskratio(crosstab_cat)\n",
    "\n",
    "    # fill the df with the RR results for the plot\n",
    "    RR_result_cat.loc[dict_cat[ic],:] = RR_cat\n",
    "    \n",
    "    CI_rounded = (round(RR_cat[1], 2), round(RR_cat[2], 2)) # round to one more decimal place than the original data has\n",
    "    \n",
    "    # new fill df with RR\n",
    "    df_combined.loc[dict_cat[ic], ['Risk Ratio', 'Confidence Interval']] = round(RR_cat[0],2) , CI_rounded\n",
    "    \n",
    "    \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Total number of AEs in IC-10 categories"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# total number of AEs in different categories\n",
    "df_sum_cat_all.sort_values(by = 'Vaccinated', ascending = False, inplace = True) # sorting\n",
    "\n",
    "# save\n",
    "df_sum_cat_all.to_csv('analysis/1c_total_AE_table.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Outcomes of Chi squared test and RR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df with p-value, significance, RR and CI\n",
    "df_combined.to_csv('analysis/1c_chi_square_RR_table.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize = (8,8))\n",
    "\n",
    "# plot results for vaccinated and not on top of each other\n",
    "for v in dict_vacORnot:\n",
    "    plt.barh(df_sum_cat_all.index, df_sum_cat_all[v], alpha = 0.5, label = v)\n",
    "\n",
    "# labelling\n",
    "plt.title('Adverse events in people with diseases in specific ICD10 categories')\n",
    "plt.xlabel('Number of AEs per thousand individuals')\n",
    "plt.legend(loc = \"upper right\")\n",
    "\n",
    "# save figure in vector format, bbox_inches='tight' so that the plot is not cropped\n",
    "plt.savefig('analysis/1c_AE_barplot.svg', bbox_inches='tight')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# plot the RR with CI\n",
    "\n",
    "# general figure configuration\n",
    "plt.figure(figsize = (6,6))\n",
    "plt.xscale(\"log\")\n",
    "plt.axvline(1, ls='--', linewidth=1, color='black')\n",
    "\n",
    "# draw the RR\n",
    "# RR results\n",
    "x = RR_result_cat['RR'].values\n",
    "# error bar with same size in both directions\n",
    "x_error = RR_result_cat['RR']-RR_result_cat['CI_lower']\n",
    "# for equal spacing of the results\n",
    "y = np.arange(len(RR_result_cat))\n",
    "\n",
    "# the plot\n",
    "plt.errorbar(x, y, xerr = x_error, marker = \"o\", markersize = 10, color = 'b', ls='none')\n",
    "\n",
    "# labelling\n",
    "plt.xlabel('Risk Ratio (log scale)')\n",
    "plt.title('Risk ratio of vaccination in people with underlying diseases in different ICD10 categories')\n",
    "# labelling with ICD10 categories\n",
    "plt.yticks(ticks = y, labels = RR_result_cat.index) \n",
    "\n",
    "# save figure in vector format, bbox_inches='tight' so that the plot is not cropped\n",
    "plt.savefig('analysis/1c_AE_RR.svg', bbox_inches='tight')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2) Effect of the vaccination on mortality"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "a) Comparison of mortality rate after vaccination of vaccinated and unvaccinated people\n",
    "\n",
    "b) Same comparison in subset of subjects with underlying medical condition\n",
    "\n",
    "c) Same comparison in subset of subjects with specific underlying medical condition \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2a) Comparison of mortality rate after vaccination of vaccinated and unvaccinated people"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Subset clinical df to only death entries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# subset of clinical_vac dataset for adverse events (=after vaccine)\n",
    "df_deaths = clinical_vac[clinical_vac['before_after_vaccine']== 'after']\n",
    "\n",
    "# make dataframe with only deaths \n",
    "df_deaths = df_deaths[df_deaths['icd10_code'] == 'death'] "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Mortality in entire study population"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Make contingency table (not using pd.crosstab to avoid another huge dataset with all individuals and because we need to count unique IDs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "crosstab_all_deaths = contingency(df_deaths, id_vac, id_nvac)\n",
    "\n",
    "# save\n",
    "crosstab_all_deaths.to_csv('analysis/2a_contigency_table.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Chi-squared test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# perform chi-squared test and name variavles differently for each test\n",
    "# e.g. 1a, 1b, etc.\n",
    "stat_2a, p_2a, dof_2a, expected_2a = sp.stats.chi2_contingency(crosstab_all_deaths.iloc[0:2,0:2], 1)\n",
    "\n",
    "# save\n",
    "chi_square_2a = pd.DataFrame([[stat_2a, p_2a, dof_2a, expected_2a]],columns=['stat_2a', 'p_2a', 'dof_2a', 'expected_2a'])\n",
    "chi_square_2a.to_csv('analysis/2a_chi_square.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Risk ratio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "RR_all_deaths = riskratio(crosstab_all_deaths)\n",
    "\n",
    "# save together with 2b"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2b) Same comparison in subset of subjects with underlying medical condition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# subset of the df with deaths with only people with an underlying disease - non in this dataset\n",
    "df_deaths_ud = df_deaths[df_deaths['eid'].isin(id_ud)]\n",
    "\n",
    "#delete\n",
    "df_deaths_ud\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# should we keep this output?\n",
    "\n",
    "# also here is total number of deaths not needed, as it is the same as later seen in the crosstable.\n",
    "df_sum_ud = incidence(df_deaths_ud, id_ud_vac, id_ud_nvac)\n",
    "df_sum_ud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# delete?\n",
    "plt.figure(figsize = (5,5))\n",
    "plt.bar(df_sum_ud.index, df_sum_ud['Number of events per thousand individuals'])\n",
    "plt.ylabel('Number of deaths per thousand individuals')\n",
    "plt.title('Deaths in people with underlying diseases')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Chi-squared test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# contingency table\n",
    "crosstab_ud_deaths = contingency(df_deaths_ud, id_ud_vac, id_ud_nvac, index = dict_vacORnot)\n",
    "\n",
    "# save\n",
    "crosstab_ud_deaths.to_csv('analysis/2b_contigency_table.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# perform chi-squared test:\n",
    "stat_2b, p_2b, dof_2b, expected_2b = sp.stats.chi2_contingency(crosstab_ud_deaths.iloc[0:2,0:2], 1)\n",
    "\n",
    "# save\n",
    "chi_square_2b = pd.DataFrame([[stat_2a, p_2a, dof_2a, expected_2a]],columns=['stat_2a', 'p_2a', 'dof_2a', 'expected_2a'])\n",
    "chi_square_2b.to_csv('analysis/2b_chi_square.csv')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Calculate and plot risk ratio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "RR_ud_deaths = riskratio(crosstab_ud_deaths)\n",
    "\n",
    "# delete\n",
    "RR_ud_deaths"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create a df for collecting the results of the risk ratio to compare the populations\n",
    "\n",
    "# initialize\n",
    "RR_result_deaths = pd.DataFrame(index = ['Whole population', 'People with underlying diseases'], columns = ['RR', 'CI_lower', 'CI_upper'])\n",
    "\n",
    "# fill it\n",
    "# RR with data from all individuals in the mortality & clinical dataset\n",
    "RR_result_deaths.loc['Whole population',:] = RR_all_deaths\n",
    "# RR with data from all individuals with underlying diseases\n",
    "RR_result_deaths.loc['People with underlying diseases',:] = RR_ud_deaths\n",
    "\n",
    "# save\n",
    "pd.DataFrame(RR_result_deaths).to_csv('analysis/1a_b_RR.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "# plot the difference in RR with CI\n",
    "\n",
    "# general figure configuration\n",
    "plt.figure(figsize = (6,6))\n",
    "plt.xscale(\"log\")\n",
    "plt.axvline(1, ls='--', linewidth=1, color='black')\n",
    "\n",
    "# draw the RR\n",
    "# RR results\n",
    "x = RR_result_deaths['RR'].values\n",
    "# error bar with same size in both directions\n",
    "x_error = RR_result_deaths['RR']-RR_result_deaths['CI_lower']\n",
    "# for equal spacing of the results\n",
    "y = np.arange(len(RR_result_deaths))\n",
    "\n",
    "# the plot\n",
    "plt.errorbar(x, y, xerr = x_error, marker = \"o\", markersize = 10, color = 'b', ls='none')\n",
    "\n",
    "# labelling\n",
    "plt.xlabel('Risk Ratio (log scale)')\n",
    "plt.title('Risk ratio of vaccination on death in subjects with underlying diseases or the general population') \n",
    "# labelling with ICD10 categories\n",
    "plt.yticks(ticks = y, labels = RR_result_deaths.index) \n",
    "plt.ylim(-0.5,1.5)\n",
    "\n",
    "# save figure in vector format, bbox_inches='tight' so that the plot is not cropped\n",
    "plt.savefig('analysis/2a_b_RR_plot.svg', bbox_inches='tight')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2c) Same comparison in subset of subjects with specific underlying medical condition "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# initialize \n",
    "\n",
    "# dataframe for collecting number of AEs per thousands with/without vaccinated people for all categories\n",
    "df_sum_cat_all_deaths = pd.DataFrame(columns = dict_vacORnot, index=dict_cat.values()) \n",
    "\n",
    "# df for saving the risk ratio and CI\n",
    "RR_result_cat_deaths = pd.DataFrame(index = dict_cat.values(), columns = ['RR', 'CI_lower', 'CI_upper'])\n",
    "\n",
    "# df for saving categories and their p-value, significance, RR and CI\n",
    "df_combined_deaths = pd.DataFrame(columns = ['P-value', 'Significant', 'Risk Ratio', 'Confidence Interval'], index = dict_cat.values())\n",
    "\n",
    "\n",
    "for i, ic in enumerate(dict_cat): # loop over ICD10 codes of each disease category\n",
    "    \n",
    "    # get dataset with only deaths AND people with specific underlying disease\n",
    "    # subset of the df with all deaths (df_deaths) with only people with specific underlying diseases\n",
    "    # reminder: leave R as adverse event & Q as underlying disease.\n",
    "    # df_ud = subset of clinical_vac df with only diagnoses before vaccine as defined further above\n",
    "\n",
    "    # subset with only underlying diseases within the specific category (=ic)\n",
    "    df_cat = df_ud[df_ud['icd10_code'].str.startswith(ic)]\n",
    "\n",
    "    # IDs with underlying disease in the specific category\n",
    "    id_cat = df_cat['eid'].unique()\n",
    "\n",
    "    # subset of the df with deaths after vaccination only in people with underlying disease in the specific category\n",
    "    df_deaths_cat = df_deaths[df_deaths['eid'].isin(id_cat)]\n",
    "\n",
    "    # get IDs of individuals in this subset with vs without vaccine \n",
    "    id_cat_vac = df_cat[df_cat['vaccinated']==True]['eid'].unique() # with vaccine\n",
    "    id_cat_nvac = df_cat[df_cat['vaccinated']==False]['eid'].unique() # without vaccine\n",
    "\n",
    "    ################################################################################\n",
    "    # calculate the number of deaths per thousand people and save the result in a table\n",
    "    df_sum_cat_deaths = incidence(df_deaths_cat, id_cat_vac, id_cat_nvac)\n",
    "    \n",
    "    # put in the values from the summary table\n",
    "    df_sum_cat_all_deaths.loc[dict_cat[ic],:] = df_sum_cat_deaths.loc[:,'Number of events per thousand individuals']\n",
    "    \n",
    "    \n",
    "    ################################################################################    \n",
    "    # contingency table\n",
    "    crosstab_cat = contingency(df_deaths_cat, id_cat_vac, id_cat_nvac, index = dict_vacORnot)\n",
    "    \n",
    "    ###############################################################################\n",
    "    # perform chi-squared test:\n",
    "    stat, p, dof, expected = sp.stats.chi2_contingency(crosstab_cat.iloc[0:2,0:2], 1)       # changed to crosstab_cat from crosstab_ud\n",
    "    \n",
    "    # save the p-value in the df\n",
    "    df_combined_deaths.loc[dict_cat[ic],'P-value'] = p\n",
    "    \n",
    "    # mark in which categories there is a significant difference between vaccinated and not vaccinated\n",
    "    if p <= alpha/2: # divided by two because of two-sided test (number of AEs could be higher or lower in vaccinated)\n",
    "        df_combined_deaths.loc[dict_cat[ic],'Significant'] = 'Yes'\n",
    "    else:\n",
    "        df_combined_deaths.loc[dict_cat[ic],'Significant'] = 'No'\n",
    "       \n",
    "    ###############################################################################\n",
    "    # risk ratio\n",
    "    RR_cat = riskratio(crosstab_cat)\n",
    "\n",
    "    # fill the df with the RR results for the plot later\n",
    "    RR_result_cat_deaths.loc[dict_cat[ic],:] = RR_cat\n",
    "    \n",
    "    CI_rounded = (round(RR_cat[1], 2), round(RR_cat[2], 2)) # round to one more decimal place than the original data has\n",
    "    \n",
    "    # fill df with RR\n",
    "    df_combined_deaths.loc[dict_cat[ic], ['Risk Ratio', 'Confidence Interval']] = round(RR_cat[0],2) , CI_rounded\n",
    "    \n",
    "\n",
    "df_sum_cat_all_deaths.sort_values(by = 'Vaccinated', ascending = False, inplace = True) # sorting\n",
    "\n",
    "# save\n",
    "df_sum_cat_all_deaths.to_csv('analysis/2c_total_mortality_table.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df with p-value, significance, RR and CI\n",
    "df_combined_deaths.to_csv('analysis/2c_chi_square_RR_table.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize = (8,8))\n",
    "\n",
    "# plot results for vaccinated and not on top of each other\n",
    "for v in dict_vacORnot:\n",
    "    plt.barh(df_sum_cat_all_deaths.index, df_sum_cat_all_deaths[v], alpha = 0.5, label = v)\n",
    "\n",
    "# labelling\n",
    "plt.title('Deaths in subjects with underlying diseases in specific ICD10 categories')\n",
    "plt.xlabel('Number of deaths per thousand individuals')\n",
    "plt.legend(loc = \"upper right\")\n",
    "\n",
    "# save figure in vector format, bbox_inches='tight' so that the plot is not cropped\n",
    "plt.savefig('analysis/2c_deaths_barplot.svg', bbox_inches='tight')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# plot the RR with CI\n",
    "\n",
    "# general figure configuration\n",
    "plt.figure(figsize = (6,6))\n",
    "plt.xscale(\"log\")\n",
    "plt.axvline(1, ls='--', linewidth=1, color='black')\n",
    "\n",
    "# draw the RR\n",
    "# RR results\n",
    "x = RR_result_cat_deaths['RR'].values\n",
    "# error bar with same size in both directions\n",
    "x_error = RR_result_cat_deaths['RR']-RR_result_cat_deaths['CI_lower']\n",
    "# for equal spacing of the results\n",
    "y = np.arange(len(RR_result_cat_deaths))\n",
    "\n",
    "# the plot\n",
    "plt.errorbar(x, y, xerr = x_error, marker = \"o\", markersize = 10, color = 'b', ls='none')\n",
    "\n",
    "# labelling\n",
    "plt.xlabel('Risk Ratio (log scale)')\n",
    "plt.title('Risk ratio of vaccination in people with underlying diseases in different ICD10 categories') # renee: Risk ratio of vaccination on an adverse event in subjects with underlying diseases in specific ICD-10 categories\n",
    "# labelling with ICD10 categories\n",
    "plt.yticks(ticks = y, labels = RR_result_cat_deaths.index) \n",
    "\n",
    "# save figure in vector format, bbox_inches='tight' so that the plot is not cropped\n",
    "plt.savefig('analysis/2c_deaths_RR.svg', bbox_inches='tight')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Discussion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# References\n",
    "Choi, W. S. & Cheong, H. J. COVID-19 Vaccination for People with Comorbidities. Infect Chemother 53, 155-158, doi:10.3947/ic.2021.0302 (2021)."
   ]
  }
 ],
 "metadata": {
  "colab": {
   "authorship_tag": "ABX9TyPbGrXdLmVVZJgOaOjZE42u",
   "collapsed_sections": [],
   "mount_file_id": "1lzi9RvA-pKhZOBpn5_9XFF17e746VcKM",
   "name": "Project_vaccine_comorbidities.ipynb",
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
