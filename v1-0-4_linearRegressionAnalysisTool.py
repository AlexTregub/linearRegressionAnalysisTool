#!/usr/bin/python3
##### Alex Tregub
##### 2024-06-21
##### Script for random resampling of data, univariate and multivariate linear
#####     regression, and p-value tests
##### ===========
##### v1.0.3
##### Merged functionality from previously seperate scripts into one script
#####     with one config. Handles: random resampling of data from one input
#####     file, univariate linear regression with p-value tests, and
#####     multivariate linear regression with p-value tests. 
##### - Current p-value based tests used are the Bonferroni correction and the
#####       Benjamini-Hochberg procedure. 
##### - Currently all functions of the script are run sequentially as follows:
#####       Resample input data > For all samples: (Run univariate linear 
#####       regression for one dependent and fixed independent variables > Run 
#####       multivariate linear regression for one dependent variable and fixed
#####       independent variables.)
##### - Logging of current operations done to console.
##### - Function/Errors are not handled. Exits with trace.
##### - Linear regression is currently hardcoded to library outputs (future 
#####       versions may change order)
##### ===========
##### - (1.0.1) Linear Regression is now handled as a function. 
##### - (1.0.1) Resampling is now handled as a function.
##### - (1.0.2) Univariate and Multivariate linear regressions are now handled
#####       as needed. (Not yet within function calls)
##### - (1.0.2) Not all samples in target resample folder will be run, only
#####       ones that were resampled by this tool will be considered (For now)
##### - (1.0.3) Outputs from univariate linear regression will be stored as an
#####       list of DataFrames, allowing for a uniform output of across linear
#####       regression models.
##### - (1.0.3) P-Value tests added for univariate
##### - (1.0.4) Cleanup hardcoded values 
#Imports...
import pandas as pd

from time import time
import numpy as np

from os import listdir
from os.path import isfile

import seaborn as sns # seaborn==0.12.2
import statsmodels.api as sm # statsmodels==0.13.5+dfsg


#Config Vars...
INPUT_CSV_FILE_PATH = "./input.csv"
DROP_COLUMNS = ['State']

RESAMPLE_FOLDER_PATH = "./resampled"
#NUMPY_SEED = time()
NUMPY_SEED = 0 # Debug seed
GENERATE_SAMPLES = 100 # Number of samples; 10000 = used samples for GST
SAMPLES_SIZE = 10 # Number of rows In each sample; 25 = used for GST

DEPENDENT_TARGET = 'SELECTED_VARIABLE'
INDEPENDENT_SELECT = ["SELECTED_INDEPENDENT_VARIABLE1","SELECTED_INDEPENDENT_VARIABLE2","SELECTED_INDEPENDENT_VARIABLE3"]
PVAL_THRESH = 0.05



#### Functions...
## Basic linear regression implementation using statsmodel Ordinary Least Squares to run regression, and then parses the full return into dataframe. Logs time to console
def linearRegression(inputData,dependentTarget,independentTargets):
    startRegression = time()

    ## Run OLS linear regression via statsmodel module 
    regressionIndependent = sm.add_constant(inputData[independentTargets])
    regressionModel = sm.OLS(inputData[dependentTarget],regressionIndependent)
    regressionRun = regressionModel.fit()
    regressionResults = regressionRun.summary2()


    ## Parse tables as necessary
    regressionTable1 = regressionResults.tables[0] # Model details 
    regressionTable2 = regressionResults.tables[1] # Per-Variable results
    regressionTable3 = regressionResults.tables[2] # Secondary Analysis of model

    # Hardcoded variable names (CORRESPOND TO ORDER OF APPENDS FOR EACH TABLE)
    regressionResultListCols = ["R-squared","AdjustedR-squared","AkaikeInformationCriterion","BayesInformationCriterion","LogLikelihood","F-Statistic","ProbabilityF-Statistic","Scaling","OmnibusNorm","ProbabilityOmnibusNorm","Skew","Kurtosis","Durbin-Watson","Jarque-Bera","ProbabilityJarque-Bera","ConditionNumber"]
    regressionResultList = []

    # Table 1
    regressionResultList.append(float(regressionTable1[1][6])) # R-squared
    regressionResultList.append(float(regressionTable1[3][0])) # Adjusted R-squared
    regressionResultList.append(float(regressionTable1[3][1])) # Akaike information criterion
    regressionResultList.append(float(regressionTable1[3][2])) # Bayes information criterion
    regressionResultList.append(float(regressionTable1[3][3])) # Log-likelihood
    regressionResultList.append(float(regressionTable1[3][4])) # F-statistic
    regressionResultList.append(float(regressionTable1[3][5])) # Probability of F-statistic
    regressionResultList.append(float(regressionTable1[3][6])) # Scaling value for predictor
    
    # Table 3
    regressionResultList.append(float(regressionTable3[1][0])) # Omnibus normtest
    regressionResultList.append(float(regressionTable3[1][1])) # Probability of Omnibus normtest
    regressionResultList.append(float(regressionTable3[1][2])) # Skew
    regressionResultList.append(float(regressionTable3[1][3])) # Kurtosis
    regressionResultList.append(float(regressionTable3[3][0])) # Durbin-Watson test
    regressionResultList.append(float(regressionTable3[3][1])) # Jarque-Bera test 
    regressionResultList.append(float(regressionTable3[3][2])) # Probability of Jarque-Bera test
    regressionResultList.append(float(regressionTable3[3][3])) # Condition Number

    # Table 2
    for i in range(len(regressionTable2.index)): # For all variables in model
        selName = regressionTable2.index.values[i]
        varName = None
        if i == 0: # and selName == "const" <- always true as add_constant() used
            varName = "intercept_"
        else:
            varName = selName.replace(",","_")+"_" # Sanitize variable names (as In/Out expected to be in csv format)
        
        regressionResultListCols.extend([varName+"Coefficient",varName+"StandardError",varName+"t",varName+"Probability-t",varName+"0.025",varName+"0.975"])
        regressionResultList.append(float(regressionTable2['Coef.'][selName]))
        regressionResultList.append(float(regressionTable2['Std.Err.'][selName]))
        regressionResultList.append(float(regressionTable2['t'][selName]))
        regressionResultList.append(float(regressionTable2['P>|t|'][selName]))
        regressionResultList.append(float(regressionTable2['[0.025'][selName]))
        regressionResultList.append(float(regressionTable2['0.975]'][selName]))


    ## Return results as DF
    #print("Regression run took "+str(time() - startRegression)+" seconds.")
    return pd.DataFrame(data=[regressionResultList],columns=regressionResultListCols)

## Resampling without replacement implementation via uniform integer distribution in numpy. No input checking done. Returns list of sample locations in order
def resampleData(inputData,sampleCount,sampleSize,outputPath,numpySeed):
    startResample = time()
    resampleLocs = []

    resamplingRng = np.random.default_rng(numpySeed) # Setup numpy random
    rowCount = len(inputData.index) # Get row length

    for sampleNum in range(sampleCount):
        availRows = list(range(rowCount)) # Creates list from 0 to row length of the dataframe (row index)
        selRows = []

        for j in range(sampleSize): # Needs the size of samples to be smaller than length of input data
            randPos = resamplingRng.integers(0,len(availRows)) # Selects random position in index of available rows
            selRows.append(availRows.pop(randPos)) # Removes DF row num from available rows list, and puts it into the selected rows list
        
        outputLoc = outputPath + "/sample" +str(sampleNum)+".csv"
        resampleLocs.append(outputLoc) # Stores location of sample for later
        (fullInputData.loc[fullInputData.index[selRows]]).to_csv(outputLoc,index=False,header=True) # DF.loc[...] Creates 'view' of input data with only selected rows without copy, then outputs to csv while dropping index; Needs resampling folder to exist (preferably empty)

    print("Resampling took "+str(time()-startResample) + " seconds. (~"+str((time()-startResample)/sampleCount)+" sec per sample for "+str(sampleCount)+" samples.")
    return resampleLocs


#### Script...
#GENERATE_SAMPLES = 10 # TEMPORARY DEBUG VAL
print("LRAT v1.0.4 started.")

## Import input csv file, drop selected columns, output 'head' of table
fullInputData = pd.read_csv(INPUT_CSV_FILE_PATH) # Input file should exist + be valid
print(fullInputData.head(5))

# Drop columns
print("Columns to drop:",DROP_COLUMNS,"\n")
for selCol in DROP_COLUMNS: 
    fullInputData.drop(columns=selCol,inplace=True) # Column name should exist

print(fullInputData.head(5))


## Resample input DF to output folder
sampleLocs = resampleData(fullInputData,GENERATE_SAMPLES,SAMPLES_SIZE,RESAMPLE_FOLDER_PATH,NUMPY_SEED)


## For every sample in resampled data...
multivariateResults = pd.DataFrame()
univariateResultList = [ pd.DataFrame() for _ in range(len(INDEPENDENT_SELECT)) ]

for sampleNum,sample in enumerate(sampleLocs):
    inputSample = pd.read_csv(sample)

    ## Run multivariate regression (use all independent variables selcted at once)
    regressionRun = linearRegression(inputSample,DEPENDENT_TARGET,INDEPENDENT_SELECT)
    multivariateResults = pd.concat([multivariateResults,regressionRun])


    ## Run univariate regression (use each selected independent variable and merge outputs into one list)
    for resultNum,selCol in enumerate(INDEPENDENT_SELECT):
        regressionRun = linearRegression(inputSample,DEPENDENT_TARGET,selCol)
        univariateResultList[resultNum] = pd.concat([univariateResultList[resultNum],regressionRun])

# Prepare data and output to console 
multivariateResults.reset_index(inplace=True,drop=True)
for i in range(len(univariateResultList)):
    univariateResultList[i].reset_index(inplace=True,drop=True)
print(multivariateResults)
print(univariateResultList)
print("---")

multivariateResults.to_csv("./multivariateOutput.csv",index=False,header=True)



## Perform p-value tests for intercept and any variables in dataframe
for i in range(len(univariateResultList)): # Sets up additional columns in univariate results 
    univariateResultList[i].insert(len(univariateResultList[i].columns.values.tolist()),'intercept_Probability-t_Rank',pd.NA)
    univariateResultList[i].insert(len(univariateResultList[i].columns.values.tolist()),"intercept_Probability-t_BonferroniThreshold",pd.NA)
    univariateResultList[i].insert(len(univariateResultList[i].columns.values.tolist()),"intercept_Probability-t_Benjamini-HochbergThreshold",pd.NA)
    univariateResultList[i].insert(len(univariateResultList[i].columns.values.tolist()),"intercept_BonferroniPass",pd.NA)
    univariateResultList[i].insert(len(univariateResultList[i].columns.values.tolist()),"intercept_Benjamini-HochbergPass",pd.NA)
    univariateResultList[i].insert(len(univariateResultList[i].columns.values.tolist()),'var_Probability-t_Rank',pd.NA)
    univariateResultList[i].insert(len(univariateResultList[i].columns.values.tolist()),"var_Probability-t_BonferroniThreshold",pd.NA)
    univariateResultList[i].insert(len(univariateResultList[i].columns.values.tolist()),"var_Probability-t_Benjamini-HochbergThreshold",pd.NA)
    univariateResultList[i].insert(len(univariateResultList[i].columns.values.tolist()),"var_BonferroniPass",pd.NA)
    univariateResultList[i].insert(len(univariateResultList[i].columns.values.tolist()),"var_Benjamini-HochbergPass",pd.NA)
    
# Univariate linear regression p-val tests
# for selVar in multivariateResults.columns.values.tolist():
#     if selVar.endswith("-t"): # Gets 'p-value' variables from DF for tests. Need to perform 
#         print(selVar)
for sampleNum in range(len(univariateResultList[0].index.values.tolist())): # For every 'sample' row in dataframe length (according to first entry)
    univariateAnalysis = pd.DataFrame() # Used for 1 full sample
    for dfIndex,univariateDf in enumerate(univariateResultList): # For all dataframes in univariate result list, create analysis row
        # Get current variables which will be tested
        availVars = []
        for selVar in univariateDf.columns.values.tolist():
            if selVar.endswith("Probability-t"):
                availVars.append(selVar)
        #print(availVars)

        #columnsAvail = ["inSampleIndex","intercept_Probability-t","intercept_Probability-t_Rank","intercept_Probability-t_BonferroniThreshold","intercept_Probability-t_Benjamini-HochbergThreshold","intercept_BonferroniPass","intercept_Benjamini-HochbergPass","var_Probability-t","var_Probability-t_Rank","var_Probability-t_BonferroniThreshold","var_Probability-t_Benjamini-HochbergThreshold","var_BonferroniPass","var_Benjamini-HochbergPass"]
        extractedVals = [dfIndex]
        extractedVals.append(univariateDf['intercept_Probability-t'][sampleNum])
        extractedVals.extend([0,0,0,False,False]) # Filler for Rank through Pass/Fail
        extractedVals.append(univariateDf[availVars[-1]][sampleNum]) # Get other variable (as univariate, only two 'variables' considered)
        extractedVals.extend([0,0,0,False,False]) # Filler for Rank through Pass/Fail

        #testDf = pd.DataFrame(data=[extractedVals],columns=["inSampleIndex","intercept_Probability-t","intercept_Probability-t_Rank","intercept_Probability-t_BonferroniThreshold","intercept_Probability-t_Benjamini-HochbergThreshold","intercept_BonferroniPass","intercept_Benjamini-HochbergPass","var_Probability-t","var_Probability-t_Rank","var_Probability-t_BonferroniThreshold","var_Probability-t_Benjamini-HochbergThreshold","var_BonferroniPass","var_Benjamini-HochbergPass"])
        #print(testDf)
        univariateAnalysis = pd.concat([univariateAnalysis,pd.DataFrame(data=[extractedVals],columns=["inSampleIndex","intercept_Probability-t","intercept_Probability-t_Rank","intercept_Probability-t_BonferroniThreshold","intercept_Probability-t_Benjamini-HochbergThreshold","intercept_BonferroniPass","intercept_Benjamini-HochbergPass","var_Probability-t","var_Probability-t_Rank","var_Probability-t_BonferroniThreshold","var_Probability-t_Benjamini-HochbergThreshold","var_BonferroniPass","var_Benjamini-HochbergPass"])])

    univariateAnalysis.reset_index(inplace=True,drop=True)


    # Compute p-tests (intercept)
    univariateAnalysis.sort_values('intercept_Probability-t',ascending=True,inplace=True,ignore_index=True) # Sorts + resets index
    for i in range(len(univariateAnalysis.index.values.tolist())):
        univariateAnalysis.at[i,'intercept_Probability-t_Rank'] = i # Stores rank record for storage in final df
        univariateAnalysis.at[i,'intercept_Probability-t_BonferroniThreshold'] = PVAL_THRESH/len(INDEPENDENT_SELECT) # alpha/hypothesis
        if univariateAnalysis['intercept_Probability-t'][i] < univariateAnalysis['intercept_Probability-t_BonferroniThreshold'][i]:
            univariateAnalysis.at[i,'intercept_BonferroniPass'] = True
        else:
            univariateAnalysis.at[i,'intercept_BonferroniPass'] = False # Redundant, kept for readability.

        univariateAnalysis.at[i,'intercept_Probability-t_Benjamini-HochbergThreshold'] = ((i+1)/len(INDEPENDENT_SELECT))*PVAL_THRESH # alpha*(rank/hypothesis)
        if univariateAnalysis['intercept_Probability-t'][i] < univariateAnalysis['intercept_Probability-t_Benjamini-HochbergThreshold'][i]:
            univariateAnalysis.at[i,'intercept_Benjamini-HochbergPass'] = True
        else:
            univariateAnalysis.at[i,'intercept_Benjamini-HochbergPass'] = False # Redundant.

    # Compute p-tests (var)
    univariateAnalysis.sort_values('var_Probability-t',ascending=True,inplace=True,ignore_index=True) # Sorts + resets index
    for i in range(len(univariateAnalysis.index.values.tolist())):
        univariateAnalysis.at[i,'var_Probability-t_Rank'] = i # Stores rank record for storage in final df
        univariateAnalysis.at[i,'var_Probability-t_BonferroniThreshold'] = PVAL_THRESH/len(INDEPENDENT_SELECT) # alpha/hypothesis
        if univariateAnalysis['var_Probability-t'][i] < univariateAnalysis['var_Probability-t_BonferroniThreshold'][i]:
            univariateAnalysis.at[i,'var_BonferroniPass'] = True
        else:
            univariateAnalysis.at[i,'var_BonferroniPass'] = False # Redundant, kept for readability.

        univariateAnalysis.at[i,'var_Probability-t_Benjamini-HochbergThreshold'] = ((i+1)/len(INDEPENDENT_SELECT))*PVAL_THRESH # alpha*(rank/hypothesis)
        if univariateAnalysis['var_Probability-t'][i] < univariateAnalysis['var_Probability-t_Benjamini-HochbergThreshold'][i]:
            univariateAnalysis.at[i,'var_Benjamini-HochbergPass'] = True
        else:
            univariateAnalysis.at[i,'var_Benjamini-HochbergPass'] = False # Redundant.


    #print(univariateAnalysis.to_string())

    # Output Analysis to original Dfs 
    for i in range(len(univariateAnalysis.index.values.tolist())): # For each variable tested // row in analysis dataframe
        outputIndex = univariateAnalysis['inSampleIndex'][i] # Selects the dataframe to output to 

        univariateResultList[outputIndex].at[sampleNum,"intercept_Probability-t_Rank"] = univariateAnalysis["intercept_Probability-t_Rank"][i]
        #print(univariateAnalysis["intercept_Probability-t_Rank"][i])
        #print(univariateResultList[outputIndex].to_string())
        
        univariateResultList[outputIndex].at[sampleNum,"intercept_Probability-t_BonferroniThreshold"] = univariateAnalysis["intercept_Probability-t_BonferroniThreshold"][i]
        univariateResultList[outputIndex].at[sampleNum,"intercept_Probability-t_Benjamini-HochbergThreshold"] = univariateAnalysis["intercept_Probability-t_Benjamini-HochbergThreshold"][i]
        univariateResultList[outputIndex].at[sampleNum,"intercept_BonferroniPass"] = univariateAnalysis["intercept_BonferroniPass"][i]
        univariateResultList[outputIndex].at[sampleNum,"intercept_Benjamini-HochbergPass"] = univariateAnalysis["intercept_Benjamini-HochbergPass"][i]
        univariateResultList[outputIndex].at[sampleNum,"var_Probability-t_Rank"] = univariateAnalysis["var_Probability-t_Rank"][i]
        univariateResultList[outputIndex].at[sampleNum,"var_Probability-t_BonferroniThreshold"] = univariateAnalysis["var_Probability-t_BonferroniThreshold"][i]
        univariateResultList[outputIndex].at[sampleNum,"var_Probability-t_Benjamini-HochbergThreshold"] = univariateAnalysis["var_Probability-t_Benjamini-HochbergThreshold"][i]
        univariateResultList[outputIndex].at[sampleNum,"var_BonferroniPass"] = univariateAnalysis["var_BonferroniPass"][i]
        univariateResultList[outputIndex].at[sampleNum,"var_Benjamini-HochbergPass"] = univariateAnalysis["var_Benjamini-HochbergPass"][i]

    
    #print(univariateResultList[outputIndex].to_string())
    #input()     


for i in range(len(univariateResultList)):
    outputLoc = "./univariateOutput"+(INDEPENDENT_SELECT[i].replace(" ","_").replace("%","_").replace("(","-").replace(")","-"))+".csv"
    univariateResultList[i].to_csv(outputLoc,index=False,header=True)


#...