Here is the bullet points about the construction of the dataset. As discussed, I don't think it is necessary that everyone uses this dataset.
What is important is to understand what columns are being used and why they are necessary for each step. This can mostly be done by identifying the 
datatypes for the columns: numeric/ordinal, enum/class, free text. These are not the datatypes as they are required for a DB or dataframe, but their semantic datatype. 


Patient-Encounter dataset

The patient encounter dataset has been created to provide an example dataset and an example of data processing using pandas. These are the steps
that have been taken to generate the final dataset.

1. Generate an initial set of data using Synthea, https://synthetichealth.github.io/synthea/
   1. Synthea is being used because it is the standard tool used to generate data for our FHIR and OMOP demos. 
2. For the our purposes, we needed to generate some free text so the codified data is used with an LLM to create a clinicians not for each encounter.
   1. The first step is creating a denormalized table that has one encounter per row. The conditions, obeservations, procedures and medications all have an N to 1 cardinality to encounters so one column for each of those tables Description column is created and the values are concatanented into that new respective column.
   2. Since some of the descriptions are very long and there can be a lot of them, an LLM was used to create abbreviated values for each. For example: 'Continued opioid use despite having persistent or recurrent social or interpersonal problems caused or exacerbated by the effects of opioids' -> 'OpioidSocialProblems'
3. The data for each record from this denormailzed table is then provided to an LLM to have it generate a free text clinical notes

