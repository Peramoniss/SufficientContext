from datasets import load_dataset
import pandas as pd
import DatasetGenerator.helperFunctions as helper
from sklearn.model_selection import train_test_split

def generate_hotpot_qa_dataset():
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

    # Convert to Pandas Dataframe
    df_train = pd.DataFrame(ds["train"])
    df_val = pd.DataFrame(ds["validation"])

    # Remove unhelpful columns, cleaning memory
    df_train.drop(columns=['id'], inplace=True)
    df_val.drop(columns=['id'], inplace=True)

    # Remove problematic instances
    helper.clean_hotpot_errors(df_train)
    helper.clean_hotpot_errors(df_val)

    # Divide the instances so half is sufficient and half isn't while maintaning the type and difficulty distribution
    strat_key = df_train["type"] + "_" + df_train["level"]
    df_train_suff, df_train_insuff  = train_test_split(df_train, stratify=strat_key, test_size=0.5, random_state=42)

    strat_key = df_val["type"] + "_" + df_val["level"]
    df_val_suff, df_val_insuff  = train_test_split(df_val, stratify=strat_key, test_size=0.5, random_state=42)

    # Treat the instances in both subsets
    df_train_suff_treated = helper.get_sufficient_context_std_format(df_train_suff)
    df_val_suff_treated = helper.get_sufficient_context_std_format(df_val_suff)

    df_train_insuff_treated = helper.get_insufficient_context_std_format(df_train_insuff)
    df_val_insuff_treated = helper.get_insufficient_context_std_format(df_val_insuff)

    # Label the data
    df_train_suff_treated['label'] = 1 # Sufficient
    df_val_suff_treated['label'] = 1 # Sufficient
    df_train_insuff_treated['label'] = 0 # Insufficient
    df_val_insuff_treated['label'] = 0 # Insufficient

    # Concatenate both subsets
    df_train_final = pd.concat([df_train_suff_treated, df_train_insuff_treated], ignore_index=True)
    df_val_final = pd.concat([df_val_suff_treated, df_val_insuff_treated], ignore_index=True)

    # Drop irrelevant columns
    df_train_final.drop(columns=['supporting_facts'], inplace=True)
    df_val_final.drop(columns=['supporting_facts'], inplace=True)

    # Generate test dataset based on validation, respecting distribution
    strat_key = df_val_final["type"] + "_" + df_val_final["level"] + "_" + df_val_final["mixed"].astype(str) + "_" + df_val_final["label"].astype(str)
    df_val_final, df_test_final  = train_test_split(df_val_final, stratify=strat_key, test_size=0.5, random_state=42)

    # Save the dataset
    df_train_final.to_csv('../Datasets/HotpotQA/train.csv', index=False)
    df_val_final.to_csv('../Datasets/HotpotQA/val.csv', index=False)
    df_test_final.to_csv('../Datasets/HotpotQA/test.csv', index=False)

    # Create generalization dataset
    strat_key = df_train_final["type"] + "_" + df_train_final["level"] + "_" + df_train_final["mixed"].astype(str) + "_" + df_train_final["label"].astype(str)
    df_train_generalization, _  = train_test_split(df_train_final, stratify=strat_key, train_size=300, random_state=42)

    df_train_generalization.to_csv('../Datasets/HotpotQA/train_generalization.csv', index=False)

# TODO: Consider - There's a lot of repeated code but I think it works better when the logic is completely separated than when there are several ifs and elses or unnecessary function calls stacked
def generate_2wikimultihop_qa_dataset():
    ds = load_dataset("framolfese/2WikiMultihopQA")

    # Convert to Pandas Dataframe
    df_train = pd.DataFrame(ds["train"])
    df_val = pd.DataFrame(ds["validation"])

    # Remove unhelpful columns, cleaning memory
    df_train.drop(columns=['id'], inplace=True)
    df_val.drop(columns=['id'], inplace=True)

    # No problematic instances, no need to clean errors

    # Divide the instances so half is sufficient and half isn't while maintaning the type distribution
    df_train_suff, df_train_insuff  = train_test_split(df_train, stratify=df_train["type"], test_size=0.5, random_state=42)
    df_val_suff, df_val_insuff  = train_test_split(df_val, stratify=df_val["type"], test_size=0.5, random_state=42)

    # Treat the instances in both subsets
    df_train_suff_treated = helper.get_sufficient_context_std_format(df_train_suff)
    df_val_suff_treated = helper.get_sufficient_context_std_format(df_val_suff)

    df_train_insuff_treated = helper.get_insufficient_context_std_format(df_train_insuff)
    df_val_insuff_treated = helper.get_insufficient_context_std_format(df_val_insuff)

    # Label the data
    df_train_suff_treated['label'] = 1 # Sufficient
    df_val_suff_treated['label'] = 1 # Sufficient
    df_train_insuff_treated['label'] = 0 # Insufficient
    df_val_insuff_treated['label'] = 0 # Insufficient

    # Concatenate both subsets
    df_train_final = pd.concat([df_train_suff_treated, df_train_insuff_treated], ignore_index=True)
    df_val_final = pd.concat([df_val_suff_treated, df_val_insuff_treated], ignore_index=True)

    # Drop irrelevant columns
    df_train_final.drop(columns=['supporting_facts'], inplace=True)
    df_val_final.drop(columns=['supporting_facts'], inplace=True)

    # Generate test dataset based on validation, respecting distribution
    strat_key = df_val_final["type"] + "_" + df_val_final["mixed"].astype(str) + "_" + df_val_final["label"].astype(str)
    df_val_final, df_test_final  = train_test_split(df_val_final, stratify=strat_key, test_size=0.5, random_state=42)

    # Save the dataset
    df_train_final.to_csv('../Datasets/2WikiMultihopQA/train.csv', index=False)
    df_val_final.to_csv('../Datasets/2WikiMultihopQA/val.csv', index=False)
    df_test_final.to_csv('../Datasets/2WikiMultihopQA/test.csv', index=False)

    # Create generalization dataset
    strat_key = df_train_final["type"] + "_" + df_train_final["mixed"].astype(str) + "_" + df_train_final["label"].astype(str)
    df_train_generalization, _  = train_test_split(df_train_final, stratify=strat_key, train_size=300, random_state=42)

    df_train_generalization.to_csv('../Datasets/2WikiMultihopQA/train_generalization.csv', index=False)

def generate_musique_dataset():
    ds = load_dataset("bdsaglam/musique")

    # Convert to Pandas Dataframe
    df_train = pd.DataFrame(ds["train"])
    df_val = pd.DataFrame(ds["validation"])

    # Remove unhelpful columns, cleaning memory
    df_train.drop(columns=['id'], inplace=True)
    df_val.drop(columns=['id'], inplace=True)

    helper.clean_musique_errors(df_train)
    helper.clean_musique_errors(df_val)

    # Musique data is already labeled, just rename the column
    df_train = df_train.rename(columns={'answerable': 'label'})
    df_val = df_val.rename(columns={'answerable': 'label'})

    # Slice the dataset
    df_train_suff, df_train_insuff = df_train[df_train['label'] == 1], df_train[df_train['label'] == 0]
    df_val_suff, df_val_insuff = df_val[df_val['label'] == 1], df_val[df_val['label'] == 0]

    # Treat the instances in both subsets
    df_train_suff_treated = helper.get_sufficient_context_musique_format(df_train_suff)
    df_val_suff_treated = helper.get_sufficient_context_musique_format(df_val_suff)

    df_train_insuff_treated = helper.get_insufficient_context_musique_format(df_train_insuff)
    df_val_insuff_treated = helper.get_insufficient_context_musique_format(df_val_insuff)

    # Concatenate both subsets
    df_train_final = pd.concat([df_train_suff_treated, df_train_insuff_treated], ignore_index=True)
    df_val_final = pd.concat([df_val_suff_treated, df_val_insuff_treated], ignore_index=True)

    # # Drop irrelevant columns
    # df_train_final.drop(columns=['supporting_facts'], inplace=True)
    # df_val_final.drop(columns=['supporting_facts'], inplace=True)

    # Generate test dataset based on validation, respecting distribution
    strat_key = df_val_final["mixed"].astype(str) + "_" + df_val_final["label"].astype(str)
    df_val_final, df_test_final  = train_test_split(df_val_final, stratify=strat_key, test_size=0.5, random_state=42)

    # Save the dataset
    df_train_final.to_csv('../Datasets/MuSiQue/train.csv', index=False)
    df_val_final.to_csv('../Datasets/MuSiQue/val.csv', index=False)
    df_test_final.to_csv('../Datasets/MuSiQue/test.csv', index=False)

    # Create generalization dataset
    strat_key = df_train_final["mixed"].astype(str) + "_" + df_train_final["label"].astype(str)
    df_train_generalization, _  = train_test_split(df_train_final, stratify=strat_key, train_size=300, random_state=42)

    df_train_generalization.to_csv('../Datasets/MuSiQue/train_generalization.csv', index=False)