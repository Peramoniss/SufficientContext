import pandas as pd
import random

# Errors from Hotpot and 2Wiki
def clean_hotpot_errors(df):
    delete_list = []
    # Removing human error
    for i, row in df.iterrows():
        sup = set(row['supporting_facts']['title'])
        cont = set(row['context']['title'])
        if not sup.issubset(cont) or sup == cont: # If sup == cont (no distractors) or sup is not contained in cont (the sufficient information is not there)
            delete_list.append(i)
        # or sup >= cont - 2 # Add if wants to set a minimum amount of distactors (in this case, 2)
    removed = len(delete_list)
    df.drop(delete_list, inplace=True)
    return removed


def get_sufficient_context_std_format(df, desired_context_size : int = 5):
    if desired_context_size < 3:
       raise ValueError("Context size for musique must be at least 3, otherwise sufficient context will contain insufficient information")
    
    df_suff_treated = df.copy()
    for i, row in df.iterrows():
        contexts = []

        sup_facts = row['supporting_facts']['title']
        refs = set(sup_facts)
        curr_context = row['context']
        for c_id, c_title in enumerate(curr_context['title']):
            if c_title in sup_facts: # Verify if the context is necessary
                # If it is, integrate the subchunks into a single chunk and add to the list of chunks
                chunk_sentences = curr_context['sentences'][c_id]
                contexts.append(''.join(chunk_sentences))
    
        # Second iteration, necessary because every necessary context must be added before adding noise [TODO: Really necessary or am I dumb?]
        for c_id, c_title in enumerate(curr_context['title']):
            chunk_sentences = curr_context['sentences'][c_id]
            if c_title not in sup_facts and len(contexts) < desired_context_size: # If it is a distractor and there's not enough passages yet
                contexts.append(''.join(chunk_sentences))

            if len(contexts) >= desired_context_size:
                break # Once reached the desired context size, stop

        df_suff_treated.at[i, 'context'] = contexts # Redefine the context column
        df_suff_treated.at[i, 'references'] = list(refs) # Redefine the refences column

    df_suff_treated['mixed'] = 0
    return df_suff_treated

def get_insufficient_context_std_format(df, desired_context_size : int = 5):
    if desired_context_size < 3:
       raise ValueError("Context size for musique must be at least 3, otherwise sufficient context will contain insufficient information")

    df_insuff_treated = df.copy()
    df_insuff_treated['mixed'] = 0

    for i, row in df.iterrows(): # Itera sobre o dataset
        contexts = []
        refs = set(row['supporting_facts']['title'])
        chunks_in_instance = 0
        if random.uniform(0, 1) < 0.5:
            mixed = True
        else:
            mixed = False

        for c_id, c_title in enumerate(row['context']['title']): # Para cada chunk de contexto (incluindo distratores)
            # matches = False
            # for fact in row['supporting_facts']['title']: # Para cada contexto necessário
            #     refs.add(fact) # Adiciona 
            #     if fact == c_title: # Verifica se esse chunk é necessário
            #         matches = True
            matches = c_title in row['supporting_facts']['title']

            if not matches or (mixed == True and matches): # Se não for necessário (for distrator) ou for necessário mas é para misturar
                if matches: # Se entrou pela segunda condição
                    df_insuff_treated.at[i, 'mixed'] = 1
                    mixed = False

                chunk_sentences = row['context']['sentences'][c_id]
                contexts.append(''.join(chunk_sentences)) # Adiciona o chunk ao conjunto
                chunks_in_instance += 1 # Aumenta a contagem

                if chunks_in_instance == desired_context_size and mixed == False: # Se já tem o valor normal e não precisa adicionar um chunk necessário
                    break
                elif chunks_in_instance == desired_context_size and mixed == True: # Se ainda precisa adicionar um chunk necessários
                    contexts.pop() # Remove um distrator
                    chunks_in_instance -= 1 # Mantém a contagem e continua procurando um chunk necessário para adicionar

        df_insuff_treated.at[i, 'context'] = contexts
        df_insuff_treated.at[i, 'references'] = list(refs)
        df_insuff_treated.at[i, 'size'] = chunks_in_instance # Just to guarantee the size is valid
    return df_insuff_treated

def clean_musique_errors(df):
    delete_list = []
    # Removing human error
    for i, row in df.iterrows():
        sp_facts = 0
        context_len = len(row["paragraphs"])
        for p in row["paragraphs"]:
            if p["is_supporting"] == True:
                sp_facts += 1

        if sp_facts >= context_len - 2:
            delete_list.append(i)

    removed = len(delete_list)
    df.drop(delete_list, inplace=True)
    return removed

def get_sufficient_context_musique_format(df, desired_context_size : int = 5):
    if desired_context_size < 3:
       raise ValueError("Context size for musique must be at least 3, otherwise sufficient context will contain insufficient information")
    df_suff_treated = df.copy()
    df_suff_treated["context"] = None
    df_suff_treated["references"] = None
    df_suff_treated['size'] = 0

    for i, row in df.iterrows():
        contexts = []
        refs = set()

        for p in row['paragraphs']: 
            if p["is_supporting"] == True: # For each necessary chunk 
                contexts.append(p["paragraph_text"]) # Add to context
                refs.add(p["title"])
                df_suff_treated.at[i, 'size'] += 1

        # Introduce noise until reaching the desired size
        for p in row['paragraphs']:
            if p["is_supporting"] == False and len(contexts) < desired_context_size:
                contexts.append(p["paragraph_text"])
                df_suff_treated.at[i, 'size'] += 1

        df_suff_treated.at[i, 'context'] = contexts # Redefine os contextos no novo dataset
        df_suff_treated.at[i, 'references'] = list(refs) # Redefine os contextos no novo dataset

    df_suff_treated['mixed'] = 0
    return df_suff_treated

def get_insufficient_context_musique_format(df, desired_context_size : int = 5):
    if desired_context_size < 3:
       raise ValueError("Context size for musique must be at least 3, otherwise sufficient context will contain insufficient information")
    
    df_insuff_treated = df.copy()
    df_insuff_treated["number_of_supporting_facts"] = 0

    for i, row in df_insuff_treated.iterrows():
        for p in row["paragraphs"]:
            if p["is_supporting"] == True:
                df_insuff_treated.loc[i, "number_of_supporting_facts"] += 1

    # Verifies how to guarantee that 50% of the instances with at least one supporting fact (number_of_supporting_facts > 0) have their noise incorporated
    mixed_chance = df_insuff_treated.shape[0] * 0.5 / (df_insuff_treated.shape[0]  - df_insuff_treated["number_of_supporting_facts"].value_counts()[0])
    df_insuff_treated["mixed"] = 0 # Assumes there's no noise, will change otherwise
    df_insuff_treated["context"] = None
    df_insuff_treated["references"] = None
    df_insuff_treated["size"] = 0

    for i, row in df.iterrows(): 
        contexts = []
        refs = set()
        chunks_in_instance = 0 # More efficient than using len all the time
        if row.get("number_of_supporting_facts", default=0) > 0 and random.uniform(0, 1) < mixed_chance:
            # mixed = True
            mixed_ctr = min(random.randint(1, row["number_of_supporting_facts"]), desired_context_size) # Defines how many supporting facts will be added (in randint, the maximum value is not included, so there will never be a case in which every supporting fact is added)
            df_insuff_treated.at[i, "mixed"] = 1
        else:
            # mixed = False
            mixed_ctr = 0

        for p in row["paragraphs"]:
            supporting = p["is_supporting"]

            if supporting == True: # Clean comparison
                refs.add(p["title"])

            if supporting == False or (supporting == True and mixed_ctr > 0):
                if supporting:  # If entered by the second condition, manage the counter
                    mixed_ctr -= 1

                contexts.append(p["paragraph_text"]) # Either way, add the chunk to the context 
                chunks_in_instance += 1 

                if chunks_in_instance == desired_context_size and mixed_ctr > 0:  # If there are still supporting facts to add but the desired length was reached
                    contexts.pop()  # Remove a distractor
                    chunks_in_instance -= 1 

            if chunks_in_instance == desired_context_size and mixed_ctr <= 0: # Se já tem o valor normal e não precisa adicionar um chunk necessário
                break

        df_insuff_treated.at[i, "context"] = contexts
        df_insuff_treated.at[i, 'references'] = list(refs) # Redefine os contextos no novo dataset
        
    return df_insuff_treated