import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
import ast

def extract_bert_shortcut_features(df, tokenizer):
    lengths = []
    overlaps = []

    for _, row in df.iterrows():
        full_context = ""
        try:
            context = ast.literal_eval(row['context'])
        except Exception as e:
            print(e)
            print(row['context'])
        for c in context:
            full_context += f"{c}\n"

        # Tokenize question and context
        q_tokens = tokenizer.tokenize(row['question'])
        c_tokens = tokenizer.tokenize(full_context)

        # Feature 1: Total concatenated token length
        total_len = len(q_tokens) + len(c_tokens)
        lengths.append(total_len)

        # Feature 2: Token overlap (Jaccard index)
        set_q = set(q_tokens)
        set_c = set(c_tokens)
        
        intersection = set_q.intersection(set_c)
        union = set_q.union(set_c)
        
        jaccard_overlap = len(intersection) / len(union) if union else 0.0
        overlaps.append(jaccard_overlap)

    df['token_length'] = lengths
    df['token_overlap'] = overlaps
    return df

def shortcut_validate(dataset:str, output_path:str = "console"):
    if dataset == 'HotpotQA':
        df_train = pd.read_csv("../Datasets/HotpotQA/train.csv")
        df_test = pd.read_csv("../Datasets/HotpotQA/test.csv")
    elif dataset == '2WikiMultihopQA':
        df_train = pd.read_csv("../Datasets/2WikiMultihopQA/train.csv")
        df_test = pd.read_csv("../Datasets/2WikiMultihopQA/test.csv")
    elif dataset == 'MuSiQue':
        df_train = pd.read_csv("../Datasets/MuSiQue/train.csv")
        df_test = pd.read_csv("../Datasets/MuSiQue/test.csv")
    else:
        raise ValueError(f'Dataset field is required and must be one of the following: HotpotQA, 2WikiMultihopQA, or MuSiQue. Value sent was {dataset}')
    
    # 2. Initialize BERT Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # 3. Extract Token Length and Overlap Features
    df_train = extract_bert_shortcut_features(df_train, tokenizer)
    df_test = extract_bert_shortcut_features(df_test, tokenizer)

    X_train, y_train = df_train[['token_length', 'token_overlap']], df_train['label']
    X_test, y_test = df_test[['token_length', 'token_overlap']], df_test['label']

    # 4. Majority Prediction Model
    majority_model = DummyClassifier(strategy="most_frequent")
    majority_model.fit(X_train, y_train)
    y_pred_majority = majority_model.predict(X_test)

    # 5. Length-Only Prediction Model
    length_model = LogisticRegression(C=1e6, max_iter=1000)
    length_model.fit(X_train[['token_length']], y_train)
    y_pred_length = length_model.predict(X_test[['token_length']])

    # 6. Overlap-Only Prediction Model
    overlap_model = LogisticRegression(C=1e6, max_iter=1000)
    overlap_model.fit(X_train[['token_overlap']], y_train)
    y_pred_overlap = overlap_model.predict(X_test[['token_overlap']])

    complete_model = LogisticRegression(C=1e6, max_iter=1000)
    complete_model.fit(X_train, y_train)
    y_pred_complete = complete_model.predict(X_test)

    equations = []
    for model in [length_model, overlap_model]:
        intercept = model.intercept_[0]
        coefficients = model.coef_[0]

        # Print the mathematical decision function (z = beta_0 + beta_1 * X_1 + ...)
        terms = [f"{intercept:.4f}"]
        for i, coef in enumerate(coefficients):
            feature_name = "X"
            sign = "+" if coef >= 0 else "-"
            terms.append(f"{sign} {abs(coef):.4f} * {feature_name}")

        equation = "z = " + " ".join(terms)
        equations.append(equation)
    
    # 7. Print Accuracy Results
    if output_path == "console":
        print("Majority baseline")
        print(classification_report(y_test, y_pred_majority))
        print("="*30)
        print("Length baseline")
        print(classification_report(y_test, y_pred_length))
        print(equations[0])
        print("="*30)
        print("Overlap baseline")
        print(classification_report(y_test, y_pred_overlap))
        print(equations[1])
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Majority baseline\n")
            f.write(classification_report(y_test, y_pred_majority))
            f.write("="*30)
            f.write("\nLength baseline\n")
            f.write(classification_report(y_test, y_pred_length))
            f.write("\n"+equations[0])
            f.write("="*30)
            f.write("\nOverlap baseline\n")
            f.write(classification_report(y_test, y_pred_overlap))
            f.write("\n"+equations[1])
            f.write("="*30)
            f.write("\nBoth baselines\n")
            f.write(classification_report(y_test, y_pred_complete))
            # f.write(equations[1])