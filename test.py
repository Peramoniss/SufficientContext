import spacy

# Carrega o modelo pré-treinado (exemplo em inglês)
nlp = spacy.load("en_core_web_lg")

# Processa um texto
texto = "Which award the performer of song Picture Book (Song) earned?"
doc = nlp(texto)

# Itera sobre as entidades encontradas
for ent in doc.ents:
  print(ent.text, "->", ent.label_)