# Complexification Prompts

WIKI_COMPLEXIFY_DEVELOPER_PROMPT = """You are an expert in drug biology. Given source material on a specific drug, please rewrite it in one paragraph suitable for an expert audience.

TASK GUIDELINES:
1. Do your best to make the material as dense as possible by using complex domain-specific jargon.
2. DO NOT ADD any external information, i.e., strictly utilize the provided context only.
3. Ignore historical details in the text, such as year of discovery, etc., focusing only on the core biological details.
4. Avoid including unnecessary text such as "Here are the key points" or other filler language."""

WIKI_COMPLEXIFY_USER_PROMPT = """DRUG NAME: {}
DRUG INFORMATION: {}"""

# Molecular Relationship Extraction Prompts

MOLECULAR_INTERACTIONS_DEVELOPER_PROMPT = """You are an expert in molecular chemistry and pharmacology. Given the molecular structures of two drugs, represented by their SMILES strings, identify one specific molecular interaction between them using only the structural information provided.

TASK GUIDELINES:
1. Focus only on interactions relevant under physiological conditions (e.g., hydrogen bonding, steric clashes, electrostatic interactions).
2. Do not infer interactions from external knowledge or assumptions about the drug identities.
3. Only report interactions supported by structural features in the SMILES.
4. If no interaction exists, respond with 'NONE'.

For any interaction you do identify, format your response as:

INTERACTION: [Specific name of the interaction]
MECHANISM: [Brief explanation of how/why this interaction occurs]
EVIDENCE: [Direct structural features or groups from the SMILES that support this]
SEVERITY: [Low / Moderate / High - based on likely pharmacological impact]"""

MOLECULAR_INTERACTIONS_USER_PROMPT = """SMILES 1: {}
SMILES 2: {}"""

# DDI Prompts

## Biological relationships

DDI_BIO_ONE_HOP_DEVELOPER_PROMPT = """You are a helpful AI assistant tasked with generating one question about a drug-drug interaction (DDI) based on background information about two drugs and a knowledge-graph triple describing their interaction (subject-predicate-object format).

TASK REQUIREMENTS:
1. Write exactly one question integrating the background knowledge of both drugs and their relationship. The answer must be either Drug 1 or Drug 2.
2. The question may be as complex as desired, but it must be answerable.
3. Do NOT mention the drugs by name in the question; use only their background descriptions.
4. The question should **specifically test knowledge of the triple-described relationship or interaction, not just isolated facts about either drug.**
5. The answer should be only the name of the correct drug.
6. Output in the following format:

Question:
Answer:"""

DDI_BIO_ONE_HOP_USER_PROMPT = """DRUG 1 NAME: {}
DRUG 1 BACKGROUND INFORMATION: {}

DRUG 2 NAME: {}
DRUG 2 BACKGROUND INFORMATION: {}

DRUG-DRUG INTERACTION TRIPLE (subject-predicate-object): {}"""

DDI_BIO_TWO_HOP_DEVELOPER_PROMPT = """You are a helpful AI assistant who writes questions about DRUG-DRUG INTERACTIONS. You are given background information on three drugs and their interaction triples (in subject-predicate-object format), which describe the side effects or outcomes of taking two drugs together. 

TASK REQUIREMENTS:

1. Write exactly one question integrating the background knowledge of each drug and its relationships. The answer must be Drug 3.
2. The question may be as complex as desired, but it must be answerable.
3. Do NOT mention the drugs by name in the question; use only their background descriptions.
4. The question should **specifically test knowledge of the triple-described relationship or interaction, not just isolated facts about the provided drugs.**
5. The answer should be only the name of the correct drug.
6. Output in the following format:

Question:
Answer:"""

DDI_BIO_TWO_HOP_USER_PROMPT = """DRUG 1 NAME: {}
DRUG 1 BACKGROUND INFORMATION: {}

DRUG 2 NAME: {}
DRUG 2 BACKGROUND INFORMATION: {}

DRUG 3 NAME: {}
DRUG 3 BACKGROUND INFORMATION: {}

DRUG 1 - DRUG 2 INTERACTION TRIPLE (subject-predicate-object): {}
DRUG 2 - DRUG 3 INTERACTION TRIPLE (subject-predicate-object): {}"""

DDI_BIO_THREE_HOP_DEVELOPER_PROMPT = """You are a helpful AI assistant who writes questions about DRUG-DRUG INTERACTIONS. You are given background information on four drugs and their interaction triples (in subject-predicate-object format), which describe the side effects or outcomes of taking two drugs together.

TASK REQUIREMENTS:

1. Write exactly one question integrating the background knowledge of each drug and its relationships. The answer must be Drug 4.
2. The question may be as complex as desired, but it must be answerable.
3. Do NOT mention the drugs by name in the question; use only their background descriptions.
4. The question should **specifically test knowledge of the triple-described relationship or interaction, not just isolated facts about the provided drugs.**
5. The answer should be only the name of the correct drug.
6. Output in the following format:

Question:
Answer:"""

DDI_BIO_THREE_HOP_USER_PROMPT = """DRUG 1 NAME: {}
DRUG 1 BACKGROUND INFORMATION: {}

DRUG 2 NAME: {}
DRUG 2 BACKGROUND INFORMATION: {}

DRUG 3 NAME: {}
DRUG 3 BACKGROUND INFORMATION: {}

DRUG 4 NAME: {}
DRUG 4 BACKGROUND INFORMATION: {}

DRUG 1 - DRUG 2 INTERACTION TRIPLE (subject-predicate-object): {}
DRUG 2 - DRUG 3 INTERACTION TRIPLE (subject-predicate-object): {}
DRUG 3 - DRUG 4 INTERACTION TRIPLE (subject-predicate-object): {}"""

## Molecular Relationships

DDI_MOL_ONE_HOP_DEVELOPER_PROMPT = """You are a helpful AI assistant tasked with generating one question about a drug-drug interaction (DDI) based on the molecular structure of two drugs (given as SMILES strings) and a knowledge-graph triple describing their interaction (subject-predicate-object format).

TASK REQUIREMENTS:
1. Write exactly one question integrating the molecular structure of both drugs and their relationship. The answer must be either Drug 1 or Drug 2.
2. The question may be as complex as desired, but must be answerable.
3. Do NOT mention the drugs by name in the question; describe them by their molecular structure.
4. The question should **specifically test knowledge of the triple-described relationship or interaction, not just isolated facts about either drug.**
5. The answer should be only the name of the correct drug.
6. Do not repeat the provided data or add any filler language.
7. Output in the following format:

Question:
Answer:"""

DDI_MOL_ONE_HOP_USER_PROMPT = """DRUG 1 NAME: {}
DRUG 1 SMILES: {}

DRUG 2 NAME: {}
DRUG 2 SMILES: {}

DRUG-DRUG INTERACTION TRIPLE (subject-predicate-object): {}"""

DDI_MOL_TWO_HOP_DEVELOPER_PROMPT = """You are a helpful AI assistant who writes questions about DRUG-DRUG INTERACTIONS. You are given background information on three drugs and their interaction triples (in subject-predicate-object format), which describe how the drugs interact with each other at the molecular level.

TASK REQUIREMENTS:
1. Write exactly one question integrating the molecular structure of each drug and their relationships. The answer must be Drug 3.
2. The question may be as complex as desired, but must be answerable.
3. Do NOT mention the drugs by name in the question; describe them by their molecular structure.
4. The question should **specifically test knowledge of the triple-described relationship or interaction, not just isolated facts about the provided drugs.**
5. The answer should be only the name of the correct drug.
6. Do not repeat the provided data or add any filler language.
7. Output in the following format:

Question:
Answer:"""

DDI_MOL_TWO_HOP_USER_PROMPT = """DRUG 1 NAME: {}
DRUG 1 BACKGROUND INFORMATION: {}

DRUG 2 NAME: {}
DRUG 2 BACKGROUND INFORMATION: {}

DRUG 3 NAME: {}
DRUG 3 BACKGROUND INFORMATION: {}

DRUG 1 - DRUG 2 INTERACTION TRIPLE (subject-predicate-object): {}
DRUG 2 - DRUG 3 INTERACTION TRIPLE (subject-predicate-object): {}"""

DDI_MOL_THREE_HOP_DEVELOPER_PROMPT = """You are a helpful AI assistant who writes questions about DRUG-DRUG INTERACTIONS. You are given background information on four drugs and their interaction triples (in subject-predicate-object format), which describe how the drugs interact with each other at the molecular level.

TASK REQUIREMENTS:
1. Write exactly one question integrating the molecular structure of each drug and their relationships. The answer must be Drug 4.
2. The question may be as complex as desired, but must be answerable.
3. Do NOT mention the drugs by name in the question; describe them by their molecular structure.
4. The question should **specifically test knowledge of the triple-described relationship or interaction, not just isolated facts about the provided drugs.**
5. The answer should be only the name of the correct drug.
6. Do not repeat the provided data or add any filler language.
7. Output in the following format:

Question:
Answer:"""

DDI_MOL_THREE_HOP_USER_PROMPT = """DRUG 1 NAME: {}
DRUG 1 BACKGROUND INFORMATION: {}

DRUG 2 NAME: {}
DRUG 2 BACKGROUND INFORMATION: {}

DRUG 3 NAME: {}
DRUG 3 BACKGROUND INFORMATION: {}

DRUG 4 NAME: {}
DRUG 4 BACKGROUND INFORMATION: {}

DRUG 1 - DRUG 2 INTERACTION TRIPLE (subject-predicate-object): {}
DRUG 2 - DRUG 3 INTERACTION TRIPLE (subject-predicate-object): {}
DRUG 3 - DRUG 4 INTERACTION TRIPLE (subject-predicate-object): {}"""

# DPI Prompts

DPI_ONE_HOP_DEVELOPER_PROMPT = """You are a helpful AI assistant tasked with generating one question about a drug-protein interaction (DPI) based on background information about a drug, protein and a knowledge-graph triple describing their interaction (subject-predicate-object format).

TASK REQUIREMENTS:
1. Write exactly one question integrating the background knowledge of the drug, protein and their relationship. The answer must be the protein.
2. The question may be as complex as desired, but it must be answerable.
3. Do NOT mention the drug or protein by name in the question; use only their background descriptions.
4. The question should **specifically test knowledge of the triple-described relationship or interaction, not just isolated facts about the drug or protein.**
5. The answer should be only the name of the protein.
6. Output in the following format:

Question:
Answer:"""

DPI_ONE_HOP_USER_PROMPT = """DRUG NAME: {}
DRUG BACKGROUND INFORMATION: {}

PROTEIN NAME: {}
PROTEIN BACKGROUND INFORMATION: {}

DRUG-PROTEIN INTERACTION TRIPLE (subject-predicate-object): {}"""

DPI_TWO_HOP_DEVELOPER_PROMPT = """You are a helpful AI assistant who writes questions about DRUG-PROTEIN INTERACTIONS. You are given background information on two drugs, one protein and their interaction triples (in subject-predicate-object format). 

You will see 2 types of relations,
1. DRUG-DRUG INTERACTION: This is the side effect of taking the two drugs together.
2. DRUG-PROTEIN INTERACTION: This explains how the drug behaves with the protein.

TASK REQUIREMENTS:
1. Write exactly one question integrating the background knowledge of each drug, protein and their relationship. The answer must be the protein.
2. The question may be as complex as desired, but it must be answerable.
3. Do NOT mention the drugs or protein by name in the question; use only their background descriptions.
4. The question should **specifically test knowledge of the triple-described relationships or interactions, not just isolated facts about the drugs or protein.**
5. The answer should be only the name of the protein.
6. Output in the following format:

Question:
Answer:"""

DPI_TWO_HOP_USER_PROMPT = """DRUG 1 NAME: {}
DRUG 1 BACKGROUND INFORMATION: {}

DRUG 2 NAME: {}
DRUG 2 BACKGROUND INFORMATION: {}

PROTEIN NAME: {}
PROTEIN BACKGROUND INFORMATION: {}

DRUG 1 - DRUG 2 INTERACTION TRIPLE (subject-predicate-object): {}
DRUG 2 - PROTEIN INTERACTION TRIPLE (subject-predicate-object): {}"""

DPI_THREE_HOP_DEVELOPER_PROMPT = """You are a helpful AI assistant who writes questions about DRUG-PROTEIN INTERACTIONS. You are given background information on three drugs, one protein and their interaction triples (in subject-predicate-object format). 

You will see 2 types of relations,
1. DRUG-DRUG INTERACTION: This is the side effect of taking the two drugs together.
2. DRUG-PROTEIN INTERACTION: This explains how the drug behaves with the protein.

TASK REQUIREMENTS:
1. Write exactly one question integrating the background knowledge of each drug, protein and their relationship. The answer must be the protein.
2. The question may be as complex as desired, but it must be answerable.
3. Do NOT mention the drugs or protein by name in the question; use only their background descriptions.
4. The question should **specifically test knowledge of the triple-described relationships or interactions, not just isolated facts about the drugs or protein.**
5. The answer should be only the name of the protein.
6. Output in the following format:

Question:
Answer:"""

DPI_THREE_HOP_USER_PROMPT = """DRUG 1 NAME: {}
DRUG 1 BACKGROUND INFORMATION: {}

DRUG 2 NAME: {}
DRUG 2 BACKGROUND INFORMATION: {}

DRUG 3 NAME: {}
DRUG 3 BACKGROUND INFORMATION: {}

PROTEIN NAME: {}
PROTEIN BACKGROUND INFORMATION: {}

DRUG 1 - DRUG 2 INTERACTION TRIPLE (subject-predicate-object): {}
DRUG 2 - DRUG 3 INTERACTION TRIPLE (subject-predicate-object): {}
DRUG 3 - PROTEIN INTERACTION TRIPLE (subject-predicate-object): {}"""