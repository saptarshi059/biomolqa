WIKI_COMPLEXIFY_DEVELOPER_PROMPT = """You are an expert in drug biology. Given source material on a specific drug, please rewrite it in one paragraph suitable for an expert audience.

TASK GUIDELINES:
1. Do your best to make the material as dense as possible by using complex domain-specific jargon.
2. DO NOT ADD any external information, i.e., strictly utilize the provided context only.
3. Ignore historical details in the text, such as year of discovery, etc., focusing only on the core biological details.
4. Avoid including unnecessary text such as "Here are the key points" or other filler language."""

WIKI_COMPLEXIFY_USER_PROMPT = """DRUG NAME: {}
DRUG INFORMATION: {}"""

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




DDI_SINGLE_HOP_PROMPT = """TASK: You are a helpful AI assistant who writes questions about DRUG-DRUG INTERACTIONS. You are provided with background information about two drug and how they are related by a knowledge-graph triple (in subject-predicate-object format). Please write 1 question by incorporating the background knowledge of each drug and their relationship such that the answer to the question is either drug 1 or drug 2. You are at complete liberty to make the question as complex as possible but still keeping it answerable.

Avoid including unnecessary text such as "Here are the key points" or other filler language. Please structure your output as,
Question:
Answer:

DRUG 1 NAME: {}
DRUG 1 BACKGROUND INFORMATION: {}

DRUG 2 NAME: {}
DRUG 2 BACKGROUND INFORMATION: {}

DRUG-DRUG INTERACTION TRIPLE: {}"""

DDI_TWO_HOP_PROMPT = """TASK: You are a helpful AI assistant who writes questions about DRUG-DRUG INTERACTIONS. You are given background information on three drugs and their interaction triples (in subject-predicate-object format), which describe the side effects or outcomes of taking two drugs together. 

Your task is to write **one question** that incorporates the background knowledge of each drug and their interactions such that the answer is Drug 3.

Please structure your output as follows:
Question:
Answer:

DRUG 1 NAME: {}
DRUG 1 BACKGROUND INFORMATION: {}

DRUG 2 NAME: {}
DRUG 2 BACKGROUND INFORMATION: {}

DRUG 3 NAME: {}
DRUG 3 BACKGROUND INFORMATION: {}

DRUG 1 - DRUG 2 INTERACTION TRIPLE (subject-predicate-object): {}
DRUG 2 - DRUG 3 INTERACTION TRIPLE (subject-predicate-object): {}
"""