# Valid (connected entities)

VALID_TRI_MODAL_FF = """For a pair of drugs, you are given their SMILES representations, background information on each, and a knowledge-graph interaction triple that describes the side-effect of taking both drugs together. I want you to write five free-form question-answer pairs utilizing this information.

General Requirements:
1. Please do not include references to the provided data in either the question or answer such as "based on the given information", "according to the knowledge-graph triple", etc.
2. It is very important that you do not make any assumptions of your own and strictly utilize the provided information.

Question Requirements:
1. Each question MUST include the SMILES, background text and knowledge-graph triple. 
2. The questions should NOT state the drug name because I want the test taker to figure out the drug based on its SMILES. 
3. Each question should be self-contained, i.e., it can be answered by studying the provided information only.
4. Utilize as much of the provided text data as you can to create the questions.
5. Utilize different parts of the text to create a variety of questions.

Answer Requirements:
1. Please keep length of answers to less than 100 words. 
2. Utilize evidence from the provided information to create your answer instead of simply restating the knowledge-graph triple as the answer.

Please structure your output as follows,
Question: <question text>
Answer: <answer text>

Drug 1 name: {}
Drug 1 SMILES: {}
Drug 1 background information: {}

Drug 2 name: {}
Drug 2 SMILES: {}
Drug 2 background information: {}

Drug-Drug interaction triple: {}
"""

VALID_TRI_MODAL_MCQA = """For a pair of drugs, you are given their SMILES representations, background information on each, and a knowledge-graph interaction triple that describes the side-effect of taking both drugs together. I want you to write five multiple-choice question-answer pairs utilizing this information.

General Requirements:
1. Please do not include references to the provided data in either the question or answer such as "based on the given information", "according to the knowledge-graph triple", etc.
2. It is very important that you do not make any assumptions of your own and strictly utilize the provided information.

Question Requirements:
1. Each question MUST include the SMILES, background text and knowledge-graph triple. 
2. The questions should NOT state the drug name because I want the test taker to figure out the drug based on its SMILES. 
3. Each question should be self-contained, i.e., it can be answered by studying the provided information only.

Answer Requirements:
1. The answers should be based directly on the provided information.

Please structure your output as follows,
Question: <question text>
Choices: <answer choices labelled as (a), (b), (c), (d)>
Correct choice: <among (a), (b), (c) or (d)>

Drug 1 name: {}
Drug 1 SMILES: {}
Drug 1 background information: {}

Drug 2 name: {}
Drug 2 SMILES: {}
Drug 2 background information: {}

Drug-Drug interaction triple: {}
"""

VALID_BI_MODAL_FF = """For a pair of drugs, you are given background information on each and a knowledge-graph interaction triple that describes the side-effect of taking both drugs together. I want you to write five free-form question-answer pairs utilizing this information.

General Requirements:
1. Please do not include references to the provided data in either the question or answer such as "based on the given information", "according to the knowledge-graph triple", etc.
2. It is very important that you do not make any assumptions of your own and strictly utilize the provided information.

Question Requirements:
1. Each question MUST incorporate background information on each drug and the knowledge-graph triple. 
2. Each question should be self-contained, i.e., it can be answered by studying the provided information only.

Answer Requirements:
1. Please keep length of answers to less than 100 words. 
2. The answers should be based directly on the provided information.

Please structure your output as follows,
Question: <question text>
Answer: <answer text>

Drug 1 name: {}
Drug 1 background information: {}

Drug 2 name: {}
Drug 2 background information: {}

Drug-Drug interaction triple: {}
"""

VALID_BI_MODAL_MCQA = """For a pair of drugs, you are given background information on each and a knowledge-graph interaction triple that describes the side-effect of taking both drugs together. I want you to write five multiple-choice question-answer pairs utilizing this information.

General Requirements:
1. Please do not include references to the provided data in either the question or answer such as "based on the given information", "according to the knowledge-graph triple", etc.
2. It is very important that you do not make any assumptions of your own and strictly utilize the provided information.

Question Requirements:
1. Each question MUST incorporate background information on each drug and the knowledge-graph triple. 
2. Each question should be self-contained, i.e., it can be answered by studying the provided information only.

Answer Requirements:
1. The answers should be based directly on the provided information.

Please structure your output as follows,
Question: <question text>
Choices: <answer choices labelled as (a), (b), (c), (d)>
Correct choice: <among (a), (b), (c) or (d)>

Drug 1 name: {}
Drug 1 background information: {}

Drug 2 name: {}
Drug 2 background information: {}

Drug-Drug interaction triple: {}
"""

# Invalid (disconnected entities)

INVALID_TRI_MODAL_FF = """For a pair of drugs, you are given their SMILES representations, background information on each, and a knowledge-graph interaction triple that describes the side-effect of taking both drugs together. I want you to write five free-form question-answer pairs utilizing this information.

General Requirements:
1. Please do not include references to the provided data in either the question or answer such as "based on the given information", "according to the knowledge-graph triple", etc.
2. It is very important that you do not make any assumptions of your own and strictly utilize the provided information.

Question Requirements:
1. Each question MUST include the SMILES, background text and knowledge-graph triple. 
2. The questions should NOT state the drug name because I want the test taker to figure out the drug based on its SMILES. 
3. Each question should be self-contained, i.e., it can be answered by studying the provided information only.

Answer Requirements:
1. Please keep length of answers to less than 100 words. 
2. The answers should be based directly on the provided information.

Please structure your output as follows,
Question: <question text>
Answer: <answer text>

Drug 1 name: {}
Drug 1 SMILES: {}
Drug 1 background information: {}

Drug 2 name: {}
Drug 2 SMILES: {}
Drug 2 background information: {}

Drug-Drug interaction triple: {}
"""

INVALID_TRI_MODAL_MCQA = """For a pair of drugs, you are given their SMILES representations, background information on each, and a knowledge-graph interaction triple that describes the side-effect of taking both drugs together. I want you to write five multiple-choice question-answer pairs utilizing this information.

General Requirements:
1. Please do not include references to the provided data in either the question or answer such as "based on the given information", "according to the knowledge-graph triple", etc.
2. It is very important that you do not make any assumptions of your own and strictly utilize the provided information.

Question Requirements:
1. Each question MUST include the SMILES, background text and knowledge-graph triple. 
2. The questions should NOT state the drug name because I want the test taker to figure out the drug based on its SMILES. 
3. Each question should be self-contained, i.e., it can be answered by studying the provided information only.

Answer Requirements:
1. The answers should be based directly on the provided information.

Please structure your output as follows,
Question: <question text>
Choices: <answer choices labelled as (a), (b), (c), (d)>
Correct choice: <among (a), (b), (c) or (d)>

Drug 1 name: {}
Drug 1 SMILES: {}
Drug 1 background information: {}

Drug 2 name: {}
Drug 2 SMILES: {}
Drug 2 background information: {}

Drug-Drug interaction triple: {}
"""

INVALID_BI_MODAL_FF = """For a pair of drugs, you are given background information on each and a knowledge-graph interaction triple that describes the side-effect of taking both drugs together. I want you to write five free-form question-answer pairs utilizing this information.

General Requirements:
1. Please do not include references to the provided data in either the question or answer such as "based on the given information", "according to the knowledge-graph triple", etc.
2. It is very important that you do not make any assumptions of your own and strictly utilize the provided information.

Question Requirements:
1. Each question MUST incorporate background information on each drug and the knowledge-graph triple. 
2. Each question should be self-contained, i.e., it can be answered by studying the provided information only.

Answer Requirements:
1. Please keep length of answers to less than 100 words. 
2. The answers should be based directly on the provided information.

Please structure your output as follows,
Question: <question text>
Answer: <answer text>

Drug 1 name: {}
Drug 1 background information: {}

Drug 2 name: {}
Drug 2 background information: {}

Drug-Drug interaction triple: {}
"""

INVALID_BI_MODAL_MCQA = """For a pair of drugs, you are given background information on each and a knowledge-graph interaction triple that describes the side-effect of taking both drugs together. I want you to write five multiple-choice question-answer pairs utilizing this information.

General Requirements:
1. Please do not include references to the provided data in either the question or answer such as "based on the given information", "according to the knowledge-graph triple", etc.
2. It is very important that you do not make any assumptions of your own and strictly utilize the provided information.

Question Requirements:
1. Each question MUST incorporate background information on each drug and the knowledge-graph triple. 
2. Each question should be self-contained, i.e., it can be answered by studying the provided information only.

Answer Requirements:
1. The answers should be based directly on the provided information.

Please structure your output as follows,
Question: <question text>
Choices: <answer choices labelled as (a), (b), (c), (d)>
Correct choice: <among (a), (b), (c) or (d)>

Drug 1 name: {}
Drug 1 background information: {}

Drug 2 name: {}
Drug 2 background information: {}

Drug-Drug interaction triple: {}
"""

