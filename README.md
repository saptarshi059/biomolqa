![Title Screenshot](our_paper/title_screenshot.png)

This repository contains the code for our dataset _BioMol-MQA: A Multi Modal Question Answering Dataset For LLM Reasoning
Over Bio Molecular Interactions_. BioMol-MQA is an _LLM-generated_ dataset whose queries involve reasoning over three
distinct modalities, viz., knowledge-graph, raw text and molecular SMILES (Simplified Molecular Input Line Entry System)
strings. The questions in our dataset are related to _polypharmacy_, i.e., the phenomena of taking two or more drugs to 
treat multiple ailments. Polypharmacy is a serious issue as taking the wrong drug combination can prove fatal. Additionally,
as more people use LLMs to answer medical queries, they need to be robust at handling such polypharmacy-related questions.
As such, our dataset provides a good testbed to evaluate model performance in dealing with this topic.

### Dataset Sample

The following instance is a sample from the _training_ split of our dataset.

```yaml
Entities: Famotidine-Metolazone
Question_Background: |
    DRUG 1 NAME: Famotidine
    DRUG 1 BACKGROUND INFORMATION: Famotidine, sold under the brand name Pepcid among 
  others, is a histamine H2 receptor antagonist medication that decreases stomach 
  acid production. It is used to treat peptic ulcer disease, gastroesophageal reflux 
  disease, and Zollinger–Ellison syndrome. It is taken by mouth or by injection into a 
  vein. It begins working within an hour. Common side effects include headache, 
  abdominal pain, diarrhea or constipation, and dizziness. Serious side effects may 
  include pneumonia and seizures. Use in pregnancy appears safe but has not been well 
  studied, while use during breastfeeding is not recommended. Famotidine was patented 
  in 1979 and came into medical use in 1985. It is available as a generic medication. 
  In 2022, it was the 49th most commonly prescribed medication in the United States, 
  with more than 13 million prescriptions.
    
    DRUG 2 NAME: Metolazone
    DRUG 2 BACKGROUND INFORMATION: Metolazone is a thiazide-like diuretic marketed 
  under the brand names Zytanix, Metoz, Zaroxolyn, and Mykrox. It is primarily used 
  to treat congestive heart failure and high blood pressure. Metolazone indirectly 
  decreases the amount of water reabsorbed into the bloodstream by the kidney, so 
  that blood volume decreases and urine volume increases. This lowers blood pressure 
  and prevents excess fluid accumulation in heart failure. Metolazone is sometimes 
  used together with loop diuretics such as furosemide or bumetanide, but these 
  highly effective combinations can lead to dehydration and electrolyte 
  abnormalities. It was patented in 1966 and approved for medical use in 1974.
    
    DRUG-DRUG INTERACTION TRIPLE (subject-predicate-object): Famotidine-
                                                             right heart failure-
                                                             Metolazone
    
Question: Which medication, either a histamine H2 receptor antagonist that decreases 
  stomach acid production or a thiazide-like diuretic primarily used for congestive 
  heart failure and hypertension, is specifically associated with the management of 
  right heart failure in combination with the other, according to known drug 
  interactions?
Answer: Metolazone
Label: 0
```










### Citation
If you found our dataset useful for your project, please consider citing it 😄

```txt
@article{sengupta2025biomol,
  title={BioMol-MQA: A Multi-Modal Question Answering Dataset For LLM Reasoning Over Bio-Molecular Interactions},
  author={Sengupta, Saptarshi and Yang, Shuhua and Yu, Paul Kwong and Wang, Fali and Wang, Suhang},
  journal={arXiv preprint arXiv:2506.05766},
  year={2025}
}
```