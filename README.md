## LLM for AVL

### Requirements
- accelerate                0.24.1
- datasets                  2.15.0
- evaluate                  0.4.1
- huggingface-hub           0.19.4
- nltk                      3.8.1
- numpy                     1.26.0
- pandas                    2.2.0
- peft                      0.6.2
- python                    3.10.13
- pytorch                   2.0.0
- requests                  2.31.0
- scikit-learn              1.3.2
- seqeval                   1.2.2
- tokenizers                0.15.0
- tqdm                      4.66.1
- transformers              4.35.2
- wandb                     0.16.1

### Step-by-Step Reproduction Guide

1. **Zero-Shot and One-Shot Learning:**  
   To experiment with zero-shot or one-shot learning, navigate to the specific project directory (e.g., ChatGPT) and execute the corresponding scripts.  
   - Zero-shot learning:  
     ```
     python prompt.py
     ```
   - One-shot learning:  
     ```
     python one-shot.py
     ```

2. **Discriminative Fine-Tuning for Open-Source LLMs (e.g., CodeBERT):**  
   For applying discriminative fine-tuning techniques to large language models, use the following commands:  
   - Fine-tuning:  
     ```
     python ls_finetune.py
     ```
   - Inference:  
     ```
     python ls_infer.py
     ```

3. **Generative Fine-Tuning:**  
   For models requiring generative fine-tuning, the respective scripts are as follows:  
   - Fine-tuning:  
     ```
     gen_finetune.py
     ```
   - Inference:  
     ```
     gen_infer.py
     ```

4. **Evaluating Performance Across Common Weakness Enumerations (CWEs):**  
   To assess model performance across various CWEs, run:
   ```
   python cwe_dist.py
   ```

6. **Cross-Project Evaluation:**  
    For evaluating models across different projects, the designated scripts are:  
    - Fine-tuning:  
      ```
      cross_ls_finetune.py
      ```
    - Inference:  
      ```
      cross_ls_infer.py
      ```

7. **Improvement Strategies:**  
    Implement various improvement strategies using the following scripts:  
    - For sliding window techniques (SWT and SWI):  
      ```
      sliding_ls_finetune.py
      sliding_ls_infer.py
      ```
    - Specifically for SWI:  
      ```
      swi_ls_infer.py
      ```
    - For right-forward embedding:  
      ```
      fu_ls_finetune.py
      fu_ls_infer.py
      ```
    - Implementing BAL:  
      ```
      bi_ls_finetune.py
      bi_ls_infer.py
      ```

