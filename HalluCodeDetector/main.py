import json
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from openai import OpenAI
from HalluCodeDetectorWithBLEU import getBLEUScores
from HalluCodeDetectorWithCodeBLEU import getCodeBLEUScores
from HalluCodeDetectorWithMRCM import getMRCMScores
from OrdinaryLLM import getOrdinaryLLMScores
from HalluCodeDetectorWithPrompt import getPromptScores
from Lynx import getLynxScores

def compute_aupr(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = auc(recall, precision)
    return precision, recall, aupr

def compute_auroc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def paint_PRC(y_true, scores_BLEU=None, scores_CodeBLEU=None, scores_Prompt=None, scores_MRCM=None, scores_OrdinaryLLM=None, scores_LYNX=None):
    plt.figure(figsize=(8, 6))
    if scores_BLEU is not None:
        precision, recall, aupr = compute_aupr(y_true, scores_BLEU)
        plt.plot(recall, precision, color='b', label=f'HalluCD-BLEU (AUPRC = {aupr:.2f})')
    if scores_CodeBLEU is not None:
        precision, recall, aupr = compute_aupr(y_true, scores_CodeBLEU)
        plt.plot(recall, precision, color='g', label=f'HalluCD-CodeBLEU (AUPRC = {aupr:.2f})')
    if scores_Prompt is not None:
        precision, recall, aupr = compute_aupr(y_true, scores_Prompt)
        plt.plot(recall, precision, color='pink', label=f'HalluCD-Prompt (AUPRC = {aupr:.2f})')
    if scores_MRCM is not None:
        precision, recall, aupr = compute_aupr(y_true, scores_MRCM)
        plt.plot(recall, precision, color='r', label=f'HalluCD-MRCM (AUPRC = {aupr:.2f})')
    if scores_OrdinaryLLM is not None:
        precision, recall, aupr = compute_aupr(y_true, scores_OrdinaryLLM)
        plt.plot(recall, precision, color='m', label=f'OrdinaryLLM (AUPRC = {aupr:.2f})')
    if scores_LYNX is not None:
        precision, recall, aupr = compute_aupr(y_true, scores_LYNX)
        plt.plot(recall, precision, color='c', label=f'LYNX (AUPRC = {aupr:.2f})')

    positive_ratio = sum(y_true) / len(y_true)
    plt.plot([0, 1], [positive_ratio, positive_ratio], linestyle='--', color='gray', label=f'Random (AUPR = {positive_ratio:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('PRC Comparison', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([positive_ratio-0.05, 1.0])
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

def paint_ROC(y_true, scores_BLEU=None, scores_CodeBLEU=None, scores_Prompt=None, scores_MRCM=None, scores_OrdinaryLLM=None, scores_LYNX=None):
    plt.figure(figsize=(8, 6))
    if scores_BLEU is not None:
        fpr, tpr, auc = compute_auroc(y_true, scores_BLEU)
        plt.plot(fpr, tpr, color='b', label=f'HalluCD-BLEU (AUROC = {auc:.2f})')
    if scores_CodeBLEU is not None:
        fpr, tpr, auc = compute_auroc(y_true, scores_CodeBLEU)
        plt.plot(fpr, tpr, color='g', label=f'HalluCD-CodeBLEU (AUROC = {auc:.2f})')
    if scores_Prompt is not None:
        fpr, tpr, auc = compute_auroc(y_true, scores_Prompt)
        plt.plot(fpr, tpr, color='pink', label=f'HalluCD-Prompt (AUROC = {auc:.2f})')
    if scores_MRCM is not None:
        fpr, tpr, auc = compute_auroc(y_true, scores_MRCM)
        plt.plot(fpr, tpr, color='r', label=f'HalluCD-MRCM (AUROC = {auc:.2f})')
    if scores_OrdinaryLLM is not None:
        fpr, tpr, auc = compute_auroc(y_true, scores_OrdinaryLLM)
        plt.plot(fpr, tpr, color='m', label=f'OrdinaryLLM (AUROC = {auc:.2f})')
    if scores_LYNX is not None:
        fpr, tpr, auc = compute_auroc(y_true, scores_LYNX)
        plt.plot(fpr, tpr, color='c', label=f'LYNX (AUROC = {auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label=f'Random (AUROC = 0.50)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Comparison', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    base_url = "" # A variable you need to set
    api_key = "" # A variable you need to set
    model = "GPT-4o" # The LLM for HalluCodeDetectorWithPrompt and OrdinaryLLM
    client = OpenAI(api_key=api_key, base_url=base_url)

    question_jsonl_path = r"..\\Datasets\\human-eval.jsonl" # Path to Datasets\\human-eval.jsonl or Datasets\\mbpp.jsonl
    filepath = r"D:\桌面\HumanEval_deepseek-v3-250324_output.json" # Path to LLMAnswers\\HumanEval_deepseek-v3_output.json or LLMAnswers\\HumanEval_deepseek-v3_output.json

    size_per_question = 10

    scores_BLEU = getBLEUScores(filepath, size_per_question)
    scores_CodeBLEU = getCodeBLEUScores(filepath, size_per_question)
    scores_Prompt = getPromptScores(filepath, size_per_question, base_url, api_key, model)
    scores_MRCM = getMRCMScores(filepath, size_per_question)
    scores_OrdinaryLLM = getOrdinaryLLMScores(question_jsonl_path, filepath, size_per_question, base_url, api_key, model)
    scores_LYNX = getLynxScores(question_jsonl_path, filepath, size_per_question)

    y_true = []
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for question_id in range(0, int(len(data)/size_per_question)):
        if data[str(question_id * size_per_question + 0)]["test_result"] == "True":
            y_true.append(1)
        else:
            y_true.append(0)

    paint_PRC(y_true, scores_BLEU=scores_BLEU, scores_CodeBLEU=scores_CodeBLEU, scores_Prompt=scores_Prompt, scores_MRCM=scores_MRCM, scores_OrdinaryLLM=scores_OrdinaryLLM, scores_LYNX=scores_LYNX)
    paint_ROC(y_true, scores_BLEU=scores_BLEU, scores_CodeBLEU=scores_CodeBLEU, scores_Prompt=scores_Prompt, scores_MRCM=scores_MRCM, scores_OrdinaryLLM=scores_OrdinaryLLM, scores_LYNX=scores_LYNX)
