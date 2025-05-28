# HalluCodeDetector
Here is the relevant code and data for HalluCodeDetector.

## HalluCodeDetector
This folder contains the code for the experiment.

The comments in the getBLEUScores/getCodeBLEUScores/getMRCMScores/getPromptScores/getOrdinaryLLMScores/getLynxScores function in HalluCodeDetectorWithBLEU.py/HalluCodeDetectorWithCodeBLEU.py/HalluCodeDetectorWithMRCM.py/HalluCodeDetectorWithPrompt.py/OrdinaryLLM.py/Lynx.py record the experimental results of different methods on different datasets.

**configuration.** Modify your custom configuration in the main function of main.py:
* base_url: Used to specify the base URL for calling the OpenAI API
* api_key: For OpenAI servers to verify identity
* question_jsonl_path: Path to Datasets\\human-eval.jsonl or Datasets\\mbpp.jsonl
* filepath: Path to LLMAnswers\\HumanEval_deepseek-v3_output.json or LLMAnswers\\HumanEval_deepseek-v3_output.json

Run main.py to get the experimental results.

## Datasets
This folder contains the jsonl files of the HumanEval and mbpp datasets.

For the HumanEval dataset, we use the content of the "prompt" section to get the answer of the LLM.

For the mbpp dataset, we concatenate the contents of the "prompt" and "code" sections and use this content to get the answer of the LLM.

## LLMAnswers
This folder contains DeepSeek-V3's answers to the HumanEval and mbpp datasets.
