# HalluCodeDetector
Here is the relevant code and data for HalluCodeDetector.

## HalluCodeDetector
This folder contains the code for the experiment.

The comments in the getBLEUScores/getCodeBLEUScores/getMRCMScores/getPromptScores/getOrdinaryLLMScores/getLynxScores function in HalluCodeDetectorWithBLEU.py/HalluCodeDetectorWithCodeBLEU.py/HalluCodeDetectorWithMRCM.py/HalluCodeDetectorWithPrompt.py/OrdinaryLLM.py/Lynx.py record our experimental results

**configuration.** Modify your custom configuration in main.py:
* testsetPath: The path to the Projects folder. (Projects folder is provided in master branch, it contains only two samples. You need to run run_setup.py to supplement the remaining samples)
* outputPath: The path to the output folder where the generated prompt are stored. (We provided our expiremental results in PromptAndResult folder in master branch)
* jsonFilePath: The path to test.json. (test.json is provided in master branch)

## Datasets
This folder contains the jsonl files of the HumanEval and mbpp datasets.

For the HumanEval dataset, we use the content of the "prompt" section to get the answer of the LLM.

For the mbpp dataset, we concatenate the contents of the "prompt" and "code" sections and use this content to get the answer of the LLM.

## LLMAnswers
This folder contains DeepSeek-V3's answers to the HumanEval and mbpp datasets.
