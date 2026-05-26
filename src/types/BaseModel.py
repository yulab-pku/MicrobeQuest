from typing import Dict, Any, List
from abc import ABC, abstractmethod
import yaml
import os


class BaseModel(ABC):
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.prompt_template_path = r'src\models\shared\prompts\TaskPromptTemplate.yaml'

        # Load prompt templates from YAML
        if os.path.exists(self.prompt_template_path):
            with open(self.prompt_template_path, 'r') as f:
                self.prompt_templates = yaml.safe_load(f)
        else:
            self.prompt_templates = {}


    def format_result(self, test_case: dict, predicted_answer: str,
                      prompt: Dict[str, Any], response_time: float,
                      token_count: Dict[str, int]) -> Dict[str, Any]:
        return {
            "model_id": self.model_name,
            "case_id": test_case["case_id"],
            "prompt": prompt,
            "predicted_answer": predicted_answer,
            "expected_answer": test_case["test_case"]["expected_answer"],
            "response_time": round(response_time * 1000, 2),  # ms
            "token_count": token_count
        }

    def determine_prompt_template(self, task_type: str) -> str:
        """
        Select a prompt template based on task type.
        """
        default_prompt = (
            "You are a helpful assistant. I will provide you with a passage of text. "
            "Your task is to carefully analyze the content and answer the question. "
            "Please answer the following question: {question}. "
            "Follow the reasoning steps provided below to complete the task: {note}. "
            "Present your final answer within the <Answer> tags as shown below: <Answer> your_answer_here </Answer>"
        )
        return self.prompt_templates.get(task_type, default_prompt)

    def format_prompt(self, test_case: Dict) -> Dict[str, Any]:
        """
        Format the system and user prompt for OpenAI Chat API using test_case info.

        Args:
            test_case: The current test case containing 'question', 'note'.

        Returns:
            A dictionary with 'system' and 'user' messages.
        """
        question = test_case["test_case"]["question"]
        note = test_case["test_case"].get("note", "")

        task_type = test_case.get("task_subcategory", "default")

        # Load corresponding prompt template
        system_prompt = self.determine_prompt_template(task_type)

        system_prompt = system_prompt.format(question=question, note=note)

        # Format few-shot examples if any
        few_shot_examples = ""
        if "few_shot_examples" in test_case["test_case"]:
            few_shot_examples = "EXAMPLES:\n"
            for i, ex in enumerate(test_case["test_case"]["few_shot_examples"], 1):
                # Format each example properly
                few_shot_examples += f"Example {i}:\nQuestion: {ex['question']}\nText: {ex['text']}\nAnswer: {ex['expected_answer']}\n\n"
        return {
            "system": {"role": "system", "content": system_prompt},
            "user": {"role": "user", "content": few_shot_examples}
        }
