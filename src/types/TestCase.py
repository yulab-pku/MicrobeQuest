import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict
from pathlib import Path

from enum import Enum

# Top-level task categories
class TaskCategory(str, Enum):
    STRUCTURED_INFO_EXTRACTION = "Structured Information Extraction"  # Extraction of structured data
    MULTIMODAL_UNDERSTANDING = "Multimodal Understanding"             # Fusion of textual and visual information
    ADVANCED_SEMANTIC_REASONING = "Complex Semantic Reasoning"        # Advanced inference tasks
    LAYOUT_ANALYSIS = "Layout Structure and Semantic Region Recognition "  # Document layout and semantic segmentation

# Subcategories for structured information extraction
class StructInfoSubcategory(str, Enum):
    STRAIN_NAME_NORMALIZATION = "Strain Entity Recognition and Normalization"  # Recognizing and normalizing strain names
    STRAIN_EQUIVALENCE = "Strain Entity Resolution"                            # Determining equivalence between strain entities
    TAXONOMY_EXTRACTION = "Strain Taxonomy Extraction"                         # Extracting taxonomic classification
    PHYSIOLOGICAL_TRAITS = "Strain Physiological Characteristic Extraction"    # Extracting physiological traits
    ENVIRONMENTAL_PARAMETERS = "Environmental Growth Parameter Extraction"     # Extracting environmental growth parameters
    ATTRIBUTE_SEMANTIC_CLASSIFICATION = "Strain Attribute Semantic Categorization"  # Semantic categorization of strain attributes
    MEDIUM_CONDITIONS = "Strain Culture Medium and Growth Condition Extraction"    # Extracting culture medium and growth conditions

# Subcategories for multimodal understanding
class MultimodalSubcategory(str, Enum):
    TABLE_QA = "Table-based Strain Attribute Extraction"         # Question answering based on strain attribute tables
    FIGURE_QA = "Figure-based Strain Attribute Extraction"       # Question answering based on figures
    MULTIMODAL_REASONING = "Multimodal Strain Attribute Reasoning"  # Reasoning with combined textual and visual inputs

# Subcategories for advanced semantic reasoning
class ReasoningSubcategory(str, Enum):
    SUBJECT_ATTRIBUTION = "Multi-Entity Attribute Association"      # Associating attributes with multiple entities
    VALUE_PRIORITY = "Multi-value Priority Resolution"              # Resolving value priority among multiple candidates
    NEGATION_CONTRAST = "Negation and Contrast Relationship Parsing"  # Parsing negation and contrastive relations
    CONDITIONAL_LOGIC = "Logical Condition Reasoning"               # Reasoning with conditional statements
    CROSS_PARAGRAPH_ENTITY_TRACKING = "Cross-Paragraph Entity Tracking"  # Tracking entities across paragraphs
    IMPLICIT_CONCLUSIONS = "Implicit Conclusion Generation"         # Generating implicit conclusions
    MULTI_INSTANCE_COMPARISON = "Multi-Instance Comparative Reasoning"  # Comparing across multiple instances

# Subcategories for layout and structural analysis
class LayoutSubcategory(str, Enum):
    SEMANTIC_REGION_ANNOTATION = "Semantic Document Region Extraction"  # Annotating semantic regions in documents

# Supported answer types for test cases
class AnswerType(str, Enum):
    NUMERIC = "Numeric"                # Numeric value
    CATEGORICAL = "Categorical"        # Categorical label
    BOOLEAN = "Boolean"                # Boolean value (True/False)
    ENTITY_LIST = "Entity List"        # List of entities
    DESCRIPTION = "Descriptive Statement"  # Descriptive/narrative answer

# Difficulty levels for test cases
class DifficultyType(str, Enum):
    Easy = "Easy"
    Medium = "Medium"
    Hard = "Hard"

# Test case definition, including inputs and expected outputs
@dataclass
class TestCase:
    input: Dict[str, any]  # Contains fields like paper_id, image_path, is_full_pdf
    question: str
    note: str
    expected_answer_type: str
    expected_answer: str

# Metadata wrapper for each test case, including task and version info
@dataclass
class TestCaseMetadata:
    case_id: str
    source: str
    task_category: TaskCategory
    task_subcategory: str
    test_case: TestCase
    difficulty: DifficultyType
    version: str = "1.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Serialize test case to JSON, optionally save to file
    def to_json(self, file_path: Optional[str] = None) -> str:
        data = {
            "case_id": self.case_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "difficulty": self.difficulty,
            "task_category": self.task_category,
            "task_subcategory": self.task_subcategory,
            "test_case": self.test_case
        }
        if file_path:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False, indent=2)
