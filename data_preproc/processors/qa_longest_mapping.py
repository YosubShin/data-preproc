"""Processor for selecting the longest QA explanation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import DatasetProcessor, register_processor


LOG = logging.getLogger(__name__)


class LongestExplanationMappingProcessor(DatasetProcessor):
    """Select the QA entry with the longest explanation and format the output."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.qa_pairs_field = config.get("qa_pairs_field", "qa_pairs")
        self.question_field = config.get("question_field", "question")
        self.explanation_field = config.get("explanation_field", "explanation")
        self.answer_field = config.get("answer_field", "answer")
        self.problem_field = config.get("problem_field", "problem")
        self.solution_field = config.get("solution_field", "solution")
        self.keep_unmapped = config.get("keep_unmapped", True)
        self.remove_source_fields = config.get("remove_source_fields", False)

    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            source = self._get_source_container(example)
            questions = self._as_list(source.get(self.question_field))
            explanations = self._as_list(source.get(self.explanation_field))
            answers = self._as_list(source.get(self.answer_field))

            if not questions or not explanations or not answers:
                LOG.debug(
                    "Skipping example: missing questions (%s), explanations (%s) or answers (%s)",
                    bool(questions),
                    bool(explanations),
                    bool(answers),
                )
                return None

            if not self._have_matching_lengths(questions, explanations, answers):
                LOG.debug(
                    "Skipping example: length mismatch (questions=%d, explanations=%d, answers=%d)",
                    len(questions),
                    len(explanations),
                    len(answers),
                )
                return None

            index = self._select_longest_index(explanations)
            if index is None:
                LOG.debug("Skipping example: failed to select longest explanation")
                return None

            question = str(questions[index]).strip()
            explanation = str(explanations[index]).strip()
            answer = str(answers[index]).strip()

            if not question or not explanation or not answer:
                LOG.debug(
                    "Skipping example: empty fields after selection (question=%s, explanation=%s, answer=%s)",
                    bool(question),
                    bool(explanation),
                    bool(answer),
                )
                return None

            result = example.copy() if self.keep_unmapped else {}
            result[self.problem_field] = question
            result[self.solution_field] = self._format_solution(explanation, answer)

            if self.remove_source_fields:
                if self.qa_pairs_field:
                    result.pop(self.qa_pairs_field, None)
                else:
                    for field in {self.question_field, self.explanation_field, self.answer_field}:
                        result.pop(field, None)

            return result

        except Exception as exc:  # pragma: no cover - defensive logging
            LOG.warning("Error in longest explanation mapping: %s", exc)
            LOG.debug("Example keys: %s", list(example.keys()))
            return example if self.keep_unmapped else None

    def get_required_columns(self) -> List[str]:
        return [self.qa_pairs_field]

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _have_matching_lengths(*sequences: List[Any]) -> bool:
        lengths = {len(seq) for seq in sequences}
        return len(lengths) == 1

    @staticmethod
    def _select_longest_index(explanations: List[Any]) -> Optional[int]:
        max_length = -1
        max_index = None

        for idx, item in enumerate(explanations):
            if item is None:
                continue
            text = str(item)
            current_length = len(text)
            if current_length > max_length:
                max_length = current_length
                max_index = idx

        return max_index

    @staticmethod
    def _format_solution(explanation: str, answer: str) -> str:
        return f"{explanation}\n\nThe answer is \\boxed{{{answer}}}."

    def _get_source_container(self, example: Dict[str, Any]) -> Dict[str, Any]:
        qa_pairs = example.get(self.qa_pairs_field)
        if isinstance(qa_pairs, dict):
            return qa_pairs
        LOG.debug(
            "qa_pairs field '%s' missing or not dict (type=%s)",
            self.qa_pairs_field,
            type(qa_pairs).__name__ if qa_pairs is not None else "None",
        )
        return {}


register_processor("longest_explanation_mapping", LongestExplanationMappingProcessor)

