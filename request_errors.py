"""Bounded recovery for temporary model failures and safe messages for the UI."""
from dataclasses import dataclass
import random
import re
from time import sleep


@dataclass(frozen=True)
class RequestIssue:
    message: str
    temporary: bool = False


class AnswerUnavailable(Exception):
    def __init__(self, issue):
        self.issue = issue
        super().__init__(issue.message)


def classify_error(error):
    """Inspect status codes through provider wrappers without exposing response data."""
    current = error
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        candidates = (
            getattr(current, "code", None), getattr(current, "status_code", None),
            getattr(getattr(current, "response", None), "status_code", None),
        )
        code = next((int(value) for value in candidates if isinstance(value, (int, str)) and str(value).isdigit()), None)
        if code is None:
            match = re.search(r"\b(400|401|403|404|408|429|500|502|503|504)\b", str(current))
            code = int(match.group(1)) if match else None
        if code in (500, 502, 503, 504):
            return RequestIssue("The model is busy or temporarily unavailable. Your question is saved. Wait a moment, then try again.", True)
        if code == 429:
            return RequestIssue("Your API rate limit or quota has been reached. Check your Google AI Studio limits, then try again when capacity is available.")
        if code in (401, 403):
            return RequestIssue("The model could not authorize this request. Check your API key and model access in the left panel, then try again.")
        if code == 404:
            return RequestIssue("This model is not available for your API key. Check the model name and your account's access in Model & search settings.")
        if code in (400, 408):
            return RequestIssue("The model could not complete this request. Try a shorter question and check the selected model's settings.")
        current = current.__cause__ or current.__context__
    return RequestIssue("Something went wrong while preparing the answer. Your question is saved. Try again or ask a shorter question.")


def answer_with_retry(operation, on_retry=None):
    """Make at most three attempts; retry only temporary server failures.

    Provider SDKs may also retry individual calls internally. This bounds the
    app-level retries and never changes the selected model or API key.
    """
    for attempt in range(3):
        try:
            return operation()
        except Exception as exc:
            issue = classify_error(exc)
            if not issue.temporary or attempt == 2:
                raise AnswerUnavailable(issue) from exc
            delay = (2 ** attempt) + random.uniform(0, 0.5)
            if on_retry is not None:
                on_retry(attempt + 2)
            sleep(delay)
