import unittest
from unittest.mock import Mock, patch

from request_errors import AnswerUnavailable, answer_with_retry, classify_error


class ServerError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(f"{code} UNAVAILABLE: request contained secret-test-key")


class RecoveryTests(unittest.TestCase):
    @patch("request_errors.time.sleep")
    @patch("request_errors.random.uniform", return_value=0)
    def test_transient_errors_back_off_and_return_success(self, jitter, sleep):
        request = Mock(side_effect=[ServerError(503), ServerError(502), "answer"])
        notify = Mock()
        self.assertEqual(answer_with_retry(request, notify), "answer")
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.assertEqual([call.args[0] for call in notify.call_args_list], [2, 3])

    @patch("request_errors.time.sleep")
    def test_persistent_failure_stops_after_three_attempts(self, sleep):
        request = Mock(side_effect=ServerError(503))
        with self.assertRaises(AnswerUnavailable) as caught:
            answer_with_retry(request)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertNotIn("secret-test-key", str(caught.exception))
        self.assertNotIn("503", str(caught.exception))

    @patch("request_errors.time.sleep")
    def test_key_quota_and_configuration_errors_are_not_retried(self, sleep):
        for code in (400, 401, 403, 404, 429):
            with self.subTest(code=code):
                request = Mock(side_effect=ServerError(code))
                with self.assertRaises(AnswerUnavailable):
                    answer_with_retry(request)
                self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_wrapped_sdk_error_is_recognized(self):
        inner = ServerError(503)
        outer = RuntimeError("Agent failed")
        outer.__cause__ = inner
        self.assertTrue(classify_error(outer).temporary)

    def test_text_only_wrapper_from_screenshot_is_recognized(self):
        error = RuntimeError("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'high demand'}}")
        self.assertTrue(classify_error(error).temporary)

    def test_unknown_errors_do_not_expose_raw_details(self):
        issue = classify_error(ValueError("secret-test-key document-private.pdf"))
        self.assertFalse(issue.temporary)
        self.assertNotIn("secret-test-key", issue.message)
        self.assertNotIn("document-private.pdf", issue.message)
