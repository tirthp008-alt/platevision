"""Console encoding must not break logging or bypass plate redaction."""
import io
import logging
import uuid

import pytest

from app.core import logging as app_logging


@pytest.mark.parametrize('encoding', ['cp1252', 'utf-8'])
def test_console_handles_unicode_after_redaction(monkeypatch, capsys, encoding):
    output = io.BytesIO()
    stream = io.TextIOWrapper(output, encoding=encoding, errors='strict')
    monkeypatch.setattr(app_logging.sys, 'stdout', stream)
    monkeypatch.setattr(app_logging.settings, 'ENABLE_REDACTED_LOGS', True)
    logger = app_logging.setup_logger('console-test-' + uuid.uuid4().hex)
    logger.propagate = False
    try:
        logger.info('Matched %s → Camera B · Δt=180', 'GJ01AB1234')
        text = output.getvalue().decode(encoding)
        assert 'GJ01****34' in text and 'GJ01AB1234' not in text
        assert ('\\u2192' if encoding == 'cp1252' else '→') in text
        assert ('\\u0394' if encoding == 'cp1252' else 'Δ') in text
        assert capsys.readouterr().err == ''
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        stream.close()
        logging.Logger.manager.loggerDict.pop(logger.name, None)


def test_narrow_console_keeps_exception_evidence_redacted(monkeypatch, capsys):
    output = io.BytesIO()
    stream = io.TextIOWrapper(output, encoding='cp1252', errors='strict')
    handler = app_logging.EncodingSafeStreamHandler(stream)
    handler.setFormatter(app_logging.PrivacyRedactingFormatter('%(message)s'))
    monkeypatch.setattr(app_logging.settings, 'ENABLE_REDACTED_LOGS', True)
    logger = logging.Logger('exception-console-test')
    logger.addHandler(handler)
    try:
        try:
            raise ValueError('GJ01AB1234 → unreadable')
        except ValueError:
            logger.exception('Fusion failed')
        text = output.getvalue().decode('cp1252')
        assert 'ValueError: GJ01****34 \\u2192 unreadable' in text
        assert 'GJ01AB1234' not in text
        assert capsys.readouterr().err == ''
    finally:
        handler.close()
        stream.close()
