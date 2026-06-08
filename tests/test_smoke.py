from io import BytesIO

from fastapi.testclient import TestClient
from docx import Document

from app.main import app

client = TestClient(app)


def _make_docx_bytes(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()


def test_healthcheck():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_json_analyze_accepts_id_alias():
    response = client.post(
        '/analyze',
        json={
            'documents': [
                {'id': 'doc1', 'text': 'abc abc abc'},
                {'id': 'doc2', 'text': 'abc abc xyz'},
            ],
            'method': 'ngram_jaccard',
        },
    )
    assert response.status_code == 200
    assert response.json()['document_count'] == 2


def test_json_analyze_returns_experiment_metrics_order_independent():
    response = client.post(
        '/analyze',
        json={
            'documents': [
                {'id': 'doc1', 'text': 'alpha beta gamma'},
                {'id': 'doc2', 'text': 'alpha beta gamma'},
                {'id': 'doc3', 'text': 'delta epsilon zeta'},
            ],
            'method': 'ngram_jaccard',
            'threshold': 0.5,
            'ground_truth': {
                'pairs': [
                    {'left_id': 'doc2', 'right_id': 'doc1', 'expected_match': True, 'scenario': 'duplicate'},
                    {'left_id': 'doc1', 'right_id': 'doc3', 'expected_match': False, 'scenario': 'different'},
                ]
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['experiment_metrics']['tp'] == 1
    assert payload['experiment_metrics']['tn'] == 1
    assert payload['experiment_metrics']['precision'] == 1
    assert payload['experiment_metrics']['recall'] == 1
    assert payload['summary']['method_specific_metrics']['average_shared_ngrams'] >= 0


def test_upload_analysis_docx_collection():
    files = [
        ('files', ('first.docx', _make_docx_bytes('совпадение текста в первом документе'), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')),
        ('files', ('second.docx', _make_docx_bytes('совпадение текста во втором документе'), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')),
    ]
    response = client.post(
        '/analyze-upload',
        data={'method': 'ngram_jaccard', 'threshold': '0.1', 'ngram_size': '2', 'shingle_size': '5', 'top_k': '5', 'query_text': ''},
        files=files,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['analysis']['document_count'] == 2
    assert len(payload['uploaded_documents']) == 2


def test_upload_analysis_accepts_ground_truth_json():
    files = [
        ('files', ('test1_a.txt', b'alpha beta gamma', 'text/plain')),
        ('files', ('test1_b.txt', b'alpha beta gamma', 'text/plain')),
    ]
    ground_truth = b'{"pairs":[{"left_id":"test1_b","right_id":"test1_a","expected_match":true,"scenario":"duplicate"}]}'
    response = client.post(
        '/analyze-upload',
        data={'method': 'ngram_jaccard', 'threshold': '0.5', 'ngram_size': '2', 'shingle_size': '5', 'top_k': '5', 'query_text': ''},
        files=[*files, ('ground_truth_file', ('ground_truth.json', ground_truth, 'application/json'))],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['analysis']['experiment_metrics']['tp'] == 1
    assert payload['analysis']['pairwise'][0]['metadata']['experiment_outcome'] == 'TP'
