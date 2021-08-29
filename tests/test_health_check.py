import json


def test_health(app, client):
    res = client.get('/health_check')
    assert res.status_code == 200
    expected = "The service is up and alive"
    assert expected == json.loads(res.get_data(as_text=True))


def test_health_with_invalid_endpoint(app, client):
    res = client.get('/health')
    assert res.status_code == 404
