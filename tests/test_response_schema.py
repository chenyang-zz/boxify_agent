from app.interfaces.schemas.base import Response


def test_success_response_uses_code_field():
    response = Response.success(data={"ok": True})

    assert response.model_dump() == {
        "code": 200,
        "msg": "success",
        "data": {"ok": True},
    }
