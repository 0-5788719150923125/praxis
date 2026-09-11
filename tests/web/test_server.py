"""APIServer lifecycle (praxis/web/server.py): port selection and shutdown."""

import socket

import requests

from praxis.web import APIServer


def test_a_busy_port_rolls_to_a_higher_one(mock_generator):
    """The port is chosen in ``__init__``, so holding one open is enough."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        server = APIServer(generator=mock_generator, port=taken)
    assert server.port > taken


def test_stop_closes_the_listener(mock_generator, free_port, monkeypatch):
    """stop() must free the port. Setting a flag the socket thread never
    reads left every stopped server listening for the life of the process."""
    from praxis.web.app import app

    # start() writes the singleton app's config; give the session's back after.
    saved = dict(app.config)
    monkeypatch.setattr(app, "debug", app.debug)

    server = APIServer(generator=mock_generator, port=free_port)
    try:
        server.start()
        ping = requests.get(f"http://localhost:{server.port}/api/ping", timeout=5)
        assert ping.status_code == 200
    finally:
        server.stop()
        app.config.clear()
        app.config.update(saved)

    server.server_thread.join(5)
    assert not server.server_thread.is_alive()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        assert probe.connect_ex(("127.0.0.1", server.port)) != 0
