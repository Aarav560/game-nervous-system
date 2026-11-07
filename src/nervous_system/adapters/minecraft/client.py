"""Localhost JSON-lines client for bridge/minecraft/bridge.js."""
from __future__ import annotations

import json
import socket
import time


class BridgeClient:
    def __init__(self, port: int, token: str, host: str = "127.0.0.1", timeout: float = 10.0):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise PermissionError("the bridge socket is localhost-only")
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setblocking(False)
        self.buf = b""
        self.state: dict | None = None
        self.acks: list[dict] = []
        self.events: list[dict] = []
        self.closed = False
        self.send({"cmd": "hello", "token": token})
        deadline = time.time() + timeout
        while time.time() < deadline:
            for m in self.poll():
                if m.get("type") == "hello" and m.get("ok"):
                    return
            time.sleep(0.02)
        raise ConnectionError("bridge did not accept the token")

    def send(self, obj: dict) -> None:
        if self.closed:
            return
        self.sock.setblocking(True)
        try:
            self.sock.sendall((json.dumps(obj) + "\n").encode())
        finally:
            self.sock.setblocking(False)

    def poll(self) -> list[dict]:
        out = []
        while True:
            try:
                chunk = self.sock.recv(65536)
            except BlockingIOError:
                break
            except OSError:
                self.closed = True
                break
            if not chunk:
                self.closed = True
                break
            self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("type") == "state":
                self.events.extend(m.get("events", []))   # bridge events, kept across skipped snapshots
                self.state = m
            else:
                self.acks.append(m)
            out.append(m)
        return out

    def wait_state(self, timeout: float = 10.0) -> dict:
        deadline = time.time() + timeout
        while self.state is None and time.time() < deadline:
            self.poll()
            time.sleep(0.02)
        if self.state is None:
            raise TimeoutError("no state from bridge")
        return self.state

    def wait_ack(self, cmd: str, timeout: float = 5.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.poll()
            for a in self.acks:
                if a.get("cmd") == cmd:
                    self.acks.remove(a)
                    return a
            time.sleep(0.02)
        return None

    def close(self) -> None:
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass
