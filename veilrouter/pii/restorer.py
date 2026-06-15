from __future__ import annotations

from veilrouter.pii.placeholders import PLACEHOLDER_RE


def restore_text(text: str, placeholder_to_original: dict[str, str]) -> str:
    return PLACEHOLDER_RE.sub(lambda match: placeholder_to_original.get(match.group(0), match.group(0)), text)


class StreamRestorer:
    def __init__(self, placeholder_to_original: dict[str, str], *, max_buffer: int = 256) -> None:
        self.placeholder_to_original = placeholder_to_original
        self.max_buffer = max_buffer
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        return self._drain(final=False)

    def finish(self) -> str:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> str:
        output: list[str] = []
        while self._buffer:
            start = self._buffer.find("[")
            if start < 0:
                if final:
                    output.append(self._buffer)
                    self._buffer = ""
                else:
                    output.append(self._buffer)
                    self._buffer = ""
                break
            if start > 0:
                output.append(self._buffer[:start])
                self._buffer = self._buffer[start:]
                continue
            end = self._buffer.find("]")
            if end < 0:
                if final:
                    output.append(self._buffer)
                    self._buffer = ""
                elif len(self._buffer) > self.max_buffer:
                    output.append(self._buffer[0])
                    self._buffer = self._buffer[1:]
                break
            else:
                token = self._buffer[: end + 1]
                if PLACEHOLDER_RE.fullmatch(token):
                    output.append(self.placeholder_to_original.get(token, token))
                else:
                    output.append(token)
                self._buffer = self._buffer[end + 1 :]
        return "".join(output)
