import type { AnswerResponse, Subject } from "@/types/assistant";

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

type StreamOptions = {
  subject: Subject;
  question: string;
  passage: string;
  signal: AbortSignal;
  onDelta: (text: string) => void;
};

type StreamEvent =
  | { type: "start"; request_id: string; response_type: string }
  | { type: "delta"; text: string }
  | ({ type: "done" } & AnswerResponse)
  | { type: "error"; detail: string }
  | { type: "ping" };

export async function streamAssistant({
  subject,
  question,
  passage,
  signal,
  onDelta,
}: StreamOptions): Promise<AnswerResponse> {
  let response: Response;

  try {
    response = await fetch(`${API_URL}/answer/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, question, passage }),
      signal,
    });
  } catch (error) {
    if (signal.aborted) throw error;

    throw new Error(
      "Cannot connect to the assistant. Check FastAPI and the allowed frontend address.",
    );
  }

  if (!response.ok) {
    const data = await response.json().catch(() => null);
    const detail = data?.detail;

    throw new Error(
      typeof detail === "string"
        ? detail
        : `Request failed (${response.status}).`,
    );
  }

  if (!response.body) {
    throw new Error("Streaming is unavailable in this response.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: AnswerResponse | null = null;

  function processLine(line: string) {
    if (!line.trim()) return;

    const event = JSON.parse(line) as StreamEvent;

    if (event.type === "delta") {
      if (typeof event.text !== "string") {
        throw new Error("Invalid streamed text.");
      }

      onDelta(event.text);
    } else if (event.type === "error") {
      throw new Error(event.detail);
    } else if (event.type === "done") {
      if (
        typeof event.answer !== "string" ||
        typeof event.duration_seconds !== "number"
      ) {
        throw new Error("Invalid completion response.");
      }

      result = event;
    }
  }

  try {
    while (true) {
      const { value, done } = await reader.read();

      if (done) {
        buffer += decoder.decode();

        if (buffer.trim()) processLine(buffer);
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      let newline = buffer.indexOf("\n");

      while (newline !== -1) {
        processLine(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
        newline = buffer.indexOf("\n");
      }
    }

    if (!result) {
      throw new Error(
        "The connection ended before the answer completed.",
      );
    }

    return result;
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}