"use client";

import { useEffect, useRef, useState } from "react";
import { streamAssistant } from "@/lib/api";
import type { ChatMessage, Subject } from "@/types/assistant";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");

  const inFlight = useRef(false);
  const mounted = useRef(true);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    mounted.current = true;

    return () => {
      mounted.current = false;
      controllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!busy) return;

    const started = Date.now();
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - started) / 1000));
    }, 1000);

    return () => clearInterval(timer);
  }, [busy]);

  async function send(
    subject: Subject,
    question: string,
    passage: string,
  ): Promise<boolean> {
    const text = question.trim();

    if (!text || inFlight.current) return false;

    inFlight.current = true;
    setBusy(true);
    setElapsed(0);
    setError("");

    const userId = crypto.randomUUID();
    const assistantId = crypto.randomUUID();
    const controller = new AbortController();
    controllerRef.current = controller;

    let streamedText = "";

    const userMessage: ChatMessage = {
      id: userId,
      role: "user",
      content: text,
      subject,
      ...(passage.trim() ? { passage: passage.trim() } : {}),
    };

    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      subject,
    };

    setMessages((previous) => [
      ...previous,
      userMessage,
      assistantMessage,
    ]);

    try {
      const result = await streamAssistant({
        subject,
        question: text,
        passage: passage.trim(),
        signal: controller.signal,

        onDelta(chunk) {
          streamedText += chunk;

          if (!mounted.current) return;

          const content = streamedText;

          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantId
                ? { ...message, content }
                : message,
            ),
          );
        },
      });

      if (mounted.current) {
        setMessages((previous) =>
          previous.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  content: result.answer,
                  duration: result.duration_seconds,
                }
              : message,
          ),
        );
      }

      return true;
    } catch (caught) {
      if (mounted.current) {
        if (streamedText) {
          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content:
                      streamedText +
                      "\n\n[Response interrupted; this answer is incomplete.]",
                  }
                : message,
            ),
          );
        } else {
          setMessages((previous) =>
            previous.filter(
              (message) =>
                message.id !== userId &&
                message.id !== assistantId,
            ),
          );
        }

        setError(
          caught instanceof Error
            ? caught.message
            : "The request failed. Please try again.",
        );
      }

      return false;
    } finally {
      controllerRef.current = null;
      inFlight.current = false;

      if (mounted.current) setBusy(false);
    }
  }

  function clear() {
    if (inFlight.current) return;

    setMessages([]);
    setError("");
  }

  return { messages, busy, elapsed, error, send, clear };
}