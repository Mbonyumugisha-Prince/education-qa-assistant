"use client";

import { useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { ArrowUp, LoaderCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { Subject } from "@/types/assistant";

type Props = {
  subject: Subject;
  busy: boolean;
  onSend: (
    subject: Subject,
    question: string,
    passage: string,
  ) => Promise<boolean>;
};

export function ChatComposer({ subject, busy, onSend }: Props) {
  const [question, setQuestion] = useState("");
  const formRef = useRef<HTMLFormElement>(null);
  const submitting = useRef(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const draft = question;

    if (busy || submitting.current || !draft.trim()) return;

    submitting.current = true;
    setQuestion("");

    try {
      const success = await onSend(subject, draft.trim(), "");

      if (!success) {
        setQuestion(draft);
      }
    } catch {
      setQuestion(draft);
    } finally {
      submitting.current = false;
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key !== "Enter" ||
      event.shiftKey ||
      event.nativeEvent.isComposing ||
      event.nativeEvent.keyCode === 229
    ) {
      return;
    }

    event.preventDefault();

    if (
      event.repeat ||
      busy ||
      submitting.current ||
      !question.trim()
    ) {
      return;
    }

    formRef.current?.requestSubmit();
  }

  return (
    <form
      ref={formRef}
      onSubmit={submit}
      className="w-full rounded-2xl border bg-background p-4 shadow-sm has-[:focus-visible]:ring-1 has-[:focus-visible]:ring-ring"
    >
      <fieldset
        disabled={busy}
        className="m-0 w-full min-w-0 space-y-4 border-0 p-0"
      >
        <label htmlFor="chat-question" className="sr-only">
          Your question
        </label>

        <textarea
          id="chat-question"
          required
          maxLength={12000}
          rows={3}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={handleKeyDown}
          aria-describedby="composer-hint"
          placeholder="Ask a question or say hello..."
          className="block w-full resize-y border-0 bg-transparent p-1 text-sm leading-7 shadow-none outline-none focus:border-0 focus:outline-none focus:ring-0 focus-visible:outline-none focus-visible:ring-0"
        />

        <div className="flex w-full items-center gap-4">
          <span
            id="composer-hint"
            className="text-xs text-muted-foreground"
          >
            Enter to send · Shift + Enter for a new line
          </span>

          <Button
            type="submit"
            disabled={busy || !question.trim()}
            className="ml-auto shrink-0 bg-emerald-700 text-white hover:bg-emerald-800"
          >
            {busy ? (
              <LoaderCircle className="animate-spin motion-reduce:animate-none" />
            ) : (
              <ArrowUp />
            )}

            {busy ? "Working" : "Send"}
          </Button>
        </div>
      </fieldset>
    </form>
  );
}