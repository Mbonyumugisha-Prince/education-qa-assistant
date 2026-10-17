"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Copy, LoaderCircle, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import { Button } from "@/components/ui/button";
import { subjectLabel } from "@/lib/subjects";
import type { ChatMessage } from "@/types/assistant";

function CopyButton({ text }: { text: string }) {
  const [status, setStatus] = useState<
    "idle" | "copied" | "error"
  >("idle");

  useEffect(() => {
    if (status === "idle") return;

    const timer = setTimeout(() => setStatus("idle"), 2000);
    return () => clearTimeout(timer);
  }, [status]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setStatus("copied");
    } catch {
      setStatus("error");
    }
  }

  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      onClick={copy}
      className="text-muted-foreground"
    >
      {status === "copied" ? <Check /> : <Copy />}

      {status === "copied"
        ? "Copied"
        : status === "error"
          ? "Copy failed"
          : "Copy"}
    </Button>
  );
}

function formatMath(text: string) {
  return text
    .split(/(```[\s\S]*?```)/g)
    .map((part) => {
      if (part.startsWith("```")) return part;

      return part
        .replace(
          /\\\[([\s\S]*?)\\\]/g,
          (_, math: string) => `\n$$\n${math}\n$$\n`,
        )
        .replace(
          /\\\(([\s\S]*?)\\\)/g,
          (_, math: string) => `$${math}$`,
        );
    })
    .join("");
}

type Props = {
  messages: ChatMessage[];
  busy: boolean;
  elapsed: number;
};

export function MessageList({
  messages,
  busy,
  elapsed,
}: Props) {
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages, busy]);

  if (!messages.length && !busy) {
    return (
      <div className="flex min-h-72 flex-col items-center justify-center px-4 text-center">
        <Sparkles
          aria-hidden="true"
          className="mb-5 size-8 text-emerald-700"
        />

        <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          A good question starts here.
        </h2>

        <p className="mt-3 max-w-sm text-sm leading-7 text-muted-foreground">
          Choose a subject and ask a question. You can also say hello.
        </p>
      </div>
    );
  }

  return (
    <section aria-label="Chat messages" className="space-y-7">
      {messages.map((message) => {
        const isUser = message.role === "user";

        return (
          <article
            key={message.id}
            className={isUser ? "flex justify-end" : "w-full"}
          >
            <div
              className={
                isUser
                  ? "w-fit min-w-24 max-w-[90%] rounded-2xl bg-muted px-4 py-3 sm:max-w-[80%]"
                  : "w-full min-w-0"
              }
            >
              <div className="mb-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                <span className="font-medium text-foreground">
                  {isUser ? "You" : "Education Assistant"}
                </span>

                <span>· {subjectLabel(message.subject)}</span>
              </div>

              {isUser ? (
                <>
                  <p className="whitespace-pre-wrap break-words text-sm leading-7">
                    {message.content}
                  </p>

                  {message.passage && (
                    <details className="mt-3 text-sm">
                      <summary className="cursor-pointer text-muted-foreground">
                        Science passage
                      </summary>

                      <p className="mt-2 whitespace-pre-wrap break-words leading-7">
                        {message.passage}
                      </p>
                    </details>
                  )}
                </>
              ) : (
                <>
                  <div className="chat-markdown">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {formatMath(message.content)}
                    </ReactMarkdown>
                  </div>

                  <div className="mt-3 flex flex-wrap items-center gap-3">
                    <CopyButton text={message.content} />

                    {typeof message.duration === "number" && (
                      <span className="text-xs text-muted-foreground">
                        {message.duration.toFixed(1)} seconds
                      </span>
                    )}
                  </div>
                </>
              )}
            </div>
          </article>
        );
      })}

      {busy && (
        <div
          role="status"
          className="flex items-start gap-3 text-sm text-muted-foreground"
        >
          <LoaderCircle
            aria-hidden="true"
            className="mt-0.5 size-4 shrink-0 animate-spin motion-reduce:animate-none"
          />

          <div>
            <p>Working on your question · {elapsed}s</p>
            <p className="mt-1 text-xs">
              Responses may take a minute or longer.
            </p>
          </div>
        </div>
      )}

      <div ref={bottom} />
    </section>
  );
}