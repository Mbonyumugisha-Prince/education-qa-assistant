"use client";

import { useState } from "react";
import { AlertCircle } from "lucide-react";

import { useChat } from "@/hooks/use-chat";
import { subjectLabel } from "@/lib/subjects";
import type { Subject } from "@/types/assistant";

import { SubjectSidebar } from "./subject-sidebar";
import { MessageList } from "./message-list";
import { ChatComposer } from "./chat-composer";

export function EducationChat() {
  const [subject, setSubject] = useState<Subject>("mathematics");
  const [chatKey, setChatKey] = useState(0);
  const chat = useChat();

  function newChat() {
    if (chat.busy) return;

    chat.clear();
    setChatKey((value) => value + 1);
  }

  return (
    <div className="min-h-dvh bg-background lg:flex lg:h-dvh lg:overflow-hidden">
      <SubjectSidebar
        subject={subject}
        busy={chat.busy}
        onSelect={setSubject}
        onNewChat={newChat}
      />

      <main className="flex min-w-0 flex-1 flex-col lg:h-dvh">
        <header className="shrink-0 border-b px-5 py-4 sm:px-8">
          <h1 className="text-sm font-semibold">
            {subjectLabel(subject)}
          </h1>

          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            Each question is answered independently.
          </p>
        </header>

        <div className="min-h-0 flex-1 lg:overflow-y-auto">
          <div className="mx-auto w-full max-w-3xl px-5 py-7 sm:px-8">
            <MessageList
              messages={chat.messages}
              busy={chat.busy}
              elapsed={chat.elapsed}
            />
          </div>
        </div>

        <div className="sticky bottom-0 z-10 shrink-0 bg-background px-5 pb-4 pt-3 sm:px-8 lg:static">
          <div className="mx-auto w-full max-w-3xl space-y-3">
            {subject === "history" && (
              <p className="rounded-xl border bg-muted/30 px-4 py-3 text-xs leading-6 text-muted-foreground">
                History displays selected textbook excerpts. They may
                not answer every part of your question.
              </p>
            )}

            {chat.error && (
              <div
                role="alert"
                className="flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-800"
              >
                <AlertCircle className="mt-0.5 size-4 shrink-0" />
                <p>{chat.error}</p>
              </div>
            )}

            <ChatComposer
              key={chatKey}
              subject={subject}
              busy={chat.busy}
              onSend={chat.send}
            />

            <p className="text-center text-[11px] leading-5 text-muted-foreground">
              Answers can contain errors. Verify important facts,
              calculations, and generated code.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}