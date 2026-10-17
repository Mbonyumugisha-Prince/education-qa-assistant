"use client";

import { BookOpen, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { subjects } from "@/lib/subjects";
import type { Subject } from "@/types/assistant";

type Props = {
  subject: Subject;
  busy: boolean;
  onSelect: (subject: Subject) => void;
  onNewChat: () => void;
};

export function SubjectSidebar({
  subject,
  busy,
  onSelect,
  onNewChat,
}: Props) {
  return (
    <aside className="border-b bg-background p-5 lg:sticky lg:top-0 lg:h-dvh lg:w-64 lg:shrink-0 lg:overflow-y-auto lg:border-r lg:border-b-0">
      <div className="mb-6 flex items-center gap-3 font-semibold">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-emerald-700 text-white">
          <BookOpen className="size-5" />
        </span>

        <span>Education Assistant</span>
      </div>

      <Button
        type="button"
        variant="outline"
        className="mb-7 min-h-12 w-full justify-start gap-3 px-4 py-2 text-[13px]"
        disabled={busy}
        onClick={onNewChat}
      >
        <Plus />
        New chat
      </Button>

      <p className="mb-3 text-xs font-medium tracking-widest text-muted-foreground">
        SUBJECTS
      </p>

      <nav
        aria-label="Select a subject"
        className="flex flex-col gap-3 pb-2"
      >
        {subjects.map((item) => (
          <Button
            key={item.id}
            type="button"
            variant={subject === item.id ? "secondary" : "ghost"}
            className="min-h-12 w-full justify-start whitespace-normal px-4 py-2 text-left text-[13px]"
            aria-pressed={subject === item.id}
            disabled={busy}
            onClick={() => onSelect(item.id)}
          >
            {item.label}
          </Button>
        ))}
      </nav>

      <p className="mt-8 hidden text-xs leading-6 text-muted-foreground lg:block">
        Explore ideas one question at a time. Verify important answers.
      </p>
    </aside>
  );
}